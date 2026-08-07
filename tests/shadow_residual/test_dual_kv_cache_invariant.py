# SPDX-License-Identifier: Apache-2.0
"""Dual-KV-cache invariant test for the shadow-residual + PEFT path.

Phase 1 of the SR refactor dropped the final-step merge gate
``torch.where(adapter_indices > 0, h_adapt, h_base)``. This test exists
to prove that removing the gate did NOT collapse the two attention
streams: ``stream_context("base")`` must still compute K/V from a frozen
base path (no LoRA delta), and ``stream_context("adapter")`` must still
compute K/V with the LoRA delta firing — i.e. the two K projections
land in disjoint caches.

We run a single forward through a tiny SR + PEFT model, hook one layer's
``k_proj``, and capture the output once per stream context. Two
assertions:

1. Base K and adapter K disagree somewhere — proves the streams diverge.
2. autograd of the base K w.r.t. any ``lora_B`` parameter returns zero —
   proves the base stream is truly LoRA-independent.

CPU-only, tiny config, fast.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.shadow_residual import (
    ShadowResidualForCausalLM,
)
from shadow_residual.shadow_residual._stream_context import current_stream
from shadow_residual.shadow_residual._stream_gated_linear import (
    _StreamGatedLinear,
)
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
)
from shadow_residual.shadow_residual.cross_stream import CrossStream
from shadow_residual.peft_shadow_residual import CrossStreamLora
from shadow_residual.peft_shadow_residual.stream_gated_lora import (
    ShadowResidualLora,
)


@pytest.fixture
def tiny_sr_peft_model():
    cfg = ShadowResidualConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_adapters=1,
        adapter_token_ids=[250],
        adapter_substitute_token_ids=[260],
        adapter_names=["expert"],
        max_lora_rank=4,
        adapter_ranks=[4],
        switch_head_dim=16,
        lora_target_modules=["qkv_proj", "o_proj"],
    )
    set_shadow_residual(config=cfg, enabled=True)
    sr_model = ShadowResidualForCausalLM(cfg)

    # Production-equivalent dispatch: CrossStream → CrossStreamLora and
    # _StreamGatedLinear → ShadowResidualLora. Without the second, k_proj
    # would not gain a LoRA delta and the two streams would trivially
    # match — defeating the whole point of this test.
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["k_proj", "cross_stream"],
        lora_dropout=0.0,
    )
    lora_cfg._register_custom_module(
        {
            CrossStream: CrossStreamLora,
            _StreamGatedLinear: ShadowResidualLora,
        }
    )
    peft_model = get_peft_model(sr_model, lora_cfg)

    # Push k_proj's LoRA-B off zero so the adapter delta is non-trivial
    # (PEFT initializes lora_B to zero by default, which would make the
    # two streams agree exactly even with the delta wired in).
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "k_proj" in name and "lora_B" in name:
                p.data.normal_(mean=0.0, std=0.02)

    return peft_model


def _find_first_k_proj_lora_wrapper(peft_model) -> ShadowResidualLora:
    for name, m in peft_model.named_modules():
        if name.endswith(".k_proj") and isinstance(m, ShadowResidualLora):
            return m
    raise AssertionError("no ShadowResidualLora-wrapped k_proj found")


def test_streams_produce_disjoint_k_outputs(tiny_sr_peft_model):
    """Base-context K and adapter-context K must differ in at least one element.

    If the gate-removal had accidentally collapsed both streams to a
    single context, the hook would fire only once, or both fires would
    produce identical tensors. Either failure mode is caught here.
    """
    peft_model = tiny_sr_peft_model
    peft_model.eval()

    k_proj = _find_first_k_proj_lora_wrapper(peft_model)

    captured: list[tuple[str, torch.Tensor]] = []

    def hook(_module, _inputs, output):
        captured.append((current_stream(), output.detach().clone()))

    handle = k_proj.register_forward_hook(hook)
    try:
        # Use_cache=True triggers the dual-cache update path; it also
        # forces cross_stream_active=True (we have cross_stream wrapped).
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        with torch.no_grad():
            peft_model(input_ids=input_ids, use_cache=True)
    finally:
        handle.remove()

    # The hook should fire twice on every k_proj invocation in the
    # dual-stream path: once under "base", once under "adapter".
    streams = [s for s, _ in captured]
    assert "base" in streams, f"no base-stream k_proj forward observed: {streams}"
    assert "adapter" in streams, f"no adapter-stream k_proj forward observed: {streams}"

    base_k = next(t for s, t in captured if s == "base")
    adapt_k = next(t for s, t in captured if s == "adapter")
    assert base_k.shape == adapt_k.shape

    # Both streams pass the same normed input (granite SR shares the
    # input norm on layer 0), so any divergence is the LoRA delta. With
    # lora_B perturbed off zero above, the two K tensors must diverge.
    assert not torch.allclose(base_k, adapt_k), (
        "base-stream and adapter-stream K projections are identical — "
        "the gate-removal patch may have collapsed the two streams"
    )


def test_base_stream_k_independent_of_lora_params(tiny_sr_peft_model):
    """The base-context K must be LoRA-independent; the adapter-context K must NOT.

    Under ``stream_context("base")`` the LoRA delta is gated to zero inside
    ``ShadowResidualLora`` — so base K has no *gradient* through any
    ``lora_*`` parameter. Under ``stream_context("adapter")`` the delta
    fires, so adapter K *does* require grad.

    NOTE: the gate is applied by scaling the delta to 0.0, not by skipping
    the LoRA ops (that structural skip changed the autograd tensor-save
    count between streams and broke activation-checkpoint recompute — see
    test_activation_checkpointing.py). So base K still has
    ``requires_grad=True`` via the ``×0`` edge; the meaningful invariant is
    that its gradient w.r.t. every ``lora_*`` parameter is exactly zero.

    Asserting both directions makes the test meaningful: a fully-frozen
    model would also yield zero base-K LoRA grads, but it would also have
    ``adapt_k.requires_grad == False`` — which fails the positive control
    and reveals that the LoRA wiring isn't actually live.
    """
    peft_model = tiny_sr_peft_model
    peft_model.train()  # autograd needs grad-tracking parameters

    k_proj = _find_first_k_proj_lora_wrapper(peft_model)

    captured: list[tuple[str, torch.Tensor]] = []

    def hook(_module, _inputs, output):
        captured.append((current_stream(), output))

    handle = k_proj.register_forward_hook(hook)
    try:
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        peft_model(input_ids=input_ids, use_cache=True)
    finally:
        handle.remove()

    base_k = next(t for s, t in captured if s == "base")
    adapt_k = next(t for s, t in captured if s == "adapter")

    # Positive control: LoRA delta IS wired on the adapter side.
    assert adapt_k.requires_grad, (
        "adapter-stream K does not require grad — the LoRA delta isn't "
        "actually wired into the adapter path; the rest of this test "
        "would be vacuously true"
    )

    # The actual invariant: base-stream K carries zero gradient w.r.t. every
    # LoRA parameter. With adapter K confirmed live above, this proves base K
    # is LoRA-independent even though the ×0-gated edge keeps requires_grad set.
    lora_params = [p for n, p in peft_model.named_parameters() if "lora_" in n]
    assert lora_params, "no lora_ params found — model wiring changed"
    grads = torch.autograd.grad(
        base_k.sum(), lora_params, retain_graph=True, allow_unused=True
    )
    assert all(g is None or torch.all(g == 0) for g in grads), (
        "base-stream K has nonzero gradient w.r.t. a LoRA parameter — the "
        "delta leaked into the frozen-base path"
    )


def test_three_backward_steps_produce_nonzero_lora_grads(tiny_sr_peft_model):
    """Regression for the with-cross grad_norm=0 bug.

    Before the gate fix, the final-step ``torch.where(adapter_indices > 0, ...)``
    routed every token to ``h_base`` (since no control
    tokens fire in unconditional adapter training), severing autograd
    from every LoRA parameter — observed as ``grad_norm = 0`` on the
    Vela ``antonp-answerability-with-cross`` job.

    Run 3 backward steps on a toy batch and assert that at least one
    ``lora_B`` parameter received a non-zero gradient. If this regresses,
    Phase 1 is broken and on-cluster training will silently learn nothing.
    """
    peft_model = tiny_sr_peft_model
    peft_model.train()

    optimizer = torch.optim.SGD(
        [p for p in peft_model.parameters() if p.requires_grad], lr=1e-3,
    )

    seen_nonzero_grad = False
    for _ in range(3):
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long,
        )
        labels = input_ids.clone()
        out = peft_model(input_ids=input_ids, labels=labels)
        loss = out.loss
        assert loss.requires_grad, (
            "loss does not require grad — the model is fully frozen "
            "or autograd is severed before the head"
        )
        optimizer.zero_grad()
        loss.backward()

        for n, p in peft_model.named_parameters():
            if "lora_B" in n and p.grad is not None and p.grad.abs().max() > 0:
                seen_nonzero_grad = True
                break

        optimizer.step()

    assert seen_nonzero_grad, (
        "no lora_B parameter received a non-zero gradient over 3 steps — "
        "the merge gate may have been re-introduced or the SR forward "
        "is again severing autograd from the LoRA path"
    )
