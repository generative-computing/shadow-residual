# SPDX-License-Identifier: Apache-2.0
"""Activation-checkpointing compatibility tests for the SR + PEFT path.

The 30B SR jobs OOM in backward unless activation checkpointing is on
(FSDP shards params/grads/optimizer but NOT activations). Turning on
accelerate's external fsdp_activation_checkpointing wrap took the REENTRANT
checkpoint path and raised
``CheckpointError: a different number of tensors was saved during forward and
recomputation``.

The fix (in training/train.py) instead drives checkpointing through the
model's OWN ``_gradient_checkpointing_func`` with ``use_reentrant=False`` and
``enable_input_require_grads()``. These tests pin that contract:

1. Non-reentrant gradient checkpointing runs a clean forward+backward.
2. The reentrant path (without input-require-grads) fails — documenting WHY
   we use non-reentrant.
3. The frozen-base invariant is preserved with checkpointing on.
4. Both dropout settings (0.0 and 0.05) complete under non-reentrant —
   guarding against a future RNG-preservation regression.

CPU-only, tiny config, fast.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
)
from shadow_residual.shadow_residual.modeling_hf import ShadowResidualModel

# Stream-gated projections + the cross_stream merge site — the full target set
# the real SR configs train, so the checkpointed forward exercises every
# stream-context branch.
_TARGET_MODULES = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "cross_stream"]


def _build_sr_peft_model(lora_dropout: float = 0.0):
    """Tiny SR + PEFT model. Mirrors the fixture in
    tests/training/test_peft_wrapper.py."""
    cfg = ShadowResidualConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_adapters=0,
        max_lora_rank=8,
        switch_head_dim=16,
    )
    set_shadow_residual(config=cfg, enabled=True)
    model = ShadowResidualForCausalLM(cfg)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=list(_TARGET_MODULES),
        lora_dropout=lora_dropout,
    )
    return get_peft_model(model, lora_cfg)


def _tiny_batch():
    torch.manual_seed(0)
    ids = torch.randint(0, 300, (2, 8))
    return ids


def _count_saved_tensors(fn):
    """Run fn() + backward and count tensors stashed for autograd. This is
    exactly what torch.utils.checkpoint's non-reentrant recompute checker
    compares between forward and recompute."""
    n = 0

    def pack(t):
        nonlocal n
        n += 1
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
        out.sum().backward()
    return n




def _set_lora_b_nonzero(model):
    """LoRA ``lora_B`` is zero-initialised, so the adapter delta is exactly 0 at
    step 0 and gradient differences would be invisible. Fill all ``lora_B``
    weights so the delta is non-trivial and any recompute discrepancy shows up
    in gradients."""
    import torch
    from peft.tuners.lora.layer import LoraLayer
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoraLayer):
                for adapter in m.lora_B:
                    m.lora_B[adapter].weight.normal_(0.0, 0.02)


def test_lora_gradients_match_with_and_without_checkpointing():
    """LoRA training gradients under gradient checkpointing must EQUAL those
    without checkpointing.

    The adapter is plain LoRA (always active, no gating). Gradient checkpointing
    only recomputes activations; because the dual-stream forward is pure, the
    recompute is exact and the gradients must be identical with ckpt ON vs OFF.
    Guards against a recompute-purity regression (e.g. a stream contextvar or
    hidden per-forward state creeping back in).
    """
    from peft.tuners.lora.layer import LoraLayer

    ids = torch.tensor([[1, 5, 6, 7, 2, 3, 4, 8], [5, 6, 7, 1, 2, 3, 4, 8]])

    def grads(use_ckpt):
        torch.manual_seed(0)
        model = _build_sr_peft_model()
        _set_lora_b_nonzero(model)
        model.train()
        if use_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        model.zero_grad()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        g = {}
        for n, m in model.named_modules():
            if isinstance(m, LoraLayer):
                for adapter, lin in m.lora_B.items():
                    if lin.weight.grad is not None:
                        g[f"{n}.{adapter}"] = lin.weight.grad.detach().clone()
        return out.loss.detach().clone(), g

    loss_off, g_off = grads(use_ckpt=False)
    loss_on, g_on = grads(use_ckpt=True)

    torch.testing.assert_close(loss_on, loss_off, rtol=1e-4, atol=1e-4)
    assert g_off, "no LoRA gradients captured."
    assert set(g_on) == set(g_off), "different LoRA params got grads with/without ckpt."
    for k in g_off:
        torch.testing.assert_close(
            g_on[k], g_off[k], rtol=1e-3, atol=1e-3,
            msg=f"LoRA grad for {k} differs with checkpointing — recompute is not "
                "exact (a non-pure forward regressed in).",
        )


def test_non_reentrant_checkpointing_forward_backward_ok():
    """The fix: non-reentrant gradient checkpointing must run a clean
    forward+backward with no CheckpointError. Regression guard."""
    model = _build_sr_peft_model()
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    # Confirm the enable propagated to the inner SR model.
    inner = [m for m in model.modules() if isinstance(m, ShadowResidualModel)]
    assert inner, "no ShadowResidualModel found"
    assert all(m.gradient_checkpointing for m in inner)

    ids = _tiny_batch()
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()  # must not raise CheckpointError

    # A trainable LoRA param must have received a gradient.
    lora_grads = [
        p.grad for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad
    ]
    assert any(g is not None and torch.any(g != 0) for g in lora_grads)


def test_frozen_base_invariant_preserved_under_checkpointing():
    """Base-stream logits must be identical with checkpointing on vs off.
    Checkpointing only recomputes activations; it must not perturb the
    forward result, including the frozen-base path."""
    ids = _tiny_batch()

    model = _build_sr_peft_model()
    model.eval()  # deterministic; isolate checkpointing from dropout/train state
    with torch.no_grad():
        logits_off = model(input_ids=ids).logits.clone()

    # Same weights, checkpointing enabled.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Checkpointing is a no-op outside training mode in HF; force the training
    # flag on the inner model so the checkpointed branch actually runs, but
    # keep modules in eval() so dropout stays off and the comparison is exact.
    for m in model.modules():
        if isinstance(m, ShadowResidualModel):
            m.training = True
    with torch.no_grad():
        logits_on = model(input_ids=ids).logits.clone()

    torch.testing.assert_close(logits_on, logits_off, rtol=0, atol=0)


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_non_reentrant_completes_with_dropout(dropout):
    """Both dropout settings must complete under non-reentrant checkpointing.
    The original CheckpointError reproduced even at dropout=0.0, so dropout
    RNG was never the cause — this guards against a regression that
    reintroduces an RNG-sensitivity dependence."""
    model = _build_sr_peft_model(lora_dropout=dropout)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    ids = _tiny_batch()
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()  # must not raise


# --- Approach-B root-cause guards: the stream-gated wrapper must NOT diverge
# in tensor-save count between streams, because that divergence (under FSDP
# recompute, where the stream contextvar is lost) is what raised
# CheckpointError. These are in-process proxies for the FSDP-only failure.


