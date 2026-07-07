# SPDX-License-Identifier: Apache-2.0
"""Tests for the recompute-stable aLoRA variant (activation-checkpoint fix).

Stock PEFT ``ALoraLinearVariant.forward`` (variants.py) applies the LoRA delta
via a boolean mask-select + in-place scatter, so the number of tensors saved
for autograd depends on ``mask.sum()`` (the aLoRA invocation offset). Under FSDP
activation-checkpoint recompute the mask population differs from the original
forward, so a different NUMBER of tensors is saved on recompute and
``torch.utils.checkpoint`` raises
``CheckpointError: a different number of tensors was saved``. Confirmed on the
cluster: the divergence frame was overwhelmingly ``variants.py:611`` and the
saved-tensor count dropped 136 -> 125 on recompute.

:class:`RecomputeStableALoraVariant` computes the delta over all tokens with
fixed shapes and gates it with a ``[B, T, 1]`` float mask — identical
saved-tensor count on every pass regardless of offset, numerically identical to
stock. ``ShadowResidualLora.resolve_lora_variant`` installs it whenever
``alora_invocation_tokens`` is set.

The FSDP recompute crash itself is not reproducible single-process; the
save-count-parity test below is the in-process regression guard for the
mechanism. CPU-only, fast.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.variants import ALoraLinearVariant

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig
from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual._stream_gated_linear import (
    _StreamGatedLinear,
)
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
)
from shadow_residual.shadow_residual.cross_stream import CrossStream
from shadow_residual.peft_shadow_residual import CrossStreamLora
from shadow_residual.peft_shadow_residual.stream_gated_lora import (
    RecomputeStableALoraVariant,
    ShadowResidualLora,
)


class _VariantStub:
    """Minimal stand-in exposing the dicts the variant.forward reads."""

    def __init__(self, in_features=16, rank=4, out_features=16, dropout=0.0, scaling=2.0):
        self.lora_A = {"default": torch.nn.Linear(in_features, rank, bias=False)}
        self.lora_B = {"default": torch.nn.Linear(rank, out_features, bias=False)}
        self.lora_dropout = {"default": torch.nn.Dropout(dropout)}
        self.scaling = {"default": scaling}


def _count_saved(fn, module, x, result, offsets):
    n = 0

    def pack(t):
        nonlocal n
        n += 1
        return t

    xx = x.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(module, "default", xx, result.clone(), alora_offsets=offsets)
        out.sum().backward()
    return n


def _build_alora_sr_model():
    cfg = GraniteSwitchConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_adapters=0,
        max_lora_rank=8, switch_head_dim=16,
    )
    set_shadow_residual(config=cfg, enabled=True)
    model = ShadowResidualForCausalLM(cfg)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "cross_stream"],
        lora_dropout=0.0,
    )
    lora_cfg.alora_invocation_tokens = [5, 6]
    lora_cfg._register_custom_module(
        {CrossStream: CrossStreamLora, _StreamGatedLinear: ShadowResidualLora}
    )
    return get_peft_model(model, lora_cfg)


def test_resolve_installs_recompute_stable_variant():
    """When alora_invocation_tokens is set, the SR wrapper must install the
    recompute-stable variant, not stock ALoraLinearVariant."""
    model = _build_alora_sr_model()
    wrappers = [m for m in model.modules() if isinstance(m, ShadowResidualLora)]
    assert wrappers, "no ShadowResidualLora installed"
    for w in wrappers:
        assert w.lora_variant, f"{w} has no lora_variant"
        for v in w.lora_variant.values():
            assert isinstance(v, RecomputeStableALoraVariant), (
                f"expected RecomputeStableALoraVariant, got {type(v).__name__}"
            )


@pytest.mark.parametrize("offsets", [[5], [0], [8], [3], None])
def test_save_count_constant_across_offsets(offsets):
    """The whole point of the fix: saved-tensor count must NOT depend on the
    aLoRA offset / mask population. A varying count is what diverges under
    checkpoint recompute. Stock varies with the masked select; ours is fixed."""
    torch.manual_seed(0)
    stub = _VariantStub()
    x = torch.randn(1, 8, 16)
    result = torch.randn(1, 8, 16)

    count = _count_saved(RecomputeStableALoraVariant.forward, stub, x, result, offsets)
    # Baseline offset with a different mask population; count must be identical.
    baseline = _count_saved(RecomputeStableALoraVariant.forward, stub, x, result, [4])
    assert count == baseline, (
        f"save count {count} != baseline {baseline} for offsets={offsets} — "
        "a varying count is exactly what breaks checkpoint recompute"
    )


def test_numerically_equivalent_to_stock():
    """Masked positions get delta*0, so the result must match stock aLoRA
    exactly (within fp tolerance) — the frozen-base / gating semantics are
    unchanged, only the autograd-graph shape is made recompute-stable."""
    torch.manual_seed(0)
    stub = _VariantStub()
    x = torch.randn(1, 8, 16)
    result = torch.randn(1, 8, 16)
    for offsets in ([5], [0], [8], [3], None):
        a = ALoraLinearVariant.forward(stub, "default", x, result.clone(), alora_offsets=offsets)
        b = RecomputeStableALoraVariant.forward(stub, "default", x, result.clone(), alora_offsets=offsets)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_alora_sr_model_checkpoint_forward_backward_ok():
    """End-to-end: an aLoRA SR model under non-reentrant gradient checkpointing
    runs forward+backward without CheckpointError. (Single-process cannot
    reproduce the FSDP-specific divergence, but this exercises the gated path
    through the stable variant.)"""
    model = _build_alora_sr_model()
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    torch.manual_seed(0)
    ids = torch.tensor([[1, 2, 5, 6, 7, 8, 9, 10]])
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()  # must not raise

    lora_grads = [
        p.grad for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad
    ]
    assert any(g is not None and torch.any(g != 0) for g in lora_grads)


# --- CrossStreamLora: the SECOND copy of the aLoRA masked computation ---
# CrossStreamLora.forward has its own hand-rolled aLoRA gating (mirrors the
# variant). It previously built the mask only `if alora_offsets is not None`,
# so the original forward vs the activation-checkpoint recompute saved a
# different NUMBER of tensors when offsets differed across the checkpoint
# boundary under FSDP. This was the actual cluster CheckpointError (confirmed
# via PROBE_DUMP on a 4-GPU debug pod): the diverging tensors were the cross
# stream's [1,T,1] float mask + rank-r LoRA tensors. Fixed by always computing
# a fixed-shape [B,T,1] float mask (all-ones when no offsets). These guard the
# branch-free invariant in-process (the FSDP crash is not reproducible
# single-process).


def _build_cross_stream_lora(rank=8, dropout=0.0):
    from peft import LoraConfig as _LC
    base = CrossStream(64)
    cfg = _LC(r=rank, lora_alpha=16, target_modules=["cross_stream"], lora_dropout=dropout)
    cfg.alora_invocation_tokens = [5, 6]
    w = CrossStreamLora(base, "default", r=rank, lora_alpha=16, lora_dropout=dropout, config=cfg)
    torch.nn.init.normal_(w.lora_B["default"].weight, std=0.1)
    return w


@pytest.mark.parametrize("offsets", [None, [3, 3], [0, 5], [5, 0]])
def test_cross_stream_lora_save_count_constant(offsets):
    """CrossStreamLora must save the same number of autograd tensors whether
    or not alora_offsets is present — otherwise checkpoint recompute diverges."""
    torch.manual_seed(0)
    w = _build_cross_stream_lora()
    x = torch.randn(2, 5, 64)

    def count(offs):
        n = 0

        def pack(t):
            nonlocal n
            n += 1
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = w(x, alora_offsets=offs)
            out.sum().backward()
        return n

    baseline = count([4, 4])
    assert count(offsets) == baseline, (
        f"CrossStreamLora save count varies with offsets={offsets} "
        f"({count(offsets)} != {baseline}) — breaks checkpoint recompute"
    )


def test_cross_stream_lora_offsets_none_is_full_delta():
    """offsets=None must equal the plain full-delta add (mask all-ones), i.e.
    numerically identical to the pre-fix no-gating path."""
    torch.manual_seed(0)
    w = _build_cross_stream_lora()
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        out_none = w(x, alora_offsets=None)
        # All-active gated offsets (>= T) should match offsets=None exactly.
        out_full = w(x, alora_offsets=[5, 5])
    torch.testing.assert_close(out_none, out_full, rtol=1e-4, atol=1e-5)


# --- Inference path: grad-disabled uses the memory-efficient masked-select ---
# The fixed-shape full-token delta is recompute-stable but materialises a full
# [B, T, out] tensor — on a 32768-wide MLP projection that OOMs single-GPU 30B
# generation. Under torch.no_grad() (inference, no checkpointing) the variant
# falls back to the masked-select; it must be numerically identical to the
# training (grad-enabled) path.


def test_variant_inference_path_matches_training_path():
    stub = _VariantStub(in_features=64, rank=8, out_features=64)
    torch.nn.init.normal_(stub.lora_B["default"].weight, std=0.1)
    x = torch.randn(2, 5, 64)
    result = torch.randn(2, 5, 64)
    for offsets in ([3, 3], [0, 5], None):
        with torch.enable_grad():
            a = RecomputeStableALoraVariant.forward(
                stub, "default", x, result.clone(), alora_offsets=offsets
            )
        with torch.no_grad():
            b = RecomputeStableALoraVariant.forward(
                stub, "default", x, result.clone(), alora_offsets=offsets
            )
        torch.testing.assert_close(a.detach(), b, rtol=1e-4, atol=1e-5)


def test_cross_stream_inference_path_matches_training_path():
    w = _build_cross_stream_lora()
    x = torch.randn(2, 5, 64)
    for offsets in ([3, 3], [0, 5], None):
        with torch.enable_grad():
            a = w(x, alora_offsets=offsets)
        with torch.no_grad():
            b = w(x, alora_offsets=offsets)
        torch.testing.assert_close(a.detach(), b, rtol=1e-4, atol=1e-5)
