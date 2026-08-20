# SPDX-License-Identifier: Apache-2.0
"""The cross-stream site is a real frozen zero-init nn.Linear wrapped by stock LoRA.

Covers issue testing-plan item 3 (cross-stream no-op at init) and the core
equivalence that lets SR drop its custom cross-stream wrapper: stock
``lora.Linear`` on the zero-init frozen linear reduces to ``B(A(x)) * scaling``
(because ``base_layer(x) == 0``), which is exactly the rank-R cross-stream
injection. Also pins that the zero-init survives HF ``post_init()`` (the
``_init_weights`` override) — the bug that would otherwise pollute the adapter
stream.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from shadow_residual.shadow_residual.cross_stream import CrossStream


def _build_tiny_sr_config():
    """Tiny CPU-only SR config for cross-stream / model post_init tests."""
    from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
    from shadow_residual.shadow_residual.config_helpers import set_shadow_residual

    cfg = ShadowResidualConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        num_adapters=0,
        max_lora_rank=4,
        switch_head_dim=16,
    )
    set_shadow_residual(cfg, enabled=True)
    return cfg


def test_cross_stream_is_frozen_zero_linear():
    cs = CrossStream(16)
    assert isinstance(cs, nn.Linear)
    assert cs.in_features == cs.out_features == 16
    assert bool((cs.weight == 0).all()), "cross-stream base weight must be zero-init"
    assert cs.weight.requires_grad is False, "cross-stream base weight must be frozen"


def test_cross_stream_forward_is_zero_when_unwrapped():
    cs = CrossStream(16)
    x = torch.randn(2, 5, 16)
    out = cs(x)
    assert torch.equal(out, torch.zeros_like(out)), (
        "bare cross-stream must contribute exactly zero (no-op site)"
    )


def test_cross_stream_ignores_extra_args():
    """PEFT may strip variant kwargs; the base forward must tolerate extras."""
    cs = CrossStream(8)
    x = torch.randn(1, 3, 8)
    out = cs(x, some_variant_kwarg=[1], some_future_kwarg=True)
    assert torch.equal(out, torch.zeros_like(out))


def test_zero_init_survives_full_model_post_init():
    """The _init_weights override must re-zero cross_stream after post_init(),
    which otherwise randomizes every nn.Linear."""
    from shadow_residual.shadow_residual import ShadowResidualForCausalLM

    model = ShadowResidualForCausalLM(_build_tiny_sr_config())
    for i, layer in enumerate(model.model.layers):
        w = layer.cross_stream.weight
        assert float(w.abs().sum()) == 0.0, f"layer {i} cross_stream not zero after post_init"
        assert w.requires_grad is False, f"layer {i} cross_stream not frozen after post_init"


def test_stock_lora_on_zero_linear_is_pure_delta():
    """Stock lora.Linear on the zero-init frozen CrossStream reduces to
    B(A(x)) * scaling — the rank-R cross-stream injection with no base
    contribution. This is what lets SR use a plain stock LoRA target for
    cross_stream instead of a custom wrapper (base_layer(x) == 0).
    """
    from peft.tuners.lora.layer import Linear as StockLoraLinear
    from peft import LoraConfig

    H, R = 16, 4
    torch.manual_seed(0)
    cfg = LoraConfig(r=R, lora_alpha=8, target_modules=["cross_stream"])

    stock = StockLoraLinear(CrossStream(H), "default", cfg, r=R, lora_alpha=8)

    with torch.no_grad():
        # lora_B is zero-init; perturb so the delta is nonzero and the test
        # actually exercises the projection.
        stock.lora_B["default"].weight.normal_(std=0.1)

    x = torch.randn(2, 5, H)
    out_stock = stock(x)

    # Hand-computed pure delta: B(A(x)) * (alpha / r), base contributes zero.
    A = stock.lora_A["default"]
    B = stock.lora_B["default"]
    scaling = stock.scaling["default"]
    expected = B(A(x)) * scaling
    torch.testing.assert_close(out_stock, expected, atol=1e-6, rtol=1e-5)
