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

from shadow_residual.shadow_residual.cross_stream import (
    CrossStream,
    CrossStreamLinear,
)


def _build_tiny_sr_config(cross_stream_taps=None):
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
    if cross_stream_taps is not None:
        cfg.cross_stream_taps = list(cross_stream_taps)
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


def test_all_configured_taps_are_frozen_zero_after_post_init():
    """Every tap named in ``cross_stream_taps`` — not just the legacy post-MLP one
    — must survive ``post_init()`` frozen and exactly zero. The ``_init_weights``
    override keys off ``isinstance(module, CrossStream)``, so this holds for the
    whole registry; pin it so a future tap can't quietly regress."""
    from shadow_residual.shadow_residual import ShadowResidualForCausalLM
    from shadow_residual.shadow_residual.cross_stream import CROSS_STREAM_TAPS

    taps = tuple(CROSS_STREAM_TAPS)
    model = ShadowResidualForCausalLM(_build_tiny_sr_config(cross_stream_taps=taps))
    for i, layer in enumerate(model.model.layers):
        assert layer.cross_stream_tap_names == taps
        for name in taps:
            w = getattr(layer, name).weight
            assert float(w.abs().sum()) == 0.0, f"layer {i} {name} not zero after post_init"
            assert w.requires_grad is False, f"layer {i} {name} not frozen after post_init"


def test_only_configured_taps_are_built():
    """A tap that isn't configured must not exist at all — an unwrapped tap is a
    frozen zero no-op, so building one would be pure dead weight (H x H per layer)."""
    from shadow_residual.shadow_residual import ShadowResidualForCausalLM

    model = ShadowResidualForCausalLM(
        _build_tiny_sr_config(cross_stream_taps=("cross_stream_post_attn",))
    )
    for layer in model.model.layers:
        assert layer.cross_stream_tap_names == ("cross_stream_post_attn",)
        assert isinstance(layer.cross_stream_post_attn, CrossStream)
        assert not hasattr(layer, "cross_stream"), (
            "post-attn-only config must not build the post-MLP tap"
        )
        assert not hasattr(layer, "cross_stream_pre_attn_to_post_mlp")
        assert not hasattr(layer, "cross_stream_post_mlp_to_pre_attn")


def test_default_tap_set_is_the_legacy_single_post_mlp_tap():
    """Back-compat: a config that names no tap builds exactly ``cross_stream``,
    so every existing YAML and saved adapter keeps loading unchanged."""
    from shadow_residual.shadow_residual import ShadowResidualForCausalLM

    model = ShadowResidualForCausalLM(_build_tiny_sr_config())
    for layer in model.model.layers:
        assert layer.cross_stream_tap_names == ("cross_stream",)
        assert isinstance(layer.cross_stream, CrossStream)


def test_taps_from_target_modules():
    """The tap set to BUILD is derived from PEFT's ``target_modules`` — that
    derivation is the single source of truth shared by training and serving."""
    from shadow_residual.shadow_residual.cross_stream import (
        DEFAULT_CROSS_STREAM_TAPS,
        cross_stream_taps_from_target_modules,
    )

    f = cross_stream_taps_from_target_modules
    assert f(["q_proj", "cross_stream"]) == ("cross_stream",)
    assert f(["q_proj", "cross_stream_post_attn"]) == ("cross_stream_post_attn",)
    # Order follows the registry, not the caller's list — FSDP meta-init needs
    # identical module insertion order on every rank.
    assert f(["cross_stream_post_attn", "cross_stream"]) == (
        "cross_stream",
        "cross_stream_post_attn",
    )
    # No tap named (plain-LoRA target set) → the historical single-tap topology,
    # so the module tree is unchanged from before the registry existed.
    assert f(["q_proj", "o_proj"]) == DEFAULT_CROSS_STREAM_TAPS
    assert f(None) == DEFAULT_CROSS_STREAM_TAPS
    # Cross-position wirings derive the same way.
    assert f(["q_proj", "cross_stream_pre_attn_to_post_mlp"]) == (
        "cross_stream_pre_attn_to_post_mlp",
    )
    assert f(["q_proj", "cross_stream_post_mlp_to_pre_attn"]) == (
        "cross_stream_post_mlp_to_pre_attn",
    )


def test_need_base_ahead_truth_table():
    """Only a tap whose DESTINATION precedes its SOURCE forces the base-ahead
    decoder forward. Getting this wrong either breaks the new wiring (a KeyError
    on a not-yet-computed base point) or silently moves every existing arm onto a
    different execution order, so pin the whole table."""
    from shadow_residual.shadow_residual.cross_stream import (
        CROSS_STREAM_TAPS,
        cross_stream_taps_need_base_ahead,
    )

    f = cross_stream_taps_need_base_ahead
    assert f(()) is False
    assert f(("cross_stream",)) is False                      # post_mlp  -> post_mlp
    assert f(("cross_stream_post_attn",)) is False            # post_attn -> post_attn
    assert f(("cross_stream_pre_attn_to_post_mlp",)) is False  # pre_attn -> post_mlp
    assert f(("cross_stream_post_mlp_to_pre_attn",)) is True   # post_mlp -> pre_attn
    # One backward tap in the set is enough to require it.
    assert f(("cross_stream", "cross_stream_post_mlp_to_pre_attn")) is True
    assert f(tuple(CROSS_STREAM_TAPS)) is True


def test_unknown_cross_stream_tap_name_is_rejected():
    """A typo'd tap name must fail loudly here. PEFT's own "Target modules not
    found" is far less legible, and silently building the default tap set would
    train a different topology than the one the YAML asked for."""
    import pytest

    from shadow_residual.shadow_residual.cross_stream import (
        cross_stream_taps_from_target_modules,
    )

    with pytest.raises(ValueError, match="Unknown cross-stream tap"):
        cross_stream_taps_from_target_modules(["q_proj", "cross_stream_post_atn"])


def test_config_validation_rejects_unknown_tap():
    """Second gate: an unknown tap reaching the config (e.g. hand-built) must be
    caught before it becomes a KeyError deep in the decoder forward."""
    import pytest

    from shadow_residual.shadow_residual.config_helpers import (
        validate_shadow_residual_config,
    )

    cfg = _build_tiny_sr_config()
    cfg.cross_stream_taps = ["cross_stream_nope"]
    with pytest.raises(ValueError, match="Unknown cross_stream_taps"):
        validate_shadow_residual_config(cfg)

    cfg.cross_stream_taps = []
    with pytest.raises(ValueError, match="at least one cross-stream site"):
        validate_shadow_residual_config(cfg)


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


# --- "linear" cross-stream type (CrossStreamLinear) --------------------------


def test_cross_stream_linear_is_single_full_matrix():
    """A single full H×H matmul — NOT a low-rank bottleneck. The `dim` arg does
    not shape the weight (cross-stream is H→H); the matrix is always H×H."""
    cs = CrossStreamLinear(64, 8)
    assert not isinstance(cs, nn.Linear), (
        "CrossStreamLinear must be a plain nn.Module (not a LoRA target, immune "
        "to target_modules='all-linear')"
    )
    # One projection, square H×H — no up/down sublayers.
    assert not hasattr(cs, "up") and not hasattr(cs, "down")
    assert cs.proj.in_features == 64 and cs.proj.out_features == 64
    assert cs.proj.weight.shape == (64, 64)
    assert cs.proj.weight.requires_grad is True
    x = torch.randn(2, 5, 64)
    assert cs(x).shape == x.shape


def test_cross_stream_linear_dim_does_not_shape_matrix():
    """`dim` is provenance only — the weight is H×H regardless of dim, and dim
    may even be omitted."""
    cs = CrossStreamLinear(48, dim=8)
    assert cs.proj.weight.shape == (48, 48)
    assert cs.dim == 8
    cs_no_dim = CrossStreamLinear(48)
    assert cs_no_dim.proj.weight.shape == (48, 48)


def test_cross_stream_linear_zero_at_init():
    """The H×H weight is zero-init, so the injection is exactly zero at step 0
    (adapter stream starts base-identical)."""
    cs = CrossStreamLinear(32, 4)
    assert bool((cs.proj.weight == 0).all()), "proj weight must be zero-init"
    x = torch.randn(3, 7, 32)
    out = cs(x)
    assert torch.equal(out, torch.zeros_like(out)), (
        "CrossStreamLinear must contribute exactly zero at init (zero-init weight)"
    )


def test_cross_stream_linear_nonzero_after_perturbed():
    cs = CrossStreamLinear(16, 4)
    with torch.no_grad():
        cs.proj.weight.normal_(std=0.1)
    x = torch.randn(1, 3, 16)
    out = cs(x)
    assert not torch.equal(out, torch.zeros_like(out))
    # A single matmul: out == proj(x) exactly.
    torch.testing.assert_close(out, cs.proj(x), atol=1e-6, rtol=1e-5)


def test_cross_stream_linear_ignores_extra_args():
    """Same call contract as CrossStream: extra args are tolerated + ignored."""
    cs = CrossStreamLinear(8, 2)
    x = torch.randn(1, 3, 8)
    out = cs(x, some_variant_kwarg=[1], some_future_kwarg=True)
    assert out.shape == x.shape


def test_linear_cross_stream_survives_full_model_post_init():
    """A "linear" SR model builds CrossStreamLinear sites that stay trainable
    after post_init(); the H×H weight must be re-zeroed (parent _init_weights
    randomizes every nn.Linear) so the injection is zero at init."""
    from shadow_residual.shadow_residual import ShadowResidualForCausalLM

    cfg = _build_tiny_sr_config()
    cfg.cross_stream_type = "linear"
    cfg.cross_stream_dim = 8
    model = ShadowResidualForCausalLM(cfg)
    for i, layer in enumerate(model.model.layers):
        cs = layer.cross_stream
        assert isinstance(cs, CrossStreamLinear), f"layer {i} not CrossStreamLinear"
        assert cs.proj.weight.requires_grad is True, f"layer {i} proj not trainable"
        assert float(cs.proj.weight.detach().abs().sum()) == 0.0, (
            f"layer {i} proj not re-zeroed after post_init (injection must start at 0)"
        )
