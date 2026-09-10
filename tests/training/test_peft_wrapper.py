# SPDX-License-Identifier: Apache-2.0
"""Tests for the shadow-residual PEFT integration (no HF download required).

SR is adapted with 100% stock PEFT: there is no custom LoRA layer class and no
``_register_custom_module`` call. The cross-stream site is a real frozen
zero-init ``nn.Linear`` (:class:`CrossStream`) that stock
:class:`peft.tuners.lora.layer.Linear` wraps like any other target when
``"cross_stream"`` is in ``target_modules``. The Q/K/V/O/MLP projections are
likewise wrapped by stock LoRA; the base-stream frozen invariant is a pure
``base_layer()`` call in the decoder, not a custom wrapper.

What this file exercises:

- Stock ``lora.Linear`` is dispatched at every ``cross_stream`` site when
  ``"cross_stream"`` is in ``target_modules``.
- ``rank_pattern`` / ``alpha_pattern`` flow through to the wrapper.
- End-to-end forward returns the right logits shape.
- ``save_pretrained`` writes the standard PEFT layout (no SR sidecar).
"""

import os
import tempfile

import pytest
import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.shadow_residual import (
    ShadowResidualForCausalLM,
)
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
)
from peft.tuners.lora.layer import Linear as StockLoraLinear
from shadow_residual.shadow_residual.cross_stream import (
    CROSS_STREAM_TAPS,
    CrossStream,
    CrossStreamLinear,
)


def _tiny_sr_model(cross_stream_taps=None):
    cfg = ShadowResidualConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_adapters=0,
        max_lora_rank=4,
        switch_head_dim=16,
    )
    if cross_stream_taps is not None:
        cfg.cross_stream_taps = list(cross_stream_taps)
    set_shadow_residual(config=cfg, enabled=True)
    return ShadowResidualForCausalLM(cfg)


@pytest.fixture
def tiny_sr_model():
    return _tiny_sr_model()


@pytest.fixture
def tiny_sr_model_both_taps():
    """Both same-point taps built — the "horizontal" ablation topology."""
    return _tiny_sr_model(("cross_stream", "cross_stream_post_attn"))


@pytest.fixture
def tiny_sr_model_all_taps():
    """Every registered tap built at once — the widest module tree PEFT sees."""
    return _tiny_sr_model(tuple(CROSS_STREAM_TAPS))


def _make_peft_sr_model(sr_model, **lora_kwargs):
    lora_cfg = LoraConfig(
        r=lora_kwargs.pop("r", 8),
        lora_alpha=lora_kwargs.pop("lora_alpha", 16),
        target_modules=lora_kwargs.pop("target_modules", ["cross_stream"]),
        lora_dropout=0.0,
        **lora_kwargs,
    )
    # No custom-module registration: cross_stream is a stock LoRA target.
    return get_peft_model(sr_model, lora_cfg), lora_cfg


def _cross_stream_wrappers(peft_model, suffix=None):
    """Stock lora.Linear layers whose wrapped base is a CrossStream site.

    ``suffix`` restricts to one tap's module name (e.g. ``"cross_stream_post_attn"``).
    """
    return [
        m for n, m in peft_model.named_modules()
        if isinstance(m, StockLoraLinear)
        and isinstance(m.base_layer, CrossStream)
        and (suffix is None or n.endswith("." + suffix))
    ]


def test_cross_stream_sites_present_on_sr_model(tiny_sr_model):
    """SR decoder has at least one CrossStream module instance."""
    sites = [m for m in tiny_sr_model.modules() if isinstance(m, CrossStream)]
    assert len(sites) >= 1


def test_peft_wraps_cross_stream_with_stock_layer(tiny_sr_model):
    """get_peft_model installs a STOCK lora.Linear at every cross_stream site."""
    peft_model, _ = _make_peft_sr_model(tiny_sr_model)

    wrappers = _cross_stream_wrappers(peft_model)
    assert wrappers, "expected at least one stock lora.Linear on a cross_stream site"

    for w in wrappers:
        assert "default" in w.lora_A
        assert w.lora_A["default"].weight.shape == (8, 64)
        assert w.lora_B["default"].weight.shape == (64, 8)


def test_rank_pattern_overrides_cross_stream_rank(tiny_sr_model):
    """``rank_pattern={'cross_stream': R}`` overrides the per-site rank."""
    peft_model, _ = _make_peft_sr_model(
        tiny_sr_model, r=8, rank_pattern={"cross_stream": 16},
    )
    wrappers = _cross_stream_wrappers(peft_model)
    assert wrappers
    for w in wrappers:
        assert w.lora_A["default"].weight.shape == (16, 64)
        assert w.lora_B["default"].weight.shape == (64, 16)


def test_alpha_pattern_overrides_cross_stream_alpha(tiny_sr_model):
    """``alpha_pattern={'cross_stream': A}`` overrides the per-site alpha."""
    peft_model, _ = _make_peft_sr_model(
        tiny_sr_model, r=8, lora_alpha=16, alpha_pattern={"cross_stream": 32},
    )
    wrappers = _cross_stream_wrappers(peft_model)
    assert wrappers
    for w in wrappers:
        # scaling = alpha / r = 32 / 8 = 4.0 (vs default 16/8 = 2.0)
        assert w.scaling["default"] == pytest.approx(4.0)


def test_peft_wraps_post_attn_tap_with_stock_layer(tiny_sr_model_both_taps):
    """The post-attention tap is an ordinary stock LoRA target too — no custom
    class, no registration, same as the post-MLP one."""
    peft_model, _ = _make_peft_sr_model(
        tiny_sr_model_both_taps, target_modules=["cross_stream_post_attn"],
    )
    wrappers = _cross_stream_wrappers(peft_model, "cross_stream_post_attn")
    assert wrappers, "expected stock lora.Linear on every cross_stream_post_attn site"
    for w in wrappers:
        assert w.lora_A["default"].weight.shape == (8, 64)
        assert w.lora_B["default"].weight.shape == (64, 8)


@pytest.mark.parametrize(
    "tap", ["cross_stream_pre_attn_to_post_mlp", "cross_stream_post_mlp_to_pre_attn"],
)
def test_peft_wraps_cross_position_taps_with_stock_layer(tiny_sr_model_all_taps, tap):
    """The cross-position wirings are ordinary stock LoRA targets too.

    They differ from the same-point taps only in WHERE the decoder reads and
    writes; at the PEFT boundary they are indistinguishable frozen zero-init
    linears, so no custom class or registration may creep in for them.
    """
    peft_model, _ = _make_peft_sr_model(tiny_sr_model_all_taps, target_modules=[tap])
    wrappers = _cross_stream_wrappers(peft_model, tap)
    assert wrappers, f"expected stock lora.Linear on every {tap} site"
    assert len(wrappers) == 3  # one per decoder layer
    for w in wrappers:
        assert w.lora_A["default"].weight.shape == (8, 64)
        assert w.lora_B["default"].weight.shape == (64, 8)


@pytest.mark.parametrize("target", list(CROSS_STREAM_TAPS))
def test_tap_names_are_separable_under_peft_suffix_matching(
    tiny_sr_model_all_taps, target,
):
    """Naming one tap in ``target_modules`` must wrap that tap and NO other.

    PEFT's list-form matching is ``key == target or key.endswith("." + target)``,
    so every registry name is cleanly separable — but the whole ablation depends on
    it (an accidental suffix collision would wrap two taps at once and make the
    single-tap arms unreproducible), so assert the full cross product rather than
    assume it. This is also the guard a future tap name must pass.
    """
    peft_model, _ = _make_peft_sr_model(tiny_sr_model_all_taps, target_modules=[target])
    assert _cross_stream_wrappers(peft_model, target)
    for other in CROSS_STREAM_TAPS:
        if other == target:
            continue
        assert not _cross_stream_wrappers(peft_model, other), (
            f"target_modules=[{target!r}] also wrapped {other!r}"
        )


def test_per_tap_rank_and_alpha_resolve_independently(tiny_sr_model_both_taps):
    """Both taps at once with DIFFERENT rank and alpha per tap.

    Guards the PEFT footgun behind the ``alpha_pattern`` support:
    ``lora.model._create_and_replace`` resolves ONE ``target_name_key`` out of
    ``chain(rank_pattern.keys(), alpha_pattern.keys())`` and indexes BOTH dicts
    with it. Here the key sets match (as the config schema enforces), so each tap
    must get its own rank AND its own alpha.
    """
    peft_model, _ = _make_peft_sr_model(
        tiny_sr_model_both_taps,
        r=8,
        lora_alpha=16,
        target_modules=["cross_stream", "cross_stream_post_attn"],
        rank_pattern={"cross_stream": 16, "cross_stream_post_attn": 4},
        alpha_pattern={"cross_stream": 32, "cross_stream_post_attn": 8},
    )

    post_mlp = _cross_stream_wrappers(peft_model, "cross_stream")
    post_attn = _cross_stream_wrappers(peft_model, "cross_stream_post_attn")
    assert post_mlp and post_attn
    assert len(post_mlp) == len(post_attn) == 3  # one per decoder layer

    for w in post_mlp:
        assert w.lora_A["default"].weight.shape == (16, 64)
        assert w.scaling["default"] == pytest.approx(32 / 16)
    for w in post_attn:
        assert w.lora_A["default"].weight.shape == (4, 64)
        assert w.scaling["default"] == pytest.approx(8 / 4)


def test_forward_through_peft_sr_model(tiny_sr_model):
    """End-to-end forward runs and produces the expected logits shape."""
    peft_model, _ = _make_peft_sr_model(tiny_sr_model)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    out = peft_model(input_ids=input_ids)
    assert out.logits.shape == (1, 6, tiny_sr_model.config.vocab_size)


def test_q_proj_and_cross_stream_both_install_stock_lora(tiny_sr_model):
    """target_modules=['q_proj','cross_stream'] dispatches STOCK LoRA on BOTH
    q_proj and cross_stream — SR uses no custom projection wrapper at all. The
    base-stream gate is a pure base_layer() call in the decoder."""
    peft_model, _ = _make_peft_sr_model(
        tiny_sr_model, target_modules=["q_proj", "cross_stream"],
    )
    q_loras = [
        (n, m) for n, m in peft_model.named_modules()
        if n.endswith(".q_proj") and isinstance(m, StockLoraLinear)
    ]
    assert q_loras, "expected a stock lora.Linear dispatched on q_proj"

    cross_wrappers = _cross_stream_wrappers(peft_model)
    assert cross_wrappers, "expected a stock lora.Linear dispatched on cross_stream"


def test_save_pretrained_writes_standard_peft_layout(tiny_sr_model):
    """Standard PEFT save format — adapter_config.json + safetensors."""
    peft_model, _ = _make_peft_sr_model(tiny_sr_model)

    with tempfile.TemporaryDirectory() as tmp:
        peft_model.save_pretrained(tmp)
        files = set(os.listdir(tmp))
        assert "adapter_config.json" in files
        assert "adapter_model.safetensors" in files
        # No SR-specific sidecar (cross_stream LoRA rides in the safetensors).
        assert "shadow_residual_cross_stream.bin" not in files
        assert "shadow_residual_config.json" not in files


# --- "linear" cross-stream type (modules_to_save round-trip) ------------------


def _tiny_linear_sr_config(dim=8):
    cfg = ShadowResidualConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_adapters=0,
        max_lora_rank=4,
        switch_head_dim=16,
    )
    set_shadow_residual(config=cfg, enabled=True)
    cfg.cross_stream_type = "linear"
    cfg.cross_stream_dim = dim
    return cfg


def test_linear_cross_stream_built_on_sr_model():
    """A "linear" SR config builds CrossStreamLinear sites (not CrossStream)."""
    model = ShadowResidualForCausalLM(_tiny_linear_sr_config())
    linear_sites = [m for m in model.modules() if isinstance(m, CrossStreamLinear)]
    frozen_sites = [m for m in model.modules() if isinstance(m, CrossStream)]
    assert len(linear_sites) == model.config.num_hidden_layers
    assert not frozen_sites, "linear config must not build any frozen CrossStream"


def _active_cross_stream_modules(peft_model):
    """The ACTIVE cross-stream module at each site.

    PEFT wraps a modules_to_save target in a ModulesToSaveWrapper holding two
    copies: `original_module` (frozen, base) and `modules_to_save[<adapter>]`
    (trainable, used in forward when the adapter is active). We want the latter —
    that is what trains and what save/from_pretrained round-trips. Selecting by a
    bare `isinstance(CrossStreamLinear)` walk would ambiguously match both copies.
    """
    from peft.utils.other import ModulesToSaveWrapper

    out = []
    for m in peft_model.modules():
        if isinstance(m, ModulesToSaveWrapper):
            active = m.active_adapter
            active = active[0] if isinstance(active, (list, tuple)) else active
            sub = m.modules_to_save[active]
            assert isinstance(sub, CrossStreamLinear)
            out.append(sub)
    return out


def test_linear_cross_stream_modules_to_save_roundtrip():
    """The linear cross-stream trains via PEFT modules_to_save, and its weights
    survive save_pretrained -> from_pretrained onto a fresh matching base."""
    from peft import PeftModel

    model = ShadowResidualForCausalLM(_tiny_linear_sr_config(dim=8))
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "o_proj"],
        modules_to_save=["cross_stream"],
    )
    peft_model = get_peft_model(model, lora_cfg)

    # The active cross-stream copy trains (via modules_to_save), NOT LoRA-wrapped.
    cs_active = _active_cross_stream_modules(peft_model)
    assert len(cs_active) == model.config.num_hidden_layers
    trainable_cs = [
        p for m in cs_active for p in m.parameters() if p.requires_grad
    ]
    assert trainable_cs, "linear cross-stream params must be trainable pre-save"
    # No cross_stream LoRA layer was created.
    assert not _cross_stream_wrappers(peft_model)

    # Perturb the trainable cross-stream weight so the round-trip is meaningful.
    with torch.no_grad():
        for m in cs_active:
            m.proj.weight.normal_(std=0.1)
    saved = [m.proj.weight.detach().clone() for m in cs_active]

    with tempfile.TemporaryDirectory() as tmp:
        peft_model.save_pretrained(tmp)
        assert "adapter_model.safetensors" in set(os.listdir(tmp))

        fresh_base = ShadowResidualForCausalLM(_tiny_linear_sr_config(dim=8))
        reloaded = PeftModel.from_pretrained(fresh_base, tmp)

    reloaded_cs = _active_cross_stream_modules(reloaded)
    assert len(reloaded_cs) == len(cs_active)
    for i, m in enumerate(reloaded_cs):
        torch.testing.assert_close(m.proj.weight, saved[i], atol=1e-5, rtol=1e-4)
