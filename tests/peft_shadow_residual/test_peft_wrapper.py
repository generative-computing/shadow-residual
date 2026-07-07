# SPDX-License-Identifier: Apache-2.0
"""Tests for the shadow-residual PEFT integration (no HF download required).

After Phase 2 of the SR refactor, SR built-in LoRA tensors
(``q_lora_A``/``o_lora_A``/``gate_up_lora_B_slices``/``down_lora_B``) and
the ``_freeze_sr_builtin_loras`` helper no longer exist. LoRA comes
exclusively from PEFT, dispatched onto the SR ``_StreamGatedLinear``
projections via :class:`ShadowResidualLora` and onto the no-op
:class:`CrossStream` site via :class:`CrossStreamLora`.

What this file exercises:

- :class:`CrossStreamLora` is dispatched at every ``cross_stream`` site
  when ``"cross_stream"`` is in ``target_modules``.
- ``rank_pattern`` / ``alpha_pattern`` flow through to the wrapper.
- End-to-end forward returns the right logits shape.
- ``save_pretrained`` writes the standard PEFT layout (no SR sidecar).
"""

import os
import tempfile

import pytest
import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig
from shadow_residual.shadow_residual import (
    ShadowResidualForCausalLM,
)
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
def tiny_sr_model():
    cfg = GraniteSwitchConfig(
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
    return ShadowResidualForCausalLM(cfg)


def _make_peft_sr_model(sr_model, **lora_kwargs):
    lora_cfg = LoraConfig(
        r=lora_kwargs.pop("r", 8),
        lora_alpha=lora_kwargs.pop("lora_alpha", 16),
        target_modules=lora_kwargs.pop("target_modules", ["cross_stream"]),
        lora_dropout=0.0,
        **lora_kwargs,
    )
    lora_cfg._register_custom_module(
        {CrossStream: CrossStreamLora, _StreamGatedLinear: ShadowResidualLora}
    )
    return get_peft_model(sr_model, lora_cfg), lora_cfg


def test_cross_stream_sites_present_on_sr_model(tiny_sr_model):
    """SR decoder has at least one CrossStream module instance."""
    sites = [m for m in tiny_sr_model.modules() if isinstance(m, CrossStream)]
    assert len(sites) >= 1


def test_peft_wraps_cross_stream_with_custom_layer(tiny_sr_model):
    """get_peft_model installs CrossStreamLora at every cross_stream site."""
    peft_model, _ = _make_peft_sr_model(tiny_sr_model)

    wrappers = [m for m in peft_model.modules() if isinstance(m, CrossStreamLora)]
    assert wrappers, "expected at least one CrossStreamLora wrapper in the model"

    for w in wrappers:
        assert "default" in w.lora_A
        assert w.lora_A["default"].weight.shape == (8, 64)
        assert w.lora_B["default"].weight.shape == (64, 8)


def test_rank_pattern_overrides_cross_stream_rank(tiny_sr_model):
    """``rank_pattern={'cross_stream': R}`` overrides the per-site rank."""
    peft_model, _ = _make_peft_sr_model(
        tiny_sr_model, r=8, rank_pattern={"cross_stream": 16},
    )
    wrappers = [m for m in peft_model.modules() if isinstance(m, CrossStreamLora)]
    assert wrappers
    for w in wrappers:
        assert w.lora_A["default"].weight.shape == (16, 64)
        assert w.lora_B["default"].weight.shape == (64, 16)


def test_alpha_pattern_overrides_cross_stream_alpha(tiny_sr_model):
    """``alpha_pattern={'cross_stream': A}`` overrides the per-site alpha."""
    peft_model, _ = _make_peft_sr_model(
        tiny_sr_model, r=8, lora_alpha=16, alpha_pattern={"cross_stream": 32},
    )
    wrappers = [m for m in peft_model.modules() if isinstance(m, CrossStreamLora)]
    assert wrappers
    for w in wrappers:
        # scaling = alpha / r = 32 / 8 = 4.0 (vs default 16/8 = 2.0)
        assert w.scaling["default"] == pytest.approx(4.0)


def test_forward_through_peft_sr_model(tiny_sr_model):
    """End-to-end forward runs and produces the expected logits shape."""
    peft_model, _ = _make_peft_sr_model(tiny_sr_model)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    out = peft_model(input_ids=input_ids)
    assert out.logits.shape == (1, 6, tiny_sr_model.config.vocab_size)


def test_q_proj_target_installs_shadow_residual_lora(tiny_sr_model):
    """target_modules=['q_proj','cross_stream'] dispatches SR LoRA on q_proj."""
    peft_model, _ = _make_peft_sr_model(
        tiny_sr_model, target_modules=["q_proj", "cross_stream"],
    )
    sr_loras = [
        (n, m) for n, m in peft_model.named_modules()
        if isinstance(m, ShadowResidualLora)
    ]
    assert sr_loras, "expected ShadowResidualLora dispatched on q_proj"
    assert any(n.endswith(".q_proj") for n, _ in sr_loras)

    cross_wrappers = [
        m for m in peft_model.modules() if isinstance(m, CrossStreamLora)
    ]
    assert cross_wrappers, "expected CrossStreamLora dispatched on cross_stream"


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
