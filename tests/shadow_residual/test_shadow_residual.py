# SPDX-License-Identifier: Apache-2.0
"""Tests for shadow-residual architecture (HF backend, experimental).

After Phase 1+2 of the SR refactor, SR no longer inherits from
``GraniteSwitchModel``: there is no ``adapter_token_ids`` buffer, no
``SingleSwitch``, no per-projection ``q_lora_B``/``o_lora_B`` attributes,
and no final-step merge gate. LoRA divergence is driven through PEFT;
divergence between the streams is exercised in
``test_dual_kv_cache_invariant.py``. What this file covers:

- Shapes are correct for adapter-only and dual-stream forward.
- Config validator still rejects mixed attention/SSM layer types.
- ``shadow_residual=False`` is the default.
- KV cache layer count matches ``num_hidden_layers`` (no +1 switch
  layer offset since the switch is gone).
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig
from shadow_residual.shadow_residual import (
    ShadowResidualDecoderLayer,
    ShadowResidualForCausalLM,
)
from shadow_residual.shadow_residual._stream_gated_linear import (
    _StreamGatedLinear,
)
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
    validate_shadow_residual_config,
)
from shadow_residual.shadow_residual.cross_stream import CrossStream
from shadow_residual.peft_shadow_residual import CrossStreamLora
from shadow_residual.peft_shadow_residual.stream_gated_lora import (
    ShadowResidualLora,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def tiny_config():
    """Minimal SR config for CPU tests — no GraniteSwitch routing fields."""
    config = GraniteSwitchConfig(
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
    set_shadow_residual(config, enabled=True)
    return config


@pytest.fixture
def tiny_model(tiny_config):
    model = ShadowResidualForCausalLM(tiny_config)
    model.eval()
    return model


def _wrap_with_peft(sr_model, *, target_modules, perturb_lora_b=True):
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
    )
    lora_cfg._register_custom_module(
        {CrossStream: CrossStreamLora, _StreamGatedLinear: ShadowResidualLora}
    )
    peft_model = get_peft_model(sr_model, lora_cfg)
    if perturb_lora_b:
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_B" in name:
                    p.data.normal_(mean=0.0, std=0.02)
    peft_model.eval()
    return peft_model


# ── Tests ─────────────────────────────────────────────────────────


class TestShadowResidualForward:
    def test_forward_shapes_adapter_only(self, tiny_model):
        """Bare SR model (no PEFT wrapping) → adapter-only early-exit path."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids)
        assert out.logits.shape == (1, 5, tiny_model.config.vocab_size)

    def test_forward_shapes_dual_stream(self, tiny_config):
        """SR + PEFT with cross_stream wrapped → dual-stream forward."""
        sr = ShadowResidualForCausalLM(tiny_config)
        peft_model = _wrap_with_peft(
            sr, target_modules=["k_proj", "cross_stream"],
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            out = peft_model(input_ids=input_ids)
        assert out.logits.shape == (1, 5, tiny_config.vocab_size)

    def test_decoder_layers_are_sr(self, tiny_model):
        for layer in tiny_model.model.layers:
            assert isinstance(layer, ShadowResidualDecoderLayer)


class TestKVCache:
    def test_kv_cache_layer_count(self, tiny_model):
        """KV cache has exactly ``num_hidden_layers`` entries — no switch layer offset."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids, use_cache=True)
        cache = out.past_key_values
        n_layers = tiny_model.config.num_hidden_layers
        # DynamicCache stores per-layer key/value via .layers[idx].keys/values.
        # Number of layers in the cache must match num_hidden_layers (no +1
        # offset for the removed switch layer).
        assert len(cache.layers) == n_layers
        for layer_idx in range(n_layers):
            k = cache.layers[layer_idx].keys
            assert k.shape[1] == tiny_model.config.num_key_value_heads


class TestConfigValidation:
    """Config validator after Phase 2 relaxation."""

    def test_shadow_residual_with_zero_adapters_is_valid(self):
        """Phase 2 dropped the ``num_adapters > 0`` requirement."""
        config = GraniteSwitchConfig(num_adapters=0)
        set_shadow_residual(config, enabled=True)
        # Must not raise.
        validate_shadow_residual_config(config)

    def test_shadow_residual_requires_attention_only(self):
        config = GraniteSwitchConfig(
            num_adapters=0,
            max_lora_rank=4,
            layer_types=["attention", "mamba", "attention"],
            num_hidden_layers=3,
        )
        config.shadow_residual = True
        with pytest.raises(ValueError, match="attention"):
            validate_shadow_residual_config(config)

    def test_cross_stream_rank_requires_shadow_residual(self):
        config = GraniteSwitchConfig(num_adapters=0)
        config.shadow_residual = False
        config.cross_stream_rank = 8
        with pytest.raises(ValueError, match="cross_stream_rank"):
            validate_shadow_residual_config(config)

    def test_shadow_residual_false_by_default(self):
        config = GraniteSwitchConfig()
        assert getattr(config, "shadow_residual", False) is False
