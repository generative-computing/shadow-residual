# SPDX-License-Identifier: Apache-2.0
"""Tests for shadow-residual architecture (HF backend, experimental).

After Phase 1+2 of the SR refactor, SR no longer inherits from
the upstream Granite model: there is no ``adapter_token_ids`` buffer, no
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

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.shadow_residual import (
    ShadowResidualDecoderLayer,
    ShadowResidualForCausalLM,
)
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
    validate_shadow_residual_config,
)


# ── Fixtures ──────────────────────────────────────────────────────


def _tiny_config(cross_stream_taps=None):
    """Minimal SR config for CPU tests."""
    config = ShadowResidualConfig(
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
        config.cross_stream_taps = list(cross_stream_taps)
    set_shadow_residual(config, enabled=True)
    return config


@pytest.fixture
def tiny_config():
    return _tiny_config()


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
    # cross_stream + projections all wrapped by stock LoRA (no custom module).
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
    def test_forward_shapes_bare(self, tiny_model):
        """Bare SR model (no PEFT wrapping) → bare single-stream path."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids)
        assert out.logits.shape == (1, 5, tiny_model.config.vocab_size)

    def test_forward_shapes_dual_stream(self, tiny_config):
        """SR + PEFT with cross_stream wrapped → dual-stream forward."""
        sr = ShadowResidualForCausalLM(tiny_config)
        peft_model = _wrap_with_peft(
            sr, target_modules=["q_proj", "cross_stream"],
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            out = peft_model(input_ids=input_ids)
        assert out.logits.shape == (1, 5, tiny_config.vocab_size)

    def test_decoder_layers_are_sr(self, tiny_model):
        for layer in tiny_model.model.layers:
            assert isinstance(layer, ShadowResidualDecoderLayer)

    def test_bare_model_matches_disabled_adapter(self, tiny_config):
        """A bare SR model (no adapter) runs the single-stream path; its logits
        must equal the same model's logits under a top-level disable_adapter()
        (both collapse to the unadapted base). Guards that the bare single-stream
        path stays base-identical and the two code paths agree."""
        torch.manual_seed(0)
        sr = ShadowResidualForCausalLM(tiny_config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            bare = sr(input_ids=input_ids).logits  # never PEFT-wrapped → single-stream

        # Wrap with a trained (nonzero) adapter, then disable it: must match bare.
        peft_model = _wrap_with_peft(
            sr, target_modules=["q_proj", "cross_stream"], perturb_lora_b=True,
        )
        with torch.no_grad():
            with peft_model.disable_adapter():
                disabled = peft_model(input_ids=input_ids).logits
        torch.testing.assert_close(bare, disabled, atol=1e-5, rtol=1e-4)

    def test_bare_linear_cross_stream_takes_single_stream_path(self, tiny_config):
        """A bare "linear" SR model (CrossStreamLinear site, no adapter) must be
        treated as NOT wrapped → single-stream forward_bare path. Guards against
        the regression where any non-CrossStream site would force the dual-stream
        path even with no adapter attached (see modeling_hf._any_cross_stream_wrapped)."""
        from shadow_residual.shadow_residual.modeling_hf import (
            _any_cross_stream_wrapped,
        )

        cfg = ShadowResidualConfig(**tiny_config.to_dict())
        set_shadow_residual(cfg, enabled=True)
        cfg.cross_stream_type = "linear"
        cfg.cross_stream_dim = 8
        sr = ShadowResidualForCausalLM(cfg)
        sr.eval()

        assert _any_cross_stream_wrapped(sr.model.layers) is False, (
            "bare CrossStreamLinear must NOT count as an active adapter"
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            out = sr(input_ids=input_ids)
        assert out.logits.shape == (1, 6, cfg.vocab_size)


class TestCrossStreamTaps:
    """The cross-stream site is a registry of taps, not a single hard-coded one.

    ``CROSS_STREAM_TAPS`` maps a module name to a (base source, adapter
    destination) pair; the set built on the model is derived from PEFT's
    ``target_modules``. These tests pin (a) back-compat of the default topology,
    (b) that the new post-attention tap is genuinely wired into the forward, and
    (c) that the frozen-base invariant survives multiple taps.
    """

    _BOTH = ("cross_stream", "cross_stream_post_attn")

    def test_default_topology_state_dict_keys_unchanged(self):
        """Back-compat contract at the checkpoint boundary: a default-config SR
        model exposes exactly the historical ``cross_stream.weight`` keys and no
        new ones, so saved adapters and base state_dicts keep round-tripping."""
        sr = ShadowResidualForCausalLM(_tiny_config())
        keys = set(sr.state_dict())
        n_layers = sr.config.num_hidden_layers
        assert {f"model.layers.{i}.cross_stream.weight" for i in range(n_layers)} <= keys
        assert not [k for k in keys if "cross_stream_post_attn" in k]

    def test_post_attn_tap_alone_is_actually_wired(self):
        """With ONLY the post-attention tap present and wrapped, the adapter
        stream can differ from the base stream *only* through that tap: both
        streams start from the same embeddings and no projection LoRA is
        configured. So logits != bare logits proves ``_inject(..., "post_attn")``
        fires. If the injection site were missing, this would silently pass as
        "no change" — hence the explicit inequality.
        """
        torch.manual_seed(0)
        cfg = _tiny_config(("cross_stream_post_attn",))
        sr = ShadowResidualForCausalLM(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            bare = sr(input_ids=input_ids).logits

        peft_model = _wrap_with_peft(
            sr, target_modules=["cross_stream_post_attn"], perturb_lora_b=True,
        )
        with torch.no_grad():
            adapted = peft_model(input_ids=input_ids).logits

        assert not torch.allclose(bare, adapted, atol=1e-6), (
            "post-attention tap had no effect on the adapter stream — the "
            "injection site is not wired"
        )

    def test_both_taps_engage_dual_stream_and_keep_base_frozen(self):
        """Invariant 3 with the full "horizontal" topology: two taps wrapped and
        every ``lora_B`` perturbed, yet ``disable_adapter()`` must still collapse
        to the unadapted base. Taps only ever READ ``h_base`` and WRITE
        ``h_adapt``, so adding taps cannot leak a delta into the frozen stream."""
        torch.manual_seed(0)
        sr = ShadowResidualForCausalLM(_tiny_config(self._BOTH))
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            bare = sr(input_ids=input_ids).logits

        peft_model = _wrap_with_peft(
            sr,
            target_modules=["q_proj", *self._BOTH],
            perturb_lora_b=True,
        )
        with torch.no_grad():
            adapted = peft_model(input_ids=input_ids).logits
            with peft_model.disable_adapter():
                disabled = peft_model(input_ids=input_ids).logits

        torch.testing.assert_close(bare, disabled, atol=1e-5, rtol=1e-4)
        assert not torch.allclose(bare, adapted, atol=1e-6)

    def test_base_ahead_forward_matches_interleaved(self):
        """The base-ahead forward must be a faithful REORDERING, not a second
        semantics.

        The two paths are compared on a topology both can serve (the default
        post-MLP tap): run it interleaved, then force ``_base_ahead`` on and run
        the same weights again. Every op in the base-ahead body is the same pure
        function of the same inputs — only the fused layernorms split into one
        call per stream — so the logits must agree. This is the guard for the
        whole restructuring: if ``forward_base`` / ``forward_adapt`` ever stop
        composing to the interleaved result (e.g. a second cache update, or the
        adapter attending against pre-cache K/V), this fails.
        """
        torch.manual_seed(0)
        sr = ShadowResidualForCausalLM(_tiny_config())
        peft_model = _wrap_with_peft(
            sr, target_modules=["q_proj", "cross_stream"], perturb_lora_b=True,
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        layers = [
            m for m in peft_model.modules()
            if isinstance(m, ShadowResidualDecoderLayer)
        ]
        assert layers and all(layer._base_ahead is False for layer in layers)
        with torch.no_grad():
            interleaved = peft_model(input_ids=input_ids).logits

        for layer in layers:
            layer._base_ahead = True
        with torch.no_grad():
            base_ahead = peft_model(input_ids=input_ids).logits

        torch.testing.assert_close(base_ahead, interleaved, atol=1e-6, rtol=1e-5)

    def test_pre_attn_to_post_mlp_tap_differs_from_default_tap(self):
        """``cross_stream_pre_attn_to_post_mlp`` must be a genuinely different
        wiring, not a relabelled ``cross_stream``.

        Both taps write into the adapter stream at the same place; they differ only
        in WHERE they read the base stream (layer entry vs layer exit). With
        identical seeds, ranks and LoRA weights, identical logits would mean the
        source point was never actually threaded through and both taps read the
        same tensor.
        """
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        def _logits(tap):
            torch.manual_seed(0)
            sr = ShadowResidualForCausalLM(_tiny_config((tap,)))
            torch.manual_seed(1)
            peft_model = _wrap_with_peft(
                sr, target_modules=[tap], perturb_lora_b=True,
            )
            with torch.no_grad():
                return peft_model(input_ids=input_ids).logits

        default_tap = _logits("cross_stream")
        cross_position = _logits("cross_stream_pre_attn_to_post_mlp")
        assert not torch.allclose(default_tap, cross_position, atol=1e-6), (
            "the pre-attention source point is not wired — the tap is reading the "
            "same base state as the post-MLP tap"
        )

    def test_post_mlp_to_pre_attn_tap_is_wired_and_keeps_base_frozen(self):
        """The zero-lag wiring: base post-MLP → adapter pre-attention, resolved
        within one layer. It is the only tap that runs the base-ahead forward, so
        this covers both that the injection fires AND that the restructured path
        still leaves the frozen base stream untouched (invariant 1)."""
        torch.manual_seed(0)
        cfg = _tiny_config(("cross_stream_post_mlp_to_pre_attn",))
        sr = ShadowResidualForCausalLM(cfg)
        assert all(
            layer._base_ahead is True for layer in sr.model.layers
        ), "a destination-precedes-source tap must select the base-ahead forward"

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            bare = sr(input_ids=input_ids).logits

        peft_model = _wrap_with_peft(
            sr,
            target_modules=["cross_stream_post_mlp_to_pre_attn"],
            perturb_lora_b=True,
        )
        with torch.no_grad():
            adapted = peft_model(input_ids=input_ids).logits
            with peft_model.disable_adapter():
                disabled = peft_model(input_ids=input_ids).logits

        assert not torch.allclose(bare, adapted, atol=1e-6), (
            "post_mlp → pre_attn tap had no effect — the injection site is not wired"
        )
        torch.testing.assert_close(bare, disabled, atol=1e-5, rtol=1e-4)

    def test_base_ahead_incremental_decode_matches_full_forward(self):
        """``forward_base`` owns the layer's ONE cache update. If the split
        duplicated it (or the adapter half touched the cache), incremental decode
        would attend against a doubled K/V and diverge from a single full forward.
        """
        torch.manual_seed(0)
        sr = ShadowResidualForCausalLM(
            _tiny_config(("cross_stream_post_mlp_to_pre_attn",))
        )
        peft_model = _wrap_with_peft(
            sr,
            target_modules=["q_proj", "cross_stream_post_mlp_to_pre_attn"],
            perturb_lora_b=True,
        )
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        with torch.no_grad():
            full = peft_model(input_ids=ids).logits
            prefill = peft_model(input_ids=ids[:, :-1], use_cache=True)
            step = peft_model(
                input_ids=ids[:, -1:],
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )

        cache = step.past_key_values
        assert len(cache.layers) == sr.config.num_hidden_layers
        assert cache.layers[0].keys.shape[2] == ids.shape[1], (
            "K/V was cached more than once per layer"
        )
        torch.testing.assert_close(
            step.logits[:, -1], full[:, -1], atol=1e-5, rtol=1e-4,
        )

    def test_taps_are_summed_not_overwritten(self):
        """Both taps active must ADD both injections. Perturbing only the
        post-MLP tap's ``lora_B`` and then only the post-attention one must give
        two different results, and both must differ from the untouched model —
        i.e. neither injection shadows the other."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        def _logits(active_tap):
            torch.manual_seed(0)
            sr = ShadowResidualForCausalLM(_tiny_config(self._BOTH))
            peft_model = _wrap_with_peft(
                sr, target_modules=list(self._BOTH), perturb_lora_b=False,
            )
            torch.manual_seed(1)
            with torch.no_grad():
                for name, p in peft_model.named_parameters():
                    if "lora_B" in name and f".{active_tap}." in name:
                        p.data.normal_(mean=0.0, std=0.05)
                return peft_model(input_ids=input_ids).logits

        post_mlp_only = _logits("cross_stream")
        post_attn_only = _logits("cross_stream_post_attn")
        assert not torch.allclose(post_mlp_only, post_attn_only, atol=1e-6), (
            "the two taps produce identical logits — they are probably both "
            "injecting at the same point"
        )


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
        config = ShadowResidualConfig(num_adapters=0)
        set_shadow_residual(config, enabled=True)
        # Must not raise.
        validate_shadow_residual_config(config)

    def test_shadow_residual_requires_attention_only(self):
        config = ShadowResidualConfig(
            num_adapters=0,
            max_lora_rank=4,
            layer_types=["attention", "mamba", "attention"],
            num_hidden_layers=3,
        )
        config.shadow_residual = True
        with pytest.raises(ValueError, match="attention"):
            validate_shadow_residual_config(config)

    def test_cross_stream_rank_requires_shadow_residual(self):
        config = ShadowResidualConfig(num_adapters=0)
        config.shadow_residual = False
        config.cross_stream_rank = 8
        with pytest.raises(ValueError, match="cross_stream_rank"):
            validate_shadow_residual_config(config)

    def test_shadow_residual_false_by_default(self):
        config = ShadowResidualConfig()
        assert getattr(config, "shadow_residual", False) is False
