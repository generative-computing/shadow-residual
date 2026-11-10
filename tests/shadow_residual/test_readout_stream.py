# SPDX-License-Identifier: Apache-2.0
"""Read-out-stream SR (``adapter_stream="readout"``) invariants, CPU only.

The read-out adapter stream runs no attention/MLP:
``h_{l+1} = readout_norm_l(h_l) + W_cross_l(h_base_out_l)``, seeded with the
embeddings, head on ``norm(h_N)``. See docs/sr_readout_stream_design.html.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.config_helpers import set_shadow_residual
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig


def _config(adapter_stream="readout"):
    config = ShadowResidualConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        residual_multiplier=0.5,
        embedding_multiplier=2.0,
        logits_scaling=3.0,
    )
    set_shadow_residual(config, enabled=True)
    config.adapter_stream = adapter_stream
    return config


def _lora(**kw):
    return LoraConfig(
        r=4, lora_alpha=8, target_modules=["cross_stream"],
        modules_to_save=["readout_norm"], lora_dropout=0.0, **kw,
    )


def _trained(model):
    """Wrap with the read-out adapter and give every trainable a nonzero value."""
    peft_model = get_peft_model(model, _lora())
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            elif "modules_to_save" in name and "readout_norm" in name:
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.2)
    return peft_model.eval()


@pytest.fixture
def ids():
    return torch.tensor([[5, 17, 42, 7, 99, 3, 250]])


@pytest.fixture
def base():
    torch.manual_seed(0)
    return ShadowResidualForCausalLM(_config()).eval()


def test_full_mode_has_no_readout_norm():
    model = ShadowResidualForCausalLM(_config("full"))
    assert not any("readout_norm" in n for n, _ in model.named_parameters())


def test_readout_norm_frozen_identity_on_bare(base):
    for layer in base.model.layers:
        assert torch.equal(layer.readout_norm.weight, torch.ones_like(layer.readout_norm.weight))
        assert not layer.readout_norm.weight.requires_grad


def test_unknown_adapter_stream_rejected():
    with pytest.raises(ValueError, match="adapter_stream"):
        ShadowResidualForCausalLM(_config("bogus"))


def test_logits_match_reference_recurrence(base, ids):
    """Model logits == hand-written h = norm_l(h) + cross_l(hb_l) over bare base layers."""
    peft_model = _trained(base)
    inner = peft_model.base_model.model
    with torch.no_grad():
        got = peft_model(input_ids=ids).logits
        h = inner.model.embed_tokens(ids) * inner.config.embedding_multiplier
        for i, layer in enumerate(inner.model.layers):
            h = layer.readout_norm(h) + layer.cross_stream(_bare_layer_out(inner, ids, i))
        want = inner.lm_head(inner.model.norm(h)) / inner.config.logits_scaling
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def _bare_layer_out(inner, ids, upto):
    h = inner.model.embed_tokens(ids) * inner.config.embedding_multiplier
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    pe = inner.model.rotary_emb(h, position_ids=pos)
    mask = torch.full((ids.shape[1], ids.shape[1]), float("-inf")).triu(1)[None, None]
    for layer in inner.model.layers[: upto + 1]:
        h = layer._base_layer(
            h, attention_mask=mask, past_key_values=None, use_cache=False,
            cache_position=None, position_embeddings=pe, base_only=True,
        )
    return h


def test_base_stream_bit_identical_to_bare(base, ids):
    """Invariant 1: with a trained read-out adapter the base stream after every
    layer equals the unadapted base's layer outputs exactly."""
    with torch.no_grad():
        bare_states = base(input_ids=ids, output_hidden_states=True).hidden_states
    peft_model = _trained(base)
    with torch.no_grad():
        dual_states = peft_model(input_ids=ids, output_hidden_states=True).hidden_states
    # Entry l (l < N) is the base-stream input to layer l in both paths.
    for b, d in zip(bare_states[:-1], dual_states[:-1]):
        assert torch.equal(b, d)


def test_disabled_adapter_equals_bare(base, ids):
    with torch.no_grad():
        bare = base(input_ids=ids).logits
    peft_model = _trained(base)
    with torch.no_grad():
        with peft_model.disable_adapter():
            disabled = peft_model(input_ids=ids).logits
        active = peft_model(input_ids=ids).logits
    torch.testing.assert_close(bare, disabled, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(active, bare, atol=1e-3)


def test_only_cross_stream_and_gamma_train_and_all_get_grad(base, ids):
    peft_model = _trained(base).train()
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    assert trainable, "nothing trainable"
    for n in trainable:
        assert ("cross_stream.lora_" in n) or ("readout_norm.modules_to_save.default" in n), n
    n_layers = base.config.num_hidden_layers
    assert sum("readout_norm" in n for n in trainable) == n_layers
    assert sum("cross_stream.lora_A" in n for n in trainable) == n_layers
    out = peft_model(input_ids=ids, labels=ids)
    out.loss.backward()
    for n, p in peft_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad for {n}"


def test_save_load_round_trip(base, ids, tmp_path):
    peft_model = _trained(base)
    with torch.no_grad():
        want = peft_model(input_ids=ids).logits
    peft_model.save_pretrained(tmp_path)

    from safetensors.torch import load_file
    saved = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert any("readout_norm" in k for k in saved), sorted(saved)[:5]

    torch.manual_seed(123)  # a fresh base with different random init would differ...
    fresh = ShadowResidualForCausalLM(_config())
    fresh.load_state_dict(base.state_dict(), strict=False)  # ...so copy base weights
    loaded = PeftModel.from_pretrained(fresh.eval(), tmp_path).eval()
    with torch.no_grad():
        got = loaded(input_ids=ids).logits
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_gradient_checkpointing_exact(ids):
    torch.manual_seed(0)
    a = _trained(ShadowResidualForCausalLM(_config())).train()
    torch.manual_seed(0)
    b = _trained(ShadowResidualForCausalLM(_config())).train()
    b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    b.enable_input_require_grads()
    la = a(input_ids=ids, labels=ids).loss
    lb = b(input_ids=ids, labels=ids).loss
    la.backward()
    lb.backward()
    torch.testing.assert_close(la, lb)
    ga = {n: p.grad for n, p in a.named_parameters() if p.requires_grad}
    gb = {n: p.grad for n, p in b.named_parameters() if p.requires_grad}
    assert ga.keys() == gb.keys()
    for n in ga:
        torch.testing.assert_close(ga[n], gb[n], atol=1e-6, rtol=1e-5, msg=n)


def test_kv_cache_generation_matches_no_cache(base, ids):
    peft_model = _trained(base)
    with torch.no_grad():
        cached = peft_model.generate(input_ids=ids, max_new_tokens=6, do_sample=False, use_cache=True)
        seq = ids
        for _ in range(6):
            nxt = peft_model(input_ids=seq, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
    assert torch.equal(cached, seq)


# --- factory / config ----------------------------------------------------------


def _tiny_upstream():
    cfg = GraniteMoeHybridConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        layers_block_type=["attention", "attention"], num_local_experts=0,
        attention_multiplier=1.0, embedding_multiplier=1.0, residual_multiplier=1.0,
        logits_scaling=1.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return GraniteMoeHybridForCausalLM(cfg).eval()


def _patch(monkeypatch, upstream):
    import transformers
    from shadow_residual.shadow_residual import build as build_mod
    from shadow_residual.training import factory as factory_mod

    class _Cfg:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return upstream.config

    class _Model:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return upstream

    monkeypatch.setattr(factory_mod, "AutoModelForCausalLM", _Model)
    monkeypatch.setattr(build_mod, "AutoModelForCausalLM", _Model)
    monkeypatch.setattr(transformers, "AutoConfig", _Cfg)


@pytest.mark.parametrize("lora_kw, match", [
    (dict(target_modules=["cross_stream", "q_proj"], modules_to_save=["readout_norm"]), "cross_stream"),
    (dict(target_modules=["q_proj"], modules_to_save=["readout_norm"]), "cross_stream"),
    (dict(target_modules=["cross_stream"], modules_to_save=None), "readout_norm"),
])
def test_factory_rejects_bad_readout_lora(lora_kw, match):
    from shadow_residual.training.factory import get_shadow_residual_peft_model
    with pytest.raises(ValueError, match=match):
        get_shadow_residual_peft_model(
            "ignored", LoraConfig(r=2, lora_alpha=4, **lora_kw), adapter_stream="readout",
        )


def test_factory_meta_materialize_resets_gamma_and_keeps_fp32(monkeypatch):
    from shadow_residual.training.factory import get_shadow_residual_peft_model
    _patch(monkeypatch, _tiny_upstream())
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    model = get_shadow_residual_peft_model(
        "ignored", _lora(), torch_dtype=torch.bfloat16, adapter_stream="readout",
    )
    norms = [(n, p) for n, p in model.named_parameters() if "readout_norm" in n]
    assert len(norms) == 2 * 2  # frozen original + trainable copy, per layer
    for n, p in norms:
        assert torch.equal(p.float(), torch.ones_like(p, dtype=torch.float32)), n
        if p.requires_grad:
            assert p.dtype == torch.float32, n
        assert ("modules_to_save" in n) == p.requires_grad, n
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert torch.isfinite(out).all()


def test_untrained_readout_is_normed_embedding_model(monkeypatch):
    """At init (B=0, gamma=1) the head sees RMSNorm^N of the embeddings only —
    the documented difference from vanilla SR, whose init equals the base."""
    from shadow_residual.training.factory import get_shadow_residual_peft_model
    up = _tiny_upstream()
    _patch(monkeypatch, up)
    monkeypatch.setenv("LOCAL_RANK", "0")
    model = get_shadow_residual_peft_model("ignored", _lora(), adapter_stream="readout").eval()
    inner = model.base_model.model
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        h = inner.model.embed_tokens(ids) * inner.config.embedding_multiplier
        for layer in inner.model.layers:
            h = layer.readout_norm(h)
        want = inner.lm_head(inner.model.norm(h)) / inner.config.logits_scaling
        got = model(input_ids=ids).logits
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_training_config_readout_validation():
    from shadow_residual.config.adapters import to_peft_config
    from shadow_residual.config.training_config import TrainingConfig

    base = {
        "model": {"base": "x"},
        "data": {"train_path": "a", "val_path": "b"},
        "save": {"output_dir": "o"},
    }
    cfg = TrainingConfig.model_validate(
        {**base, "adapter": {"adapter_stream": "readout", "target_modules": {"cross_stream": 256}, "alpha": 512}}
    )
    lc = to_peft_config(cfg)
    assert lc.modules_to_save == ["readout_norm"]
    assert set(lc.target_modules) == {"cross_stream"}
    assert lc.rank_pattern == {"cross_stream": 256}
    for bad in (32, {"cross_stream": 32, "q_proj": 32}):
        with pytest.raises(ValueError, match="readout"):
            TrainingConfig.model_validate(
                {**base, "adapter": {"adapter_stream": "readout", "target_modules": bad}}
            )
    full = TrainingConfig.model_validate(base)
    assert full.adapter.adapter_stream == "full"
    assert to_peft_config(full).modules_to_save is None


def test_readout_yaml_loads(monkeypatch):
    from pathlib import Path
    from shadow_residual.config import load_training_config
    for var in ("BASE_MODEL", "DATA_ROOT", "RUN_DIR"):
        monkeypatch.setenv(var, "/x")
    path = Path(__file__).resolve().parents[2] / "src/shadow_residual/config/sr-readout-c256-g42.yaml"
    cfg = load_training_config(path)
    assert cfg.adapter.adapter_stream == "readout"
    assert cfg.adapter.target_modules == {"cross_stream": 256}


def test_read_adapter_stream_from_adapter(tmp_path):
    import json
    from shadow_residual.training.generation_utils import read_adapter_stream_from_adapter
    assert read_adapter_stream_from_adapter(str(tmp_path)) == "full"
    (tmp_path / "adapter_config.json").write_text(json.dumps({"adapter_stream": "readout"}))
    assert read_adapter_stream_from_adapter(str(tmp_path)) == "readout"
