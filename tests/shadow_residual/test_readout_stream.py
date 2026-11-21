# SPDX-License-Identifier: Apache-2.0
"""Attention-free / read-out SR (``adapter_stream`` = readout[+…] / layermix[+…]) invariants, CPU only.

The read-out adapter stream is
``h_{l+1} = c_l + branch_l + W_cross_l(h_base_out_l)`` with ``c_l = readout_norm_l(h_l)``,
seeded with the embeddings, head on ``norm(h_N)``. ``branch`` is empty for plain
``readout``; the ablations add adapter attention on the shared base K/V, the
adapter MLP, or a dense layer on ``c``. See docs/sr_readout_stream_design.html.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual._stream_gated_linear import _base_only
from shadow_residual.shadow_residual.config_helpers import set_shadow_residual
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig

STREAM_TARGETS = {
    "readout": ["cross_stream"],
    "readout+attn": ["cross_stream", "q_proj", "o_proj"],
    "readout+mlp": ["cross_stream", "gate_proj", "up_proj", "down_proj"],
    "readout+dense": ["cross_stream", "readout_dense"],
    "readout+pool": ["cross_stream", "readout_pool"],
    "readout+heads": ["cross_stream", "readout_heads"],
    "layermix": ["cross_stream"],
    "layermix+pool": ["cross_stream", "readout_pool"],
    "layermix+heads": ["cross_stream", "readout_heads"],
}
STREAMS = list(STREAM_TARGETS)
ZERO_SITES = ("readout_dense", "readout_pool", "readout_heads")


def _is_readout(stream):
    return stream.split("+")[0] == "readout"


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


def _lora(stream="readout", **kw):
    return LoraConfig(
        r=4, lora_alpha=8, target_modules=STREAM_TARGETS[stream],
        modules_to_save=["readout_norm"] if _is_readout(stream) else None,
        lora_dropout=0.0, **kw,
    )


def _trained(model, stream="readout"):
    """Wrap with the read-out adapter and give every trainable a nonzero value."""
    peft_model = get_peft_model(model, _lora(stream))
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


@pytest.fixture(params=STREAMS)
def stream(request):
    return request.param


@pytest.fixture
def base(stream):
    torch.manual_seed(0)
    return ShadowResidualForCausalLM(_config(stream)).eval()


def test_full_mode_has_no_readout_modules():
    model = ShadowResidualForCausalLM(_config("full"))
    assert not any("readout" in n for n, _ in model.named_parameters())


def test_readout_modules_frozen_identity_on_bare(base, stream):
    for layer in base.model.layers:
        if _is_readout(stream):
            assert torch.equal(layer.readout_norm.weight, torch.ones_like(layer.readout_norm.weight))
            assert not layer.readout_norm.weight.requires_grad
        else:
            assert not hasattr(layer, "readout_norm")
        for site in ZERO_SITES:
            if site in STREAM_TARGETS[stream]:
                w = getattr(layer, site).weight
                assert torch.equal(w, torch.zeros_like(w)) and not w.requires_grad
            else:
                assert not hasattr(layer, site)


def test_attn_with_taps_rejected():
    from shadow_residual.shadow_residual.adapter_streams import ADAPTER_STREAMS
    assert not any("attn" in s and ("pool" in s or "heads" in s) for s in ADAPTER_STREAMS)


def test_training_config_literal_matches_grammar():
    import typing
    from shadow_residual.config.training_config import AdapterConfig
    from shadow_residual.shadow_residual.adapter_streams import ADAPTER_STREAMS
    literal = AdapterConfig.model_fields["adapter_stream"].annotation
    assert set(typing.get_args(literal)) == set(ADAPTER_STREAMS)


@pytest.mark.parametrize("stream", [s for s in STREAMS if not _is_readout(s)])
def test_untrained_layermix_equals_base(stream, ids):
    """layermix is base-seeded: with zero-init LoRA B the logits equal the bare base."""
    torch.manual_seed(0)
    base = ShadowResidualForCausalLM(_config(stream)).eval()
    with torch.no_grad():
        bare = base(input_ids=ids).logits
        wrapped = get_peft_model(base, _lora(stream)).eval()
        got = wrapped(input_ids=ids).logits
    assert torch.equal(got, bare)


@pytest.mark.parametrize("bad", ["bogus", "readout+bogus"])
def test_unknown_adapter_stream_rejected(bad):
    with pytest.raises(ValueError, match="adapter_stream"):
        ShadowResidualForCausalLM(_config(bad))


def _base_trace(inner, ids):
    """Bare base-stream (input, output) per layer, plus the mask/rope it used."""
    h = inner.model.embed_tokens(ids) * inner.config.embedding_multiplier
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    pe = inner.model.rotary_emb(h, position_ids=pos)
    mask = torch.full((ids.shape[1], ids.shape[1]), float("-inf")).triu(1)[None, None]
    trace = []
    for layer in inner.model.layers:
        out = layer._base_layer(
            h, attention_mask=mask, past_key_values=None, use_cache=False,
            cache_position=None, position_embeddings=pe, base_only=True,
        )
        trace.append((h, out))
        h = out
    return trace, mask, pe


def _ref_adapter_attn(layer, h, hb_in, mask, pe):
    """Adapter Q (wrapped q_proj) against base-only K/V, wrapped o_proj — by hand."""
    attn = layer.self_attn
    xa, xb = layer.input_layernorm(h), layer.input_layernorm(hb_in)
    q = attn.q_proj(xa)
    k, v = _base_only(attn.k_proj, xb), _base_only(attn.v_proj, xb)
    b, s, _ = q.shape
    q = q.view(b, s, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(b, s, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = v.view(b, s, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import apply_rotary_pos_emb
    q, k = apply_rotary_pos_emb(q, k, *pe)
    k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
    v = v.repeat_interleave(attn.num_key_value_groups, dim=1)
    w = torch.softmax(q @ k.transpose(-1, -2) * attn.scaling + mask, dim=-1)
    o = (w @ v).transpose(1, 2).reshape(b, s, -1)
    return attn.o_proj(o)


def _ref_base_taps(layer, hb_in, mask, pe):
    """Base per-head attention outputs (pre o_proj) and the uniform causal mean of V, by hand."""
    attn = layer.self_attn
    x = layer.input_layernorm(hb_in)
    q, k, v = (_base_only(p, x) for p in (attn.q_proj, attn.k_proj, attn.v_proj))
    b, s, _ = q.shape
    q = q.view(b, s, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(b, s, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = v.view(b, s, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import apply_rotary_pos_emb
    q, k = apply_rotary_pos_emb(q, k, *pe)
    k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
    v = v.repeat_interleave(attn.num_key_value_groups, dim=1)
    w = torch.softmax(q @ k.transpose(-1, -2) * attn.scaling + mask, dim=-1)
    heads = (w @ v).transpose(1, 2).reshape(b, s, -1)
    counts = torch.arange(1, s + 1, dtype=v.dtype).view(1, 1, s, 1)
    pool = (v.cumsum(dim=2) / counts).transpose(1, 2).reshape(b, s, -1)
    return heads, pool


def test_logits_match_reference_recurrence(base, stream, ids):
    """Model logits == the hand-written read-out recurrence over bare base layers."""
    peft_model = _trained(base, stream)
    inner = peft_model.base_model.model
    rm = inner.config.residual_multiplier
    with torch.no_grad():
        got = peft_model(input_ids=ids).logits
        trace, mask, pe = _base_trace(inner, ids)
        h = inner.model.embed_tokens(ids) * inner.config.embedding_multiplier
        for layer, (hb_in, hb_out) in zip(inner.model.layers, trace):
            c = layer.readout_norm(h) if _is_readout(stream) else hb_out + (h - hb_in)
            new = c + layer.cross_stream(hb_out)
            if stream.endswith("+pool") or stream.endswith("+heads"):
                heads, pool = _ref_base_taps(layer, hb_in, mask, pe)
                if stream.endswith("+pool"):
                    new = new + layer.readout_pool(pool)
                else:
                    new = new + layer.readout_heads(heads)
            if stream == "readout+attn":
                new = new + rm * _ref_adapter_attn(layer, h, hb_in, mask, pe)
            if stream == "readout+mlp":
                new = new + rm * layer.mlp(layer.post_attention_layernorm(h))
            if stream == "readout+dense":
                new = new + layer.readout_dense(c)
            h = new
        want = inner.lm_head(inner.model.norm(h)) / inner.config.logits_scaling
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_base_stream_bit_identical_to_bare(base, stream, ids):
    """Invariant 1: with a trained read-out adapter the base stream after every
    layer equals the unadapted base's layer outputs exactly."""
    with torch.no_grad():
        bare_states = base(input_ids=ids, output_hidden_states=True).hidden_states
    peft_model = _trained(base, stream)
    with torch.no_grad():
        dual_states = peft_model(input_ids=ids, output_hidden_states=True).hidden_states
    # Entry l (l < N) is the base-stream input to layer l in both paths.
    for b, d in zip(bare_states[:-1], dual_states[:-1]):
        assert torch.equal(b, d)


def test_disabled_adapter_equals_bare(base, stream, ids):
    with torch.no_grad():
        bare = base(input_ids=ids).logits
    peft_model = _trained(base, stream)
    with torch.no_grad():
        with peft_model.disable_adapter():
            disabled = peft_model(input_ids=ids).logits
        active = peft_model(input_ids=ids).logits
    torch.testing.assert_close(bare, disabled, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(active, bare, atol=1e-3)


def test_only_allowed_params_train_and_all_get_grad(base, stream, ids):
    peft_model = _trained(base, stream).train()
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    assert trainable, "nothing trainable"
    lora_sites = STREAM_TARGETS[stream]
    for n in trainable:
        ok = "readout_norm.modules_to_save.default" in n or any(
            f"{site}.lora_" in n for site in lora_sites
        )
        assert ok, n
    n_layers = base.config.num_hidden_layers
    assert sum("readout_norm" in n for n in trainable) == (n_layers if _is_readout(stream) else 0)
    for site in lora_sites:
        assert sum(f"{site}.lora_A" in n for n in trainable) == n_layers, site
    out = peft_model(input_ids=ids, labels=ids)
    out.loss.backward()
    for n, p in peft_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad for {n}"


def test_save_load_round_trip(base, stream, ids, tmp_path):
    # Snapshot before get_peft_model renames wrapped weights to ``*.base_layer.*``.
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    peft_model = _trained(base, stream)
    with torch.no_grad():
        want = peft_model(input_ids=ids).logits
    peft_model.save_pretrained(tmp_path)

    from safetensors.torch import load_file
    saved = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert any("readout_norm" in k for k in saved) == _is_readout(stream), sorted(saved)[:5]
    for site in STREAM_TARGETS[stream]:
        assert any(f"{site}.lora_B" in k for k in saved), site

    torch.manual_seed(123)  # a fresh base with different random init would differ...
    fresh = ShadowResidualForCausalLM(_config(stream))
    fresh.load_state_dict(base_state)  # ...so copy base weights
    loaded = PeftModel.from_pretrained(fresh.eval(), tmp_path).eval()
    with torch.no_grad():
        got = loaded(input_ids=ids).logits
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_gradient_checkpointing_exact(stream, ids):
    torch.manual_seed(0)
    a = _trained(ShadowResidualForCausalLM(_config(stream)), stream).train()
    torch.manual_seed(0)
    b = _trained(ShadowResidualForCausalLM(_config(stream)), stream).train()
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


def test_kv_cache_generation_matches_no_cache(base, stream, ids):
    peft_model = _trained(base, stream)
    with torch.no_grad():
        cached = peft_model.generate(input_ids=ids, max_new_tokens=6, do_sample=False, use_cache=True)
        seq = ids
        for _ in range(6):
            nxt = peft_model(input_ids=seq, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
    assert torch.equal(cached, seq)


def test_left_padded_batch_matches_unpadded(base, stream):
    """Batched eval left-pads; padding must not leak into the read-out (e.g. the pool mean)."""
    peft_model = _trained(base, stream)
    short, long = [5, 17, 42], [9, 8, 7, 6, 5]
    pad = 0
    ids = torch.tensor([[pad, pad] + short, long])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    with torch.no_grad():
        batched = peft_model(input_ids=ids, attention_mask=mask).logits
        alone = peft_model(input_ids=torch.tensor([short])).logits
    torch.testing.assert_close(batched[0, 2:], alone[0], atol=1e-5, rtol=1e-4)


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


@pytest.mark.parametrize("adapter_stream, lora_kw, match", [
    ("readout", dict(target_modules=["cross_stream", "q_proj"], modules_to_save=["readout_norm"]), "cross_stream"),
    ("readout", dict(target_modules=["q_proj"], modules_to_save=["readout_norm"]), "cross_stream"),
    ("readout", dict(target_modules=["cross_stream"], modules_to_save=None), "readout_norm"),
    ("readout+mlp", dict(target_modules=["cross_stream", "q_proj"], modules_to_save=["readout_norm"]), "q_proj"),
    ("readout+attn", dict(target_modules=["cross_stream", "readout_dense"], modules_to_save=["readout_norm"]), "readout_dense"),
    ("readout+dense", dict(target_modules=["cross_stream"], modules_to_save=["readout_norm"]), "readout_dense"),
    ("readout+pool", dict(target_modules=["cross_stream"], modules_to_save=["readout_norm"]), "readout_pool"),
    ("layermix", dict(target_modules=["cross_stream"], modules_to_save=["readout_norm"]), "readout_norm"),
    ("layermix+heads", dict(target_modules=["cross_stream", "readout_pool"], modules_to_save=None), "readout_heads"),
])
def test_factory_rejects_bad_readout_lora(adapter_stream, lora_kw, match):
    from shadow_residual.training.factory import get_shadow_residual_peft_model
    with pytest.raises(ValueError, match=match):
        get_shadow_residual_peft_model(
            "ignored", LoraConfig(r=2, lora_alpha=4, **lora_kw), adapter_stream=adapter_stream,
        )


def test_factory_meta_materialize_resets_gamma_and_keeps_fp32(monkeypatch, stream):
    from shadow_residual.training.factory import get_shadow_residual_peft_model
    _patch(monkeypatch, _tiny_upstream())
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    model = get_shadow_residual_peft_model(
        "ignored", _lora(stream), torch_dtype=torch.bfloat16, adapter_stream=stream,
    )
    norms = [(n, p) for n, p in model.named_parameters() if "readout_norm" in n]
    # frozen original + trainable copy, per layer (readout carry only)
    assert len(norms) == (2 * 2 if _is_readout(stream) else 0)
    for n, p in norms:
        assert torch.equal(p.float(), torch.ones_like(p, dtype=torch.float32)), n
        if p.requires_grad:
            assert p.dtype == torch.float32, n
        assert ("modules_to_save" in n) == p.requires_grad, n
    for site in ZERO_SITES:
        ws = [p for n, p in model.named_parameters() if f"{site}.base_layer" in n]
        assert len(ws) == (2 if site in STREAM_TARGETS[stream] else 0), site
        for p in ws:
            assert torch.equal(p, torch.zeros_like(p)) and not p.requires_grad
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("stream", ["layermix", "layermix+pool", "layermix+heads"])
def test_factory_untrained_layermix_matches_upstream(monkeypatch, stream):
    """Through the real meta-materialize path, an untrained layermix adapter == the upstream base."""
    from shadow_residual.training.factory import get_shadow_residual_peft_model
    up = _tiny_upstream()
    _patch(monkeypatch, up)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        want = up(input_ids=ids).logits
    model = get_shadow_residual_peft_model("ignored", _lora(stream), adapter_stream=stream).eval()
    with torch.no_grad():
        got = model(input_ids=ids).logits
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize("stream, ok, bad", [
    ("readout+attn", {"cross_stream": 256, "q_proj": 32, "o_proj": 32}, {"cross_stream": 256, "up_proj": 32}),
    ("readout+mlp", {"cross_stream": 256, "gate_proj": 32, "up_proj": 32, "down_proj": 32},
     {"cross_stream": 256, "q_proj": 32}),
    ("readout+dense", {"cross_stream": 256, "readout_dense": 64}, {"cross_stream": 256}),
    ("readout+pool", {"cross_stream": 256, "readout_pool": 64}, {"cross_stream": 256}),
    ("readout+heads", {"cross_stream": 256, "readout_heads": 64}, {"cross_stream": 256, "readout_pool": 64}),
    ("layermix", {"cross_stream": 32}, {"cross_stream": 32, "readout_dense": 32}),
    ("layermix+pool", {"cross_stream": 256, "readout_pool": 64}, {"cross_stream": 256}),
    ("layermix+heads", {"cross_stream": 256, "readout_heads": 64}, {"readout_heads": 64}),
])
def test_training_config_readout_ablation_validation(stream, ok, bad):
    from shadow_residual.config.adapters import to_peft_config
    from shadow_residual.config.training_config import TrainingConfig

    base = {"model": {"base": "x"}, "data": {"train_path": "a", "val_path": "b"}, "save": {"output_dir": "o"}}
    alpha = {k: 2 * v for k, v in ok.items()}
    cfg = TrainingConfig.model_validate(
        {**base, "adapter": {"adapter_stream": stream, "target_modules": ok, "alpha": alpha}}
    )
    lc = to_peft_config(cfg)
    assert lc.modules_to_save == (["readout_norm"] if _is_readout(stream) else None)
    assert lc.rank_pattern == ok and lc.alpha_pattern == alpha
    with pytest.raises(ValueError, match="adapter_stream"):
        TrainingConfig.model_validate({**base, "adapter": {"adapter_stream": stream, "target_modules": bad}})


@pytest.mark.parametrize("name, stream, targets", [
    ("sr-readout-c256-g42.yaml", "readout", {"cross_stream": 256}),
    ("sr-readout-attn-c256-qo32-g42.yaml", "readout+attn", {"cross_stream": 256, "q_proj": 32, "o_proj": 32}),
    ("sr-readout-mlp-c256-mlp32-g42.yaml", "readout+mlp",
     {"cross_stream": 256, "gate_proj": 32, "up_proj": 32, "down_proj": 32}),
    ("sr-readout-dense-c256-d64-g42.yaml", "readout+dense", {"cross_stream": 256, "readout_dense": 64}),
    ("sr-readout-pool-c256-p64-g42.yaml", "readout+pool", {"cross_stream": 256, "readout_pool": 64}),
    ("sr-readout-heads-c256-h64-g42.yaml", "readout+heads", {"cross_stream": 256, "readout_heads": 64}),
    ("sr-layermix-c256-g42.yaml", "layermix", {"cross_stream": 256}),
    ("sr-layermix-c32-g42.yaml", "layermix", {"cross_stream": 32}),
    ("sr-layermix-pool-c256-p64-g42.yaml", "layermix+pool", {"cross_stream": 256, "readout_pool": 64}),
    ("sr-layermix-heads-c256-h64-g42.yaml", "layermix+heads", {"cross_stream": 256, "readout_heads": 64}),
])
def test_readout_yaml_loads(monkeypatch, name, stream, targets):
    from pathlib import Path
    from shadow_residual.config import load_training_config
    from shadow_residual.config.adapters import to_peft_config
    for var in ("BASE_MODEL", "DATA_ROOT", "RUN_DIR"):
        monkeypatch.setenv(var, "/x")
    path = Path(__file__).resolve().parents[2] / "src/shadow_residual/config" / name
    cfg = load_training_config(path)
    assert cfg.adapter.adapter_stream == stream
    assert cfg.adapter.target_modules == targets
    lc = to_peft_config(cfg)
    scales = {k: lc.alpha_pattern.get(k, lc.lora_alpha) / r for k, r in lc.rank_pattern.items()} \
        if lc.alpha_pattern else {k: lc.lora_alpha / r for k, r in lc.rank_pattern.items()}
    assert set(scales.values()) == {2.0}, scales


def test_read_adapter_stream_from_adapter(tmp_path):
    import json
    from shadow_residual.training.generation_utils import read_adapter_stream_from_adapter
    assert read_adapter_stream_from_adapter(str(tmp_path)) == "full"
    (tmp_path / "adapter_config.json").write_text(json.dumps({"adapter_stream": "readout+attn"}))
    assert read_adapter_stream_from_adapter(str(tmp_path)) == "readout+attn"
