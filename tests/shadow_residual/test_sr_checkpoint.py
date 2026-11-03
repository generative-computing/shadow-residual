# SPDX-License-Identifier: Apache-2.0
"""SR checkpoints: convert an upstream base once, then load with ``from_pretrained``.

Covers the standalone-checkpoint path end to end on tiny CPU models (no HF
download):

- ``convert_to_sr_checkpoint`` writes a ``model_type: "shadow_residual"``
  checkpoint without the frozen all-zero cross-stream weights, plus the
  upstream generation config and a model card.
- Stock ``AutoConfig`` / ``AutoModelForCausalLM.from_pretrained`` resolve it to
  the SR classes (auto-class registration), and the bare model still reproduces
  the upstream logits — invariant 3 holds across the round trip.
- The training factory builds the same PEFT model from the SR checkpoint as from
  the upstream base, on the single-process path and the FSDP rank-0 / meta paths.
- ``build_sr_base`` (serving) accepts either kind of base.
- Loading is strict: a checkpoint whose weights don't match its config raises.
"""

from __future__ import annotations

import json

import pytest
import torch
from peft import LoraConfig
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GraniteConfig,
    GraniteForCausalLM,
    GraniteMoeConfig,
    GraniteMoeForCausalLM,
)

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.build import (
    build_sr_base,
    is_sr_checkpoint,
    load_sr_checkpoint,
)
from shadow_residual.shadow_residual.convert import convert_to_sr_checkpoint
from shadow_residual.shadow_residual.cross_stream import CrossStream
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig


IDS = torch.tensor([[1, 5, 9, 13, 2, 7, 11]])


def _dense_config(*, tie_word_embeddings: bool) -> GraniteConfig:
    """Tiny plain-``granite`` config (the granite-4.1 shape: unfused, dense)."""
    return GraniteConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_multiplier=0.25,
        embedding_multiplier=12.0,
        residual_multiplier=0.22,
        logits_scaling=10.0,
        rms_norm_eps=1e-5,
        rope_theta=1e7,
        tie_word_embeddings=tie_word_embeddings,
    )


def _moe_config() -> GraniteMoeConfig:
    return GraniteMoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )


def _save_upstream(model_cls, cfg, path):
    torch.manual_seed(0)
    model = model_cls(cfg).eval()
    if not cfg.tie_word_embeddings:
        # A genuinely distinct head, so a test can tell it from the embedding.
        with torch.no_grad():
            model.lm_head.weight.normal_(0.0, 0.05)
    with torch.no_grad():
        logits = model(input_ids=IDS).logits
    model.save_pretrained(path)
    return logits


@pytest.fixture(params=["dense-tied", "dense-untied", "moe"])
def converted(request, tmp_path):
    """An upstream tiny base, its logits, and its converted SR checkpoint."""
    upstream = tmp_path / "upstream"
    if request.param == "moe":
        logits = _save_upstream(GraniteMoeForCausalLM, _moe_config(), upstream)
    else:
        cfg = _dense_config(tie_word_embeddings=request.param == "dense-tied")
        logits = _save_upstream(GraniteForCausalLM, cfg, upstream)
    sr_dir = convert_to_sr_checkpoint(str(upstream), tmp_path / "sr", repo_id="someone/tiny-sr")
    return {"upstream": str(upstream), "sr": str(sr_dir), "logits": logits, "kind": request.param}


# ── Conversion output ───────────────────────────────────────────────


def test_checkpoint_layout(converted):
    sr_dir = converted["sr"]
    cfg = json.load(open(f"{sr_dir}/config.json"))
    assert cfg["model_type"] == "shadow_residual"
    assert cfg["architectures"] == ["ShadowResidualForCausalLM"]
    assert cfg["shadow_residual"] is True
    assert "auto_map" not in cfg, "SR checkpoints must not need trust_remote_code"

    with safe_open(f"{sr_dir}/model.safetensors", "pt") as f:
        keys = set(f.keys())
    assert not [k for k in keys if "cross_stream" in k], (
        "the frozen all-zero cross-stream weights must not be serialized"
    )
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert ("lm_head.weight" in keys) == (converted["kind"] != "dense-tied")

    card = open(f"{sr_dir}/README.md").read()
    assert 'from_pretrained("someone/tiny-sr")' in card
    gen_cfg = json.load(open(f"{sr_dir}/generation_config.json"))
    assert gen_cfg.get("bos_token_id") == cfg.get("bos_token_id")


def test_convert_refuses_sr_checkpoint(converted, tmp_path):
    with pytest.raises(ValueError, match="already an SR checkpoint"):
        convert_to_sr_checkpoint(converted["sr"], tmp_path / "again")


def test_convert_push_needs_repo_id(converted, tmp_path):
    with pytest.raises(ValueError, match="repo_id"):
        convert_to_sr_checkpoint(converted["upstream"], tmp_path / "x", push=True)


# ── Loading with stock from_pretrained ──────────────────────────────


def test_detection(converted):
    assert is_sr_checkpoint(converted["sr"])
    assert not is_sr_checkpoint(converted["upstream"])
    assert isinstance(AutoConfig.from_pretrained(converted["sr"]), ShadowResidualConfig)


def test_auto_from_pretrained_is_base_identical(converted):
    """Stock ``AutoModelForCausalLM.from_pretrained`` → bare SR == upstream logits."""
    model = AutoModelForCausalLM.from_pretrained(converted["sr"]).eval()
    assert isinstance(model, ShadowResidualForCausalLM)
    with torch.no_grad():
        logits = model(input_ids=IDS).logits
    torch.testing.assert_close(logits, converted["logits"], atol=1e-5, rtol=1e-4)

    sites = [m for m in model.modules() if isinstance(m, CrossStream)]
    assert len(sites) == model.config.num_hidden_layers
    assert all(torch.equal(m.weight, torch.zeros_like(m.weight)) for m in sites)
    assert not any(m.weight.requires_grad for m in sites)

    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert tied == (converted["kind"] == "dense-tied")


def test_load_matches_in_memory_conversion(converted):
    """The SR checkpoint and the in-memory conversion give identical models."""
    for dtype in (torch.float32, torch.bfloat16):
        from_ckpt = load_sr_checkpoint(converted["sr"], torch_dtype=dtype).state_dict()
        in_memory = build_sr_base(converted["upstream"], torch_dtype=dtype).state_dict()
        assert set(from_ckpt) == set(in_memory)
        for key, val in in_memory.items():
            assert from_ckpt[key].dtype == val.dtype, key
            assert torch.equal(from_ckpt[key], val), key


def test_build_sr_base_accepts_sr_checkpoint(converted):
    """Serving path: ``build_sr_base`` takes an SR checkpoint as the base id."""
    model = build_sr_base(converted["sr"]).eval()
    with torch.no_grad():
        logits = model(input_ids=IDS).logits
    torch.testing.assert_close(logits, converted["logits"], atol=1e-5, rtol=1e-4)


def test_load_is_strict(converted, tmp_path):
    """A checkpoint missing a real weight raises instead of random-initializing it."""
    from safetensors.torch import load_file, save_file
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(converted["sr"], broken)
    state = load_file(str(broken / "model.safetensors"))
    del state["model.layers.0.self_attn.q_proj.weight"]
    save_file(state, str(broken / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="does not match"):
        load_sr_checkpoint(str(broken))


def test_share_moe_routing_is_caller_controlled(converted):
    """The routing mode is a runtime choice even though config.json stores one."""
    for value in (True, False):
        model = load_sr_checkpoint(converted["sr"], share_moe_routing=value)
        assert model.config.share_moe_routing is value


# ── Training factory ────────────────────────────────────────────────


def _lora():
    return LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "o_proj", "cross_stream"])


def _build(base, monkeypatch, *, rank=None, dtype=torch.bfloat16):
    from shadow_residual.training.factory import get_shadow_residual_peft_model

    if rank is None:
        monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "0")
    else:
        monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
        monkeypatch.setenv("LOCAL_RANK", str(rank))
    torch.manual_seed(1)
    return get_shadow_residual_peft_model(base, _lora(), torch_dtype=dtype)


@pytest.mark.parametrize("rank", [None, 0], ids=["single-process", "fsdp-rank0"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_factory_sr_checkpoint_matches_upstream(converted, monkeypatch, rank, dtype):
    """Training from the SR checkpoint == training from the upstream base."""
    upstream = _build(converted["upstream"], monkeypatch, rank=rank, dtype=dtype)
    from_ckpt = _build(converted["sr"], monkeypatch, rank=rank, dtype=dtype)

    up_state, ckpt_state = upstream.state_dict(), from_ckpt.state_dict()
    assert set(up_state) == set(ckpt_state)
    for key, val in up_state.items():
        assert ckpt_state[key].dtype == val.dtype == dtype, key
        if "lora_A" not in key:  # lora_A is randomly initialized
            assert torch.equal(ckpt_state[key], val), key

    trainable = lambda m: {n for n, p in m.named_parameters() if p.requires_grad}  # noqa: E731
    assert trainable(from_ckpt) == trainable(upstream)
    assert all("lora_" in n for n in trainable(from_ckpt))

    inner = from_ckpt.base_model.model
    tied = inner.lm_head.weight.data_ptr() == inner.model.embed_tokens.weight.data_ptr()
    assert tied == (converted["kind"] == "dense-tied")
    with pytest.raises(NotImplementedError):
        from_ckpt.merge_and_unload()

    with torch.no_grad(), from_ckpt.disable_adapter():
        bare = from_ckpt(input_ids=IDS).logits
    with torch.no_grad(), upstream.disable_adapter():
        expected = upstream(input_ids=IDS).logits
    assert torch.equal(bare, expected)


def test_factory_sr_checkpoint_fsdp_non_rank0_stays_on_meta(converted, monkeypatch):
    """FSDP non-rank-0 builds the same meta skeleton, loading nothing."""
    from shadow_residual.shadow_residual import build as build_mod
    from shadow_residual.training import factory as factory_mod

    def _fail(*a, **kw):
        raise AssertionError("non-rank-0 must not load weights")

    monkeypatch.setattr(factory_mod, "_load_sr_checkpoint", _fail)
    monkeypatch.setattr(build_mod, "load_sr_checkpoint", _fail)
    meta = _build(converted["sr"], monkeypatch, rank=1)
    assert all(p.is_meta for p in meta.parameters())

    monkeypatch.undo()
    rank0 = _build(converted["sr"], monkeypatch, rank=0)
    assert [n for n, _ in meta.named_parameters()] == [n for n, _ in rank0.named_parameters()]
