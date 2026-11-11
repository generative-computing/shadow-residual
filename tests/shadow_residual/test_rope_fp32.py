# SPDX-License-Identifier: Apache-2.0
"""RoPE frequencies stay fp32 when the SR model is built in bf16.

The rotary embedding's ``inv_freq`` / ``original_inv_freq`` buffers are
non-persistent, so ``state_dict()`` comparisons never see them, and a
whole-module ``.to(bfloat16)`` silently rounds them. Stock HF
``from_pretrained(dtype=bfloat16)`` keeps them fp32; bf16 frequencies drift the
rotation angle by radians at long positions. These CPU tests cover every SR
build path (serving, the training factory on single-process / FSDP rank 0 / FSDP
meta ranks), the FSDP1 mixed-precision buffer cast, and train/serve parity.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, GraniteConfig, GraniteForCausalLM

from shadow_residual.shadow_residual.build import build_sr_base, reinit_rope_buffers


IDS = torch.tensor([[1, 5, 9, 13, 2, 7, 11]])


@pytest.fixture
def upstream(tmp_path):
    """A saved tiny plain-``granite`` base (the granite-4.1 shape)."""
    cfg = GraniteConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=1e7,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    path = tmp_path / "upstream"
    GraniteForCausalLM(cfg).eval().save_pretrained(path)
    return str(path)


def _assert_rope_fp32(model):
    """Every RoPE frequency buffer is fp32 and equals a fresh fp32 compute."""
    rotaries = [m for m in model.modules() if "inv_freq" in m._buffers]
    assert len(rotaries) == 1
    rotary = rotaries[0]
    expected = type(rotary)(config=rotary.config, device=torch.device("cpu"))
    for name in ("inv_freq", "original_inv_freq"):
        buf = getattr(rotary, name)
        assert buf.dtype == torch.float32, (name, buf.dtype)
        if not buf.is_meta:
            assert torch.equal(buf, getattr(expected, name)), name


def _build(base, monkeypatch, *, rank=None, dtype=torch.bfloat16):
    from shadow_residual.training.factory import get_shadow_residual_peft_model

    if rank is None:
        monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "0")
    else:
        monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
        monkeypatch.setenv("LOCAL_RANK", str(rank))
    torch.manual_seed(1)
    lora = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "o_proj", "cross_stream"])
    return get_shadow_residual_peft_model(base, lora, torch_dtype=dtype)


def test_stock_hf_keeps_rope_fp32(upstream):
    """The reference behavior SR matches: stock bf16 loading keeps fp32 RoPE."""
    stock = AutoModelForCausalLM.from_pretrained(upstream, dtype=torch.bfloat16)
    assert stock.model.rotary_emb.inv_freq.dtype == torch.float32


def test_rope_fp32_serving_build(upstream):
    _assert_rope_fp32(build_sr_base(upstream, torch_dtype=torch.bfloat16))


@pytest.mark.parametrize("rank", [None, 0, 1], ids=["single-process", "fsdp-rank0", "fsdp-rank1-meta"])
def test_rope_fp32_factory_builds(upstream, monkeypatch, rank):
    """Training builds: every factory path."""
    model = _build(upstream, monkeypatch, rank=rank, dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    _assert_rope_fp32(model)


def test_rope_fp32_bf16_train_matches_serve(upstream, monkeypatch):
    """A bf16 adapter sees identical bare logits at train and serve time."""
    trained = _build(upstream, monkeypatch, dtype=torch.bfloat16)
    served = build_sr_base(upstream, torch_dtype=torch.bfloat16)
    with torch.no_grad(), trained.disable_adapter():
        train_logits = trained(input_ids=IDS).logits
    with torch.no_grad():
        serve_logits = served(input_ids=IDS).logits
    assert torch.equal(train_logits, serve_logits)


def test_reinit_rope_buffers_upgrades_bf16_to_fp32(upstream):
    """A bf16 buffer (e.g. after a plain ``.to``) is recomputed as exact fp32."""
    model = build_sr_base(upstream).to(dtype=torch.bfloat16)
    assert model.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    reinit_rope_buffers(model)
    _assert_rope_fp32(model)


def test_fsdp1_mixed_precision_keeps_buffers_unconverted():
    """accelerate's FSDP1 bf16 policy would cast buffers; train.py unsets that."""
    from types import SimpleNamespace

    from torch.distributed.fsdp import MixedPrecision

    from shadow_residual.training.train import _keep_fsdp_buffers_unconverted

    policy = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
    )
    plugin = SimpleNamespace(fsdp_version=1, mixed_precision_policy=policy)
    _keep_fsdp_buffers_unconverted(SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin)))
    assert policy.buffer_dtype is None
    assert policy.param_dtype == torch.bfloat16

    # No FSDP plugin → no-op.
    _keep_fsdp_buffers_unconverted(SimpleNamespace(state=SimpleNamespace(fsdp_plugin=None)))
