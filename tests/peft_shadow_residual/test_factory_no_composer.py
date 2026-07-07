# SPDX-License-Identifier: Apache-2.0
"""Phase 3 verification: factory builds SR without the GraniteSwitch composer.

The new ``_build_sr_base`` loads via :class:`AutoModelForCausalLM`,
constructs an :class:`ShadowResidualForCausalLM` with the same config,
and slices fused→unfused projections via
:func:`transfer_base_weights`. There is no ``from_base_and_adapters_shadow_residual``
call, no ``SingleSwitch``, no ``adapter_token_ids`` buffer, no
``num_adapters=1`` placeholder.

These tests exercise that path against a tiny in-memory
``GraniteMoeHybridForCausalLM`` (no HF download). We bypass
``AutoModelForCausalLM.from_pretrained`` by monkeypatching it.
"""

from __future__ import annotations

import pytest
import torch
from transformers import (
    GraniteMoeHybridConfig,
    GraniteMoeHybridForCausalLM,
)

from shadow_residual.peft_shadow_residual.factory import _build_sr_base
from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual._stream_gated_linear import (
    _StreamGatedLinear,
)
from shadow_residual.shadow_residual.weight_transfer import (
    transfer_base_weights,
)


def _make_tiny_hybrid_config():
    """Tiny GraniteMoeHybrid config — all-attention layers (no mamba)."""
    return GraniteMoeHybridConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layers_block_type=["attention", "attention"],
        num_local_experts=0,  # dense, no MoE branch
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,  # matches granite-4.1 (upstream ties lm_head↔embed)
    )


@pytest.fixture
def tiny_base_model():
    cfg = _make_tiny_hybrid_config()
    torch.manual_seed(0)
    model = GraniteMoeHybridForCausalLM(cfg)
    model.eval()
    return model


def _patch_auto(monkeypatch, _factory_mod, tiny_base_model):
    """Patch both AutoConfig and AutoModelForCausalLM to return the fixture.

    ``_build_sr_config`` calls ``AutoConfig.from_pretrained`` (config-only,
    cheap on every rank); ``_build_sr_base`` then calls
    ``AutoModelForCausalLM.from_pretrained`` for the real weights. Tests
    must patch both symbols so they can run offline against the tiny
    in-memory fixture.
    """
    import transformers as _transformers_mod

    class _StubAutoConfig:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return tiny_base_model.config

    class _StubAuto:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return tiny_base_model

    monkeypatch.setattr(_factory_mod, "AutoModelForCausalLM", _StubAuto)
    # _build_sr_config does ``from transformers import AutoConfig`` at call
    # time, so patch the canonical attribute on the transformers module.
    monkeypatch.setattr(_transformers_mod, "AutoConfig", _StubAutoConfig)


def test_build_sr_base_via_monkeypatch(tiny_base_model, monkeypatch):
    """``_build_sr_base`` (single-process variant) builds an SR model.

    The factory normally calls ``AutoConfig.from_pretrained`` (for the
    config) and ``AutoModelForCausalLM.from_pretrained`` (for the
    weights); we monkey-patch both to return our tiny in-memory model so
    the test runs offline and fast.
    """
    from shadow_residual.peft_shadow_residual import factory as _factory_mod

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)

    sr = _build_sr_base("ignored-path")

    assert isinstance(sr, ShadowResidualForCausalLM)
    # No GraniteSwitch routing baggage on the new model.
    assert not hasattr(sr.model, "switch")
    assert not hasattr(sr.model, "adapter_token_ids")
    assert getattr(sr.config, "num_adapters", 0) == 0
    assert getattr(sr.config, "shadow_residual", False) is True

    # Projections are _StreamGatedLinear so PEFT can dispatch SR LoRA.
    layer0 = sr.model.layers[0].self_attn
    assert isinstance(layer0.q_proj, _StreamGatedLinear)
    assert isinstance(layer0.k_proj, _StreamGatedLinear)
    assert isinstance(layer0.v_proj, _StreamGatedLinear)
    assert isinstance(layer0.o_proj, _StreamGatedLinear)


def test_weight_transfer_attention_unfused(tiny_base_model):
    """Upstream HF Granite already exposes q/k/v_proj unfused; transfer is by name.

    (When the source is the open-source GraniteSwitch model, projections
    are fused as ``qkv_proj`` and the same helper slices them — that path
    is exercised indirectly via the existing SR composer tests.)
    """
    src = tiny_base_model
    src_cfg = src.config

    from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig
    from shadow_residual.shadow_residual.config_helpers import (
        set_shadow_residual,
    )

    sr_cfg = GraniteSwitchConfig(**src_cfg.to_dict())
    set_shadow_residual(sr_cfg, enabled=True)
    sr = ShadowResidualForCausalLM(sr_cfg)
    transfer_base_weights(src, sr)

    src_sd = src.state_dict()
    for layer_idx in range(src_cfg.num_hidden_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            src_w = src_sd[f"model.layers.{layer_idx}.self_attn.{proj}.weight"]
            sr_attn = sr.model.layers[layer_idx].self_attn
            torch.testing.assert_close(getattr(sr_attn, proj).weight, src_w)


def test_weight_transfer_mlp_slicing(tiny_base_model):
    """gate_proj / up_proj / down_proj rows match the fused source MLP."""
    src = tiny_base_model
    src_cfg = src.config

    from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig
    from shadow_residual.shadow_residual.config_helpers import (
        set_shadow_residual,
    )

    sr_cfg = GraniteSwitchConfig(**src_cfg.to_dict())
    set_shadow_residual(sr_cfg, enabled=True)
    sr = ShadowResidualForCausalLM(sr_cfg)
    transfer_base_weights(src, sr)

    src_sd = src.state_dict()
    for layer_idx in range(src_cfg.num_hidden_layers):
        gate_up_key = f"model.layers.{layer_idx}.shared_mlp.input_linear.weight"
        gate_up = src_sd[gate_up_key]
        inter = gate_up.shape[0] // 2
        gate_w = gate_up[:inter, :]
        up_w = gate_up[inter:, :]

        down_key = f"model.layers.{layer_idx}.shared_mlp.output_linear.weight"
        down_w = src_sd[down_key]

        sr_layer = sr.model.layers[layer_idx]
        torch.testing.assert_close(sr_layer.gate_proj.weight, gate_w)
        torch.testing.assert_close(sr_layer.up_proj.weight, up_w)
        torch.testing.assert_close(sr_layer.down_proj.weight, down_w)


def test_get_peft_model_non_rank0_stays_on_meta(tiny_base_model, monkeypatch):
    """Non-rank-0 ranks build SR + PEFT on meta — no real CPU alloc, no transfer.

    Under FSDP with ``fsdp_sync_module_states: true`` and
    ``fsdp_cpu_ram_efficient_loading: true``, only rank 0 should hold a
    real copy of the base + SR target in CPU RAM. Other ranks build a
    meta-device skeleton; FSDP fills the real values via post-shard
    broadcast. Without this gating, every DDP rank materializes the full
    base concurrently and OOMs the pod (verified on 30B: SIGKILL on all
    ranks during from_pretrained).

    This test sets LOCAL_RANK=1 and asserts:
      - The peft model's parameters live on the meta device.
      - ``AutoModelForCausalLM.from_pretrained`` is NOT called (no real
        upstream weights are loaded on non-rank-0).
      - ``transfer_base_weights`` is NOT called (skipped on non-rank-0).
    """
    from peft import LoraConfig
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.setenv("LOCAL_RANK", "1")
    # FSDP active → non-rank-0 stays on meta and relies on sync_module_states.
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")

    # Spy on AutoModelForCausalLM.from_pretrained — it must NOT be called
    # on non-rank-0 (only AutoConfig is consulted to get the config).
    auto_calls = []
    real_auto_from_pretrained = _factory_mod.AutoModelForCausalLM.from_pretrained

    class _SpyAuto:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            auto_calls.append((a, kw))
            return real_auto_from_pretrained(*a, **kw)

    monkeypatch.setattr(_factory_mod, "AutoModelForCausalLM", _SpyAuto)

    transfer_calls = []

    def _spy_transfer(src, dst, **kw):
        transfer_calls.append((src, dst, kw))

    monkeypatch.setattr(_factory_mod, "transfer_base_weights", _spy_transfer)

    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=None,
    )

    assert auto_calls == [], (
        f"AutoModelForCausalLM.from_pretrained called {len(auto_calls)} time(s) on non-rank-0; "
        "it must be skipped — only AutoConfig should be consulted."
    )
    assert transfer_calls == [], (
        f"transfer_base_weights called {len(transfer_calls)} time(s) on non-rank-0; "
        "it must be skipped to keep meta-rank CPU footprint at zero."
    )

    # Every parameter should be on meta.
    non_meta = [
        (name, p.device) for name, p in peft_model.named_parameters()
        if p.device.type != "meta"
    ]
    assert not non_meta, (
        f"non-rank-0 peft model has {len(non_meta)} non-meta params; "
        f"first few: {non_meta[:3]}. init_empty_weights wrapping likely "
        "didn't catch this construction site."
    )

    # lm_head MUST be tied to embed_tokens even on the non-rank-0 meta path.
    # The tie runs on EVERY rank (outside the rank-0 materialize gate) so the
    # module structure is identical across ranks before the FSDP wrap. If it
    # only ran on rank 0, rank 0 would have one shared storage while meta ranks
    # kept two separate params — FSDP flatten/sync_module_states would then
    # mismatch and an _ALLGATHER_BASE collective would hang until the NCCL
    # watchdog aborts (Signal 6, no checkpoint saved). Regression guard.
    sr_inner = peft_model.base_model.model
    lm_w = sr_inner.lm_head.weight
    emb_w = sr_inner.model.embed_tokens.weight
    assert lm_w is emb_w or lm_w.data_ptr() == emb_w.data_ptr(), (
        "lm_head.weight must be tied to embed_tokens.weight on the non-rank-0 "
        "meta path; tying only on rank 0 causes a cross-rank FSDP structure "
        "mismatch and an NCCL allgather hang."
    )


def test_get_peft_model_rank0_materializes_and_transfers(tiny_base_model, monkeypatch):
    """Rank 0 path: real CPU tensors, ``transfer_base_weights`` runs once with drain.

    Pins down the rank-0 contract under path D: meta build → to_empty(cpu)
    → from_pretrained → transfer with drain_src=True → reset_lora_parameters.
    """
    from peft import LoraConfig
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.setenv("LOCAL_RANK", "0")

    transfer_calls = []
    real_transfer = _factory_mod.transfer_base_weights

    def _spy_transfer(src, dst, **kw):
        transfer_calls.append(kw)
        real_transfer(src, dst, **kw)

    monkeypatch.setattr(_factory_mod, "transfer_base_weights", _spy_transfer)

    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=None,
    )

    assert len(transfer_calls) == 1, (
        f"rank 0 should call transfer_base_weights exactly once; got {len(transfer_calls)}."
    )
    assert transfer_calls[0].get("drain_src") is True, (
        "rank 0 must pass drain_src=True (host-RAM safety on 30B). "
        f"got kwargs={transfer_calls[0]}."
    )

    # Real, populated parameters.
    p = next(peft_model.parameters())
    assert p.device.type == "cpu", f"rank 0 should have CPU params; got {p.device}."


def test_materialize_reties_lm_head_to_embed_tokens(tiny_base_model, monkeypatch):
    """After meta-materialize, lm_head.weight must SHARE storage with embed_tokens.

    Regression guard for the untied-head bug: ``to_empty(device="cpu")`` in
    ``_materialize_and_transfer`` allocates fresh independent storage per
    parameter, which breaks the ``_tied_weights_keys`` tie between
    ``lm_head.weight`` and ``model.embed_tokens.weight``. transfer_base_weights
    repopulates embed_tokens but not the now-disjoint lm_head, so the
    loss-producing projection stays at to_empty garbage and never tracks the
    trained embedding (frozen / divergent loss; token-accuracy pinned at init).
    The factory re-ties via ``tie_weights()`` after the transfer. This pins that
    invariant on the rank-0 / non-FSDP (all-ranks-materialize) path.
    """
    from peft import LoraConfig
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.setenv("LOCAL_RANK", "0")

    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=None,
    )

    sr_inner = peft_model.base_model.model  # PeftModel → LoraModel → SR causal-LM
    lm_w = sr_inner.lm_head.weight
    emb_w = sr_inner.model.embed_tokens.weight

    assert lm_w.data_ptr() == emb_w.data_ptr(), (
        "lm_head.weight must share storage with embed_tokens.weight after "
        "materialize; to_empty() unties them and nothing re-tied → frozen "
        f"loss. lm_head ptr={lm_w.data_ptr()} embed ptr={emb_w.data_ptr()}."
    )
    assert torch.isfinite(lm_w).all(), "tied lm_head must be finite (no to_empty garbage)."


def test_reinit_rope_buffers_restores_nonfinite_inv_freq():
    """``_reinit_rope_buffers`` repopulates a garbage (non-finite) inv_freq.

    Regression for the 30B step-1 NaN: the rotary embedding registers
    ``inv_freq`` / ``original_inv_freq`` as persistent=False buffers, so the
    upstream state_dict lacks them and ``transfer_base_weights`` never copies
    them. After ``to_empty(cpu)`` they hold uninitialized garbage; on
    granite-4.1-30b this was verified non-finite (via SR_DIAG_MATERIALIZE),
    and FSDP ``sync_module_states`` then broadcast the garbage → NaN
    attention on the first forward (loss=nan even at learning_rate=0).

    Build a real rotary, clobber its inv_freq with NaNs to simulate the
    to_empty garbage, wrap it in a module so _reinit_rope_buffers can walk
    it, and assert the buffer is restored to a finite tensor matching a
    fresh rotary's frequencies.
    """
    import torch.nn as nn
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridRotaryEmbedding,
    )
    from shadow_residual.peft_shadow_residual.factory import (
        _reinit_rope_buffers,
    )

    cfg = _make_tiny_hybrid_config()
    rotary = GraniteMoeHybridRotaryEmbedding(config=cfg, device=torch.device("cpu"))
    good = rotary.inv_freq.clone()
    assert torch.isfinite(good).all()

    # Simulate to_empty garbage: clobber the buffer with NaN/Inf in-place.
    with torch.no_grad():
        rotary.inv_freq.fill_(float("nan"))
    assert not torch.isfinite(rotary.inv_freq).all()

    wrapper = nn.Module()
    wrapper.rotary_emb = rotary
    _reinit_rope_buffers(wrapper)

    assert torch.isfinite(rotary.inv_freq).all(), (
        "inv_freq still non-finite after _reinit_rope_buffers — the fix did "
        "not repopulate the buffer (this is the 30B NaN bug)."
    )
    torch.testing.assert_close(rotary.inv_freq, good)


def test_get_peft_model_unset_local_rank_treated_as_rank0(tiny_base_model, monkeypatch):
    """When LOCAL_RANK is unset (single-process / pytest / Notebook), behave as rank 0.

    Important so existing single-GPU users and the test suite don't
    accidentally hit the meta-device path.
    """
    from peft import LoraConfig
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=None,
    )

    p = next(peft_model.parameters())
    assert p.device.type == "cpu"


def test_get_peft_model_rank0_lora_init_is_kaiming_a_zero_b(tiny_base_model, monkeypatch):
    """After ``to_empty(cpu)``, LoRA params must be re-initialised — not garbage.

    ``model.to_empty(device='cpu')`` allocates fresh CPU storage with
    uninitialised values. Without an explicit re-init step, ``lora_A``
    would be whatever was in that allocator slot and ``lora_B`` would NOT
    be zeros — FSDP would then broadcast that garbage to other ranks and
    training would diverge or NaN immediately.

    This test pins down the contract that path D's
    ``_materialize_and_transfer`` re-runs PEFT's standard init on every
    LoraLayer: ``lora_A`` is non-zero (kaiming-uniform), ``lora_B`` is
    exactly zeros.
    """
    from peft import LoraConfig
    from peft.tuners.lora.layer import LoraLayer
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.setenv("LOCAL_RANK", "0")

    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=None,
    )

    n_lora_layers = 0
    for module in peft_model.modules():
        if not isinstance(module, LoraLayer):
            continue
        for adapter_name in module.lora_A.keys():
            lora_a_w = module.lora_A[adapter_name].weight
            lora_b_w = module.lora_B[adapter_name].weight
            assert not torch.all(lora_a_w == 0), (
                f"lora_A on {type(module).__name__} is all zeros — "
                "post-to_empty re-init likely didn't run."
            )
            torch.testing.assert_close(
                lora_b_w, torch.zeros_like(lora_b_w),
                msg=lambda m: (
                    f"lora_B is not zero ({m}); PEFT's stock init recipe "
                    "requires lora_B = 0 so that the initial delta = 0."
                ),
            )
            n_lora_layers += 1

    assert n_lora_layers > 0, "no LoraLayer was found in the peft model"


def test_get_peft_model_rank0_base_weights_match_upstream(tiny_base_model, monkeypatch):
    """End-to-end rank-0 path produces SR base weights that match the upstream.

    The path-D pipeline is: meta-build → to_empty(cpu) → from_pretrained →
    transfer_base_weights. This test verifies the round-trip end-to-end:
    SR's q_proj / k_proj / v_proj / o_proj weights match the upstream's
    after the factory returns.

    Catches any regression in `to_empty(cpu)` / `transfer_base_weights`
    interaction where the underlying storage was replaced but the copy
    landed in a stale tensor reference.
    """
    from peft import LoraConfig
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.setenv("LOCAL_RANK", "0")

    # Snapshot upstream weights BEFORE the factory mutates them via drain_src.
    upstream_sd = {
        k: v.detach().clone() for k, v in tiny_base_model.state_dict().items()
    }

    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=None,
    )

    # Reach the SR causal-LM through the peft wrappers.
    sr_inner = peft_model.base_model.model
    src_cfg = tiny_base_model.config
    for layer_idx in range(src_cfg.num_hidden_layers):
        sr_attn = sr_inner.model.layers[layer_idx].self_attn
        # q_proj is now a ShadowResidualLora wrapper; the base weight
        # lives on .base_layer.weight.
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sr_proj = getattr(sr_attn, proj)
            sr_w = (
                sr_proj.base_layer.weight
                if hasattr(sr_proj, "base_layer")
                else sr_proj.weight
            )
            up_w = upstream_sd[
                f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            ]
            torch.testing.assert_close(
                sr_w, up_w,
                msg=lambda m, p=proj, l=layer_idx: (
                    f"layer {l} {p}: SR base weight does not match "
                    f"upstream after path-D rank-0 materialize ({m})."
                ),
            )


def test_weight_transfer_drains_src_when_requested(tiny_base_model):
    """``drain_src=True`` unbinds source params after copy, freeing CPU RAM.

    Two invariants:
      1. Destination still receives correct weights (drain doesn't break copy).
      2. Source leaf params are unbound (set to None on the parent module),
         so Python can free the underlying storage.

    This is what lets 30B bases fit in the FSDP rank-0 RAM budget.
    """
    src = tiny_base_model
    src_cfg = src.config

    from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig
    from shadow_residual.shadow_residual.config_helpers import (
        set_shadow_residual,
    )

    sr_cfg = GraniteSwitchConfig(**src_cfg.to_dict())
    set_shadow_residual(sr_cfg, enabled=True)
    sr = ShadowResidualForCausalLM(sr_cfg)

    # Snapshot src tensors BEFORE the drain — transfer mutates dst in place
    # and unbinds src as it goes; we need the pre-drain values to verify.
    src_sd_pre = src.state_dict()
    q_pre = src_sd_pre["model.layers.0.self_attn.q_proj.weight"].detach().clone()
    gate_up_pre = src_sd_pre["model.layers.0.shared_mlp.input_linear.weight"].detach().clone()
    down_pre = src_sd_pre["model.layers.0.shared_mlp.output_linear.weight"].detach().clone()
    del src_sd_pre

    transfer_base_weights(src, sr, drain_src=True)

    # (1) Destination has the right values. Attention is name-matched (the
    # tiny fixture is already unfused); MLP is sliced from the fused source.
    sr_attn0 = sr.model.layers[0].self_attn
    torch.testing.assert_close(sr_attn0.q_proj.weight, q_pre)
    inter = gate_up_pre.shape[0] // 2
    torch.testing.assert_close(sr.model.layers[0].gate_proj.weight, gate_up_pre[:inter, :])
    torch.testing.assert_close(sr.model.layers[0].up_proj.weight, gate_up_pre[inter:, :])
    torch.testing.assert_close(sr.model.layers[0].down_proj.weight, down_pre)

    # (2) Source leaf params have been unbound. Reach into _parameters because
    # regular attribute access on a Parameter set to None can raise.
    src_attn0 = src.model.layers[0].self_attn
    assert src_attn0.q_proj._parameters.get("weight") is None
    assert src_attn0.k_proj._parameters.get("weight") is None
    assert src_attn0.v_proj._parameters.get("weight") is None
    assert src_attn0.o_proj._parameters.get("weight") is None
    src_mlp0 = src.model.layers[0].shared_mlp
    assert src_mlp0.input_linear._parameters.get("weight") is None
    assert src_mlp0.output_linear._parameters.get("weight") is None


# ── Upstream-unfused MLP path (GraniteForCausalLM, granite-4.1-3b family) ──


def _make_tiny_granite_config():
    """Tiny `granite` (NOT granitemoehybrid) config — unfused MLP, all-attention."""
    from transformers import GraniteConfig

    return GraniteConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
        rms_norm_eps=1e-5,
    )


@pytest.fixture
def tiny_granite_base_model():
    """Tiny `granite` model — exercises the upstream-unfused MLP transfer path."""
    from transformers import GraniteForCausalLM

    cfg = _make_tiny_granite_config()
    torch.manual_seed(0)
    model = GraniteForCausalLM(cfg)
    model.eval()
    return model


def test_build_sr_base_from_unfused_granite(tiny_granite_base_model, monkeypatch):
    """Regression: building SR from a `granite` (not granitemoehybrid) source.

    Two bugs previously caused SR to silently train with random-initialized
    MLP weights when the upstream was unfused (granite-4.1-3b family):

      1. transfer_base_weights had no handler for upstream-unfused
         mlp.gate_proj / mlp.up_proj / mlp.down_proj keys (the SR decoder
         layer carries them as direct attributes, no `.mlp.` parent).
      2. shared_intermediate_size defaulted to GraniteMoeHybridConfig's
         own default (e.g. 1024) instead of upstream's intermediate_size,
         so MLP shapes didn't even match upstream.

    This test reproduces both: every dst MLP weight must equal the src.
    """
    from shadow_residual.peft_shadow_residual import factory as _factory_mod

    _patch_auto(monkeypatch, _factory_mod, tiny_granite_base_model)

    # Snapshot src state BEFORE _build_sr_base — the factory now passes
    # drain_src=True to halve peak CPU RAM on 30B bases, which unbinds src
    # params as they are copied. Cloning here keeps the post-drain assertions
    # well-defined.
    src_cfg = tiny_granite_base_model.config
    src_sd = {k: v.detach().clone() for k, v in tiny_granite_base_model.state_dict().items()}

    sr = _build_sr_base("ignored-path")

    for layer_idx in range(src_cfg.num_hidden_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            src_w = src_sd[f"model.layers.{layer_idx}.mlp.{proj}.weight"]
            sr_w = getattr(sr.model.layers[layer_idx], proj).weight
            torch.testing.assert_close(
                sr_w, src_w,
                msg=lambda m, p=proj, l=layer_idx: (
                    f"layer {l} {p}: {m}\n"
                    "Most likely cause: transfer_base_weights doesn't recognise "
                    "upstream-unfused mlp.{gate,up,down}_proj keys, OR "
                    "shared_intermediate_size != upstream intermediate_size — "
                    "see scripts/diagnose_sr_weight_transfer.py."
                ),
            )

    # The SR config must have shared_intermediate_size == upstream
    # intermediate_size whenever the upstream's MLP is already unfused.
    assert sr.config.shared_intermediate_size == src_cfg.intermediate_size, (
        f"shared_intermediate_size={sr.config.shared_intermediate_size} "
        f"!= upstream intermediate_size={src_cfg.intermediate_size}. "
        "_build_sr_base should force these equal when constructing SR from an "
        "unfused-MLP upstream."
    )


# --- dtype invariant ---------------------------------------------------------
#
# FSDP refuses to flatten a parameter group with mixed dtypes:
#   "Must flatten tensors with uniform dtype but got
#    torch.bfloat16 and torch.float32"
# PEFT's get_peft_model invokes cast_adapter_dtype, which UPCASTS lora_A /
# lora_B to fp32 on top of an fp32-default base. The factory then casts
# the SR base to torch_dtype (e.g. bf16), which leaves lora at fp32 and
# triggers the FSDP error at accelerator.prepare time. The fix is a final
# peft_model.to(dtype=torch_dtype) inside _build_sr_peft_model_meta —
# applied on meta, so it's a metadata flip with no allocation.
#
# Two tests: meta path (every rank) and rank-0 materialized path.


def test_get_peft_model_meta_path_uniform_dtype(tiny_base_model, monkeypatch):
    """Non-rank-0 meta build: every parameter is torch_dtype, including LoRA.

    Regression test for the FSDP "must flatten tensors with uniform dtype"
    crash. PEFT's cast_adapter_dtype upcasts lora_A/lora_B to fp32 during
    get_peft_model; without an explicit re-cast the model has bf16 base
    and fp32 LoRA, which FSDP rejects when it tries to flatten a wrapped
    block containing both.
    """
    from peft import LoraConfig
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.setenv("LOCAL_RANK", "1")  # non-rank-0: stays on meta

    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=torch.bfloat16,
    )

    wrong_dtype = [
        (name, p.dtype) for name, p in peft_model.named_parameters()
        if p.dtype != torch.bfloat16
    ]
    assert not wrong_dtype, (
        f"non-rank-0 meta peft model has {len(wrong_dtype)} param(s) not in bf16; "
        f"first few: {wrong_dtype[:5]}. PEFT's cast_adapter_dtype likely "
        "left lora_A/lora_B at fp32 and the post-wrap .to(dtype) re-cast "
        "is missing — FSDP will fail with 'must flatten tensors with "
        "uniform dtype' when this hits accelerator.prepare."
    )


def test_get_peft_model_rank0_materialized_uniform_dtype(tiny_base_model, monkeypatch):
    """Rank-0 materialized build: every CPU param is torch_dtype, including LoRA.

    After to_empty(cpu) + transfer_base_weights + reset_lora_parameters,
    the model must still have uniform dtype across base and LoRA. This
    catches regressions where to_empty preserves dtype metadata but
    reset_lora_parameters reallocates the lora tensors at fp32 (the
    default for empty Linear).
    """
    from peft import LoraConfig
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.setenv("LOCAL_RANK", "0")

    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=torch.bfloat16,
    )

    wrong_dtype = [
        (name, p.dtype) for name, p in peft_model.named_parameters()
        if p.dtype != torch.bfloat16
    ]
    assert not wrong_dtype, (
        f"rank-0 materialized peft model has {len(wrong_dtype)} param(s) not in bf16; "
        f"first few: {wrong_dtype[:5]}. Likely cause: to_empty / "
        "reset_lora_parameters reallocated lora tensors at fp32. The "
        "post-wrap .to(dtype=torch_dtype) in _build_sr_peft_model_meta "
        "must hold across materialization."
    )


def test_get_peft_model_non_rank0_materializes_without_fsdp(tiny_base_model, monkeypatch):
    """Without FSDP (plain DDP / single-GPU), EVERY rank must materialize real
    weights — there is no fsdp_sync_module_states broadcast to fill meta params.

    Regression for: non-FSDP DDP runs failed at the Trainer's `.to(device)` with
    "NotImplementedError: Cannot copy out of meta tensor" because non-rank-0
    processes were left on meta (the meta-only gating assumed FSDP would
    broadcast). With ACCELERATE_USE_FSDP unset, rank 1 must load upstream
    weights and transfer, and end with no meta parameters.
    """
    import torch
    from peft import LoraConfig
    from shadow_residual.peft_shadow_residual import factory as _factory_mod
    from shadow_residual.peft_shadow_residual.factory import (
        get_shadow_residual_peft_model,
    )

    _patch_auto(monkeypatch, _factory_mod, tiny_base_model)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)  # FSDP NOT active

    auto_calls = []
    real_auto_from_pretrained = _factory_mod.AutoModelForCausalLM.from_pretrained

    class _SpyAuto:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            auto_calls.append((a, kw))
            return real_auto_from_pretrained(*a, **kw)

    monkeypatch.setattr(_factory_mod, "AutoModelForCausalLM", _SpyAuto)

    transfer_calls = []
    real_transfer = _factory_mod.transfer_base_weights

    def _spy_transfer(src, dst, **kw):
        transfer_calls.append(1)
        return real_transfer(src, dst, **kw)

    monkeypatch.setattr(_factory_mod, "transfer_base_weights", _spy_transfer)

    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    peft_model = get_shadow_residual_peft_model(
        "ignored-path", lora_config, torch_dtype=None,
    )

    assert len(auto_calls) >= 1, (
        "without FSDP, non-rank-0 MUST load upstream weights (from_pretrained), "
        "otherwise it is left on meta and .to(device) fails"
    )
    assert len(transfer_calls) >= 1, "without FSDP, non-rank-0 must transfer base weights"
    meta_params = [n for n, p in peft_model.named_parameters() if p.device.type == "meta"]
    assert not meta_params, f"non-rank-0 still has meta params without FSDP: {meta_params[:3]}"
