# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual model construction — config assembly + weight materialization.

This module is pure model/HF code with **no PEFT dependency**. It builds a
:class:`ShadowResidualConfig` from an upstream Granite config, constructs a real
:class:`ShadowResidualForCausalLM`, transfers upstream base weights into SR's
unfused projections, and repairs the tensors that a meta-init ``to_empty()``
leaves as garbage (non-persistent RoPE ``inv_freq`` buffers).

Two kinds of source are accepted wherever a base id/path is taken:

- an **SR checkpoint** (``model_type: "shadow_residual"``, produced once by
  :mod:`shadow_residual.shadow_residual.convert`) — loaded directly with
  ``from_pretrained``; its config and weights are already in SR layout;
- an **upstream Granite checkpoint** — converted in memory on every load (the
  original path).

It is imported by :mod:`shadow_residual.training.factory` (the
PEFT-wrapping training factory) and can be used standalone to build a bare SR
model for plain inference — no adapters involved.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from .model_config import ShadowResidualConfig
from . import ShadowResidualForCausalLM
from .config_helpers import set_shadow_residual
from .weight_transfer import transfer_base_weights

logger = logging.getLogger(__name__)


def is_sr_checkpoint(model_name_or_path: str) -> bool:
    """True iff ``model_name_or_path`` is an already-converted SR checkpoint.

    Reads only the config (cheap, cached). An SR checkpoint's ``config.json``
    carries ``model_type: "shadow_residual"``, which resolves to
    :class:`ShadowResidualConfig` through the auto-class registration in this
    package's ``__init__``; an upstream Granite checkpoint resolves to its own
    config class.
    """
    from transformers import AutoConfig

    return isinstance(AutoConfig.from_pretrained(model_name_or_path), ShadowResidualConfig)


def build_sr_config(
    base_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    share_moe_routing: bool = True,
) -> ShadowResidualConfig:
    """Build the SR config by reading the upstream HF config (no weights).

    Pulled out so every rank can call it cheaply (config-only download,
    no checkpoint shards) — the SR module tree is then constructed
    identically on every rank from this config. Keeping construction
    config-driven (vs. via from_pretrained) is what makes meta-init on
    non-rank-0 produce a structurally identical model to rank 0's real
    init: from_pretrained has its own meta→load-in-place machinery that
    historically diverged the parameter insertion order, breaking FSDP's
    sync_module_states broadcast.

    For an SR checkpoint the stored config is already final; only the runtime
    choices (``share_moe_routing``, dtype, attention backend) are applied.
    """
    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(base_model_name_or_path)
    if isinstance(base_config, ShadowResidualConfig):
        return _apply_runtime_options(
            set_shadow_residual(base_config, enabled=True),
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            share_moe_routing=share_moe_routing,
        )

    config_dict = base_config.to_dict()
    # Normalize layer_types to all-attention. Some Granite configs ship a
    # `layer_types` list that includes non-"attention" entries (e.g. mamba
    # positions on hybrid configs); the SR architecture is attention-only,
    # so any base whose actual checkpoint is attention-only — including
    # plain `granite` models that have no mamba weights at all — should
    # train fine after this normalization. Bases whose checkpoint really
    # does carry mamba weights would surface as missing q/k/v/o tensors
    # in `transfer_base_weights` and fail there with a clear error.
    if config_dict.get("layer_types"):
        config_dict["layer_types"] = ["attention" for _ in config_dict["layer_types"]]
    # Force shared_intermediate_size to the upstream's intermediate_size
    # whenever the upstream's MLP is already unfused. Without this, the
    # GraniteMoeHybridConfig parent of ShadowResidualConfig defaults
    # shared_intermediate_size to its own value (1024 for some bases),
    # leaving SR's MLP a fraction the size of the upstream's. Verified
    # against ibm-granite/granite-4.1-3b: upstream intermediate_size=8192,
    # SR was building MLP with shared_intermediate_size=1024 → MLP
    # weights couldn't be transferred (shape mismatch), MLP stayed
    # randomly initialized, hidden states blew up 78x at layer 0 (see
    # scripts/diagnose_sr_vs_upstream.py output).
    #
    # Sparse-MoE bases (num_local_experts > 0) have NO dense
    # shared_mlp — their `intermediate_size` is the *per-expert* FFN width. The
    # SR MLP on such a base is the frozen expert bank (built from num_local_experts
    # + intermediate_size directly, see decoder_hf.ShadowResidualMLP), which never
    # reads shared_intermediate_size. Rewriting it to the per-expert width would be
    # harmless but misleading, so skip the force in MoE mode.
    is_moe = int(config_dict.get("num_local_experts") or 0) > 0
    if (
        not is_moe
        and "intermediate_size" in config_dict
        and config_dict.get("shared_intermediate_size") is None
    ):
        config_dict["shared_intermediate_size"] = config_dict["intermediate_size"]
    sr_config = ShadowResidualConfig(**config_dict)
    set_shadow_residual(sr_config, enabled=True)
    return _apply_runtime_options(
        sr_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        share_moe_routing=share_moe_routing,
    )


def _apply_runtime_options(
    sr_config: ShadowResidualConfig,
    *,
    torch_dtype: Optional[torch.dtype],
    attn_implementation: Optional[str],
    share_moe_routing: bool,
) -> ShadowResidualConfig:
    # MoE-only routing mode (no-op on dense bases). It is a forward-path choice,
    # not a weight: a saved adapter's weights carry no record of it, so train and
    # serve must pass the same value (see build_sr_base). An SR checkpoint's
    # config.json does store it, but always as the caller's value here.
    sr_config.share_moe_routing = bool(share_moe_routing)
    if torch_dtype is not None:
        sr_config.torch_dtype = torch_dtype
    if attn_implementation is not None:
        sr_config._attn_implementation = attn_implementation
    return sr_config


def load_sr_checkpoint(
    model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    share_moe_routing: bool = True,
) -> ShadowResidualForCausalLM:
    """Load a converted SR checkpoint with stock ``from_pretrained``.

    Strict: any missing, unexpected or shape-mismatched weight raises instead of
    being silently re-initialized. The frozen all-zero cross-stream weights are
    deliberately absent from the checkpoint and re-zeroed on load, so they are
    not reported as missing.
    """
    sr_config = build_sr_config(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        share_moe_routing=share_moe_routing,
    )
    load_kwargs = {"config": sr_config, "output_loading_info": True}
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    sr_model, info = ShadowResidualForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    problems = {
        kind: sorted(info.get(kind) or [])
        for kind in ("missing_keys", "unexpected_keys", "mismatched_keys")
    }
    if any(problems.values()) or info.get("error_msgs"):
        raise ValueError(
            f"SR checkpoint {model_name_or_path!r} does not match the SR model "
            f"built from its config: {problems} {info.get('error_msgs') or ''}. "
            "Re-convert it with the current shadow_residual version (sr-convert)."
        )
    if torch_dtype is not None:
        # Same cast as the in-memory conversion path (build_sr_base below and the
        # training factory), so a base built either way is tensor-for-tensor
        # identical and an adapter sees the same numerics at train and serve time.
        sr_model = to_dtype_keep_rope_fp32(sr_model, torch_dtype)
    return sr_model


def build_sr_base(
    base_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    share_moe_routing: bool = True,
) -> ShadowResidualForCausalLM:
    """Single-process SR base build (no FSDP / meta machinery).

    Used by the serving/eval path — build a base here, then attach a saved
    adapter with stock :func:`peft.PeftModel.from_pretrained`. Every
    rank-equivalent process needs a fully real SR model and the FSDP
    cpu-ram-efficient pattern doesn't apply.
    The training-time FSDP path goes through
    :func:`get_shadow_residual_peft_model` instead, which keeps non-rank-0
    processes on meta until the broadcast.

    ``share_moe_routing`` (MoE bases only) routes once on the base stream and
    reuses that expert partition for the adapter stream; on by default. Set
    ``False`` to route each stream independently. It is a forward-path choice,
    not a weight — the served value must match what the adapter was **trained**
    with or the forward diverges from training. When serving a saved adapter,
    prefer sourcing it from the adapter itself via
    :func:`shadow_residual.training.generation_utils.read_share_moe_routing_from_adapter`
    (train.py records it in ``adapter_config.json``) rather than hand-passing.

    ``base_model_name_or_path`` may be an SR checkpoint (loaded directly via
    :func:`load_sr_checkpoint`) or an upstream Granite checkpoint (converted in
    memory below).
    """
    if is_sr_checkpoint(base_model_name_or_path):
        return load_sr_checkpoint(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            share_moe_routing=share_moe_routing,
        )

    sr_config = build_sr_config(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        share_moe_routing=share_moe_routing,
    )
    sr_model = ShadowResidualForCausalLM(sr_config)
    if torch_dtype is not None:
        sr_model = to_dtype_keep_rope_fp32(sr_model, torch_dtype)

    load_kwargs = {"low_cpu_mem_usage": True}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, **load_kwargs,
    )
    transfer_base_weights(base_model, sr_model, drain_src=True)
    del base_model
    return sr_model


_ROPE_BUFFERS = ("inv_freq", "original_inv_freq")


def _rope_buffers_to_fp32(model) -> None:
    """Re-register every RoPE frequency buffer as float32 (values not recomputed).

    On meta tensors this is a pure metadata flip. On real tensors the values
    keep whatever rounding they already had — follow with
    :func:`reinit_rope_buffers` to recompute them exactly.
    """
    for module in model.modules():
        for name in _ROPE_BUFFERS:
            buf = module._buffers.get(name)
            if buf is not None and buf.dtype != torch.float32:
                module._buffers[name] = buf.float()


def to_dtype_keep_rope_fp32(model, dtype: torch.dtype):
    """``model.to(dtype=dtype)``, except the RoPE frequency buffers stay float32.

    A whole-module ``.to(bfloat16)`` also casts the non-persistent
    ``inv_freq`` / ``original_inv_freq`` buffers, and bf16 frequencies drift the
    rotation angle by radians at long positions (measured on granite-4.1-3b:
    up to ~1.2 rad at position 1024, ~9.6 rad at 8192). Stock HF
    ``from_pretrained(dtype=bfloat16)`` keeps them fp32; every SR build path
    casts through here so training and serving match it and each other.
    Real buffers are recomputed exactly; meta buffers only change dtype.
    """
    model = model.to(dtype=dtype)
    _rope_buffers_to_fp32(model)
    reinit_rope_buffers(model)
    return model


def reinit_rope_buffers(model) -> None:
    """Recompute non-persistent RoPE ``inv_freq`` buffers after meta-materialize.

    Walks every rotary module (anything carrying a real ``inv_freq`` buffer)
    and recomputes its frequencies by re-instantiating the rotary class
    fresh on CPU — its ``__init__`` runs the correct rope-init for whatever
    rope_type the config declares, so we don't reimplement the HF dispatch
    (which varies across transformers versions). The freshly-computed
    ``inv_freq`` / ``original_inv_freq`` are copied into the materialized
    buffers, which are always left in float32 (see
    :func:`to_dtype_keep_rope_fp32`). No-op when the model has no rotary module
    (e.g. configs that disable RoPE). Idempotent and safe on a fully-real model.
    See :func:`shadow_residual.training.factory._materialize_and_transfer`
    for why this is required under the FSDP meta-init path.
    """
    _rope_buffers_to_fp32(model)
    n_fixed = 0
    for module in model.modules():
        buf = getattr(module, "inv_freq", None)
        if buf is None or getattr(buf, "is_meta", False):
            continue
        config = getattr(module, "config", None)
        if config is None:
            continue
        # Re-instantiate the rotary class fresh (real init, CPU). Runs the
        # model's own rope-init recipe regardless of rope_type.
        fresh = type(module)(config=config, device=torch.device("cpu"))
        with torch.no_grad():
            module.inv_freq.copy_(fresh.inv_freq)
            if hasattr(module, "original_inv_freq") and hasattr(fresh, "original_inv_freq"):
                module.original_inv_freq.copy_(fresh.original_inv_freq)
        if hasattr(fresh, "attention_scaling"):
            module.attention_scaling = fresh.attention_scaling
        n_fixed += 1

    logger.info("Re-initialized RoPE inv_freq buffers on %d rotary module(s).", n_fixed)


def diagnose_materialized(model) -> None:
    """Rank-0 finite/zero audit of every parameter and buffer post-materialize."""
    nonfinite, allzero, n_total = [], [], 0
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t is None or getattr(t, "is_meta", False):
            nonfinite.append((name, "META"))
            continue
        n_total += 1
        try:
            if not torch.isfinite(t).all().item():
                nonfinite.append((name, f"nonfinite shape={tuple(t.shape)} dtype={t.dtype}"))
            elif t.numel() > 0 and bool((t == 0).all().item()):
                allzero.append((name, f"all-zero shape={tuple(t.shape)}"))
        except Exception as e:  # pragma: no cover - diagnostic only
            nonfinite.append((name, f"check-failed: {e}"))

    logger.info("[SR-DIAG] audited %d real tensors.", n_total)
    logger.info("[SR-DIAG] non-finite / meta tensors: %d", len(nonfinite))
    for name, why in nonfinite[:60]:
        logger.info("[SR-DIAG]   NONFINITE %-70s %s", name, why)
    logger.info(
        "[SR-DIAG] all-zero tensors: %d (expected: lora_B.*, biases; "
        "suspicious: any weight/norm/embedding).", len(allzero),
    )
    for name, why in allzero[:60]:
        suspicious = not ("lora_B" in name or name.endswith(".bias"))
        logger.info("[SR-DIAG]   %s %-66s %s",
                    "SUSPECT-ZERO" if suspicious else "ok-zero", name, why)


__all__ = [
    "is_sr_checkpoint",
    "build_sr_config",
    "build_sr_base",
    "load_sr_checkpoint",
    "reinit_rope_buffers",
    "to_dtype_keep_rope_fp32",
    "diagnose_materialized",
]
