# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual model construction — config assembly + weight materialization.

This module is pure model/HF code with **no PEFT dependency**. It builds a
:class:`ShadowResidualConfig` from an upstream Granite config, constructs a real
:class:`ShadowResidualForCausalLM`, transfers upstream base weights into SR's
unfused projections, and repairs the tensors that a meta-init ``to_empty()``
leaves as garbage (non-persistent RoPE ``inv_freq`` buffers).

It is imported by :mod:`shadow_residual.training.factory` (the
PEFT-wrapping training factory) and can be used standalone to build a bare SR
model for plain inference — no adapters involved.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM

from .model_config import ShadowResidualConfig
from . import ShadowResidualForCausalLM
from .config_helpers import set_shadow_residual
from .cross_stream import (
    DEFAULT_CROSS_STREAM_TAPS,
    is_cross_stream_tap,
    normalize_cross_stream_tap_types,
)
from .weight_transfer import transfer_base_weights

logger = logging.getLogger(__name__)


def build_sr_config(
    base_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    share_moe_routing: bool = True,
    cross_stream_taps: Optional[Sequence[str]] = None,
    cross_stream_tap_types: Optional[Mapping[str, Any]] = None,
    cross_stream_type: Optional[str] = None,
    cross_stream_dim: Optional[int] = None,
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
    """
    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(base_model_name_or_path)
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
    # MoE bases (Granite 5.0 / granitemoe, num_local_experts > 0) have NO dense
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
    # Which cross-stream sites the decoder builds. Set BEFORE
    # set_shadow_residual so its validation covers the names. None → the
    # historical single post-MLP tap. Normally derived from the PEFT
    # target_modules by training.factory (the tap set IS the set of
    # cross_stream* LoRA targets), so train/serve cannot drift.
    sr_config.cross_stream_taps = (
        list(cross_stream_taps)
        if cross_stream_taps is not None
        else list(DEFAULT_CROSS_STREAM_TAPS)
    )
    # Cross-stream layer TYPE per tap — the second, ORTHOGONAL axis (the tap NAME is
    # the wiring). STRUCTURAL: it decides which module the decoder builds at each
    # site (CrossStream for "lora", a trainable full-H×H CrossStreamLinear for
    # "linear", a full-rank MonarchCrossStream for "monarch"), so the base tree must
    # be built the way it was trained BEFORE PEFT attaches — otherwise
    # from_pretrained can't bind the saved cross-stream weights. Unlike
    # share_moe_routing, a non-lora tap's weights ARE saved in the adapter (via PEFT
    # modules_to_save), and the serving path recovers type + numeric parameter from
    # those saved tensor SHAPES (see
    # training.generation_utils.read_cross_stream_tap_types_from_adapter).
    #
    # Normalized here (and set BEFORE set_shadow_residual so validation covers it)
    # from either the per-tap map or the legacy scalar cross_stream_type /
    # cross_stream_dim pair. Set on every rank so the FSDP meta-init tree is
    # identical.
    sr_config.cross_stream_tap_types = normalize_cross_stream_tap_types(
        sr_config.cross_stream_taps,
        cross_stream_tap_types if cross_stream_tap_types is not None else cross_stream_type,
        cross_stream_dim,
    )
    set_shadow_residual(sr_config, enabled=True)
    # MoE-only routing mode (no-op on dense bases). Not serialized — it is a
    # forward-path choice, not a weight, so a saved adapter carries no record of
    # it; train and serve must pass the same value (see build_sr_base).
    sr_config.share_moe_routing = bool(share_moe_routing)
    if torch_dtype is not None:
        sr_config.torch_dtype = torch_dtype
    if attn_implementation is not None:
        sr_config._attn_implementation = attn_implementation
    return sr_config


def build_sr_base(
    base_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    share_moe_routing: bool = True,
    cross_stream_taps: Optional[Sequence[str]] = None,
    cross_stream_tap_types: Optional[Mapping[str, Any]] = None,
    cross_stream_type: Optional[str] = None,
    cross_stream_dim: Optional[int] = None,
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

    ``cross_stream_taps`` selects which cross-stream sites the decoder builds
    (names from :data:`cross_stream.CROSS_STREAM_TAPS`); ``None`` → the historical
    single post-MLP ``cross_stream`` tap. ``cross_stream_tap_types`` selects WHAT
    KIND of module sits at each site — the orthogonal TYPE axis, a
    ``{tap -> "lora" | "linear" | "monarch"}`` or ``{tap -> {"type", "num"}}`` map
    (``"lora"``: frozen H×H, LoRA-wrapped; ``"linear"``: directly-trainable full
    ``H×H``, ``num`` is provenance only; ``"monarch"``: full-rank two-factor
    butterfly, ``num`` is the block count ``b``). The legacy scalar
    ``cross_stream_type`` / ``cross_stream_dim`` pair is still accepted and applies
    to every built tap. All of these MUST match what the adapter was trained with,
    or ``PeftModel.from_pretrained`` finds no matching module to bind the saved
    tensors to. Source them from the adapter itself via
    :func:`shadow_residual.training.generation_utils.read_cross_stream_taps_from_adapter`
    and
    :func:`shadow_residual.training.generation_utils.read_cross_stream_tap_types_from_adapter`
    — the saved ``adapter_config.json`` records the taps (in ``target_modules`` /
    ``modules_to_save``) and the saved tensor SHAPES identify each non-lora tap's
    type and numeric parameter.
    """
    sr_config = build_sr_config(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        share_moe_routing=share_moe_routing,
        cross_stream_taps=cross_stream_taps,
        cross_stream_tap_types=cross_stream_tap_types,
        cross_stream_type=cross_stream_type,
        cross_stream_dim=cross_stream_dim,
    )
    sr_model = ShadowResidualForCausalLM(sr_config)
    if torch_dtype is not None:
        sr_model = sr_model.to(dtype=torch_dtype)

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


def reinit_rope_buffers(model) -> None:
    """Recompute non-persistent RoPE ``inv_freq`` buffers after meta-materialize.

    Walks every rotary module (anything carrying a real ``inv_freq`` buffer)
    and recomputes its frequencies by re-instantiating the rotary class
    fresh on CPU — its ``__init__`` runs the correct rope-init for whatever
    rope_type the config declares, so we don't reimplement the HF dispatch
    (which varies across transformers versions). The freshly-computed
    ``inv_freq`` / ``original_inv_freq`` are copied in-place into the
    materialized buffers. No-op when the model has no rotary module (e.g.
    configs that disable RoPE). Idempotent and safe on a fully-real model.
    See :func:`shadow_residual.training.factory._materialize_and_transfer`
    for why this is required under the FSDP meta-init path.
    """
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
            module.inv_freq.copy_(fresh.inv_freq.to(module.inv_freq.dtype))
            if hasattr(module, "original_inv_freq") and hasattr(fresh, "original_inv_freq"):
                module.original_inv_freq.copy_(
                    fresh.original_inv_freq.to(module.original_inv_freq.dtype)
                )
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
        "[SR-DIAG] all-zero tensors: %d (expected: lora_B.*, biases, and every "
        "cross-stream tap's zero-init output factor; suspicious: any other "
        "weight/norm/embedding).", len(allzero),
    )
    # Every cross-stream tap type starts as an exact ZERO injection, so a tensor
    # under a tap is legitimately all-zero post-materialize: the frozen H×H weight
    # for "lora", `proj.weight` for "linear", `f_out` for "monarch". `f_in` is the
    # one exception (per-block Kaiming), so it stays in the suspicious set.
    for name, why in allzero[:60]:
        parts = name.split(".")
        under_tap = any(is_cross_stream_tap(p) for p in parts)
        suspicious = not (
            "lora_B" in name
            or name.endswith(".bias")
            or (under_tap and not name.endswith(".f_in"))
        )
        logger.info("[SR-DIAG]   %s %-66s %s",
                    "SUSPECT-ZERO" if suspicious else "ok-zero", name, why)


__all__ = [
    "build_sr_config",
    "build_sr_base",
    "reinit_rope_buffers",
    "diagnose_materialized",
]
