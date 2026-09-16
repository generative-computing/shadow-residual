# SPDX-License-Identifier: Apache-2.0
"""Factory: ``get_shadow_residual_peft_model(base_id, lora_config)``.

Builds a :class:`ShadowResidualForCausalLM` and layers PEFT on top.
Under FSDP, every rank constructs the SR + PEFT module tree on meta
with identical traversal order; only rank 0 materializes real CPU
tensors and loads upstream base weights. ``fsdp_sync_module_states``
broadcasts rank 0's values to other ranks after sharding — peak host
RAM is one rank's copy instead of N.

Whether the dual-stream forward actually runs is a runtime decision made
by the SR model itself (see ``modeling_hf.py``): if no cross-stream
site has been wrapped by a stock LoRA layer (i.e. the user listed no tap
name in ``target_modules``), every forward takes a
single-stream early-exit path with compute equivalent to plain Granite +
the wrapped LoRA deltas. This is why there is no factory-level branch: one
architecture, two forward paths.

Which cross-stream sites exist is derived from the LoRA config's two selections
(``target_modules`` ∪ ``modules_to_save``) — see
``cross_stream.cross_stream_taps_from_target_modules``. WHICH selection a tap
lands in is its layer TYPE: ``target_modules`` → a frozen-zero ``CrossStream``
that stock ``lora.Linear`` wraps; ``modules_to_save`` → a directly-trainable
``CrossStreamLinear`` / ``MonarchCrossStream`` that ``ModulesToSaveWrapper``
persists whole (``cross_stream_tap_types`` selects which).

There is no custom PEFT code either way: no custom LoRA classes and no
``_register_custom_module``. The adapter is plain LoRA and always active — the
delta fires on every position, in both streams' Q/O/MLP and the cross-stream
injections. There is no gated/aLoRA activation.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.shadow_residual import (
    ShadowResidualForCausalLM,
)
from shadow_residual.shadow_residual.cross_stream import (
    CROSS_STREAM_TAPS,
    CrossStreamTap,
    cross_stream_taps_from_target_modules,
    normalize_cross_stream_tap_types,
)
from shadow_residual.shadow_residual.weight_transfer import (
    transfer_base_weights,
)
# Model-construction helpers live in the model package (no PEFT dependency);
# imported here for the meta-init / materialize orchestration below. Aliased to
# the historical ``_``-prefixed names so existing importers keep working.
from shadow_residual.shadow_residual.build import (
    build_sr_config as _build_sr_config,
    build_sr_base as _build_sr_base,
    reinit_rope_buffers as _reinit_rope_buffers,
    diagnose_materialized as _diagnose_materialized,
)

logger = logging.getLogger(__name__)


def _materialize_and_transfer(
    peft_model,
    base_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype],
    attn_implementation: Optional[str],
) -> None:
    """Rank-0 only: turn meta tensors into real CPU tensors, then load real
    base weights and re-init LoRA.

    The peft_model arrives with every parameter on meta. ``to_empty(cpu)``
    rewrites those into real (uninitialized) CPU storage with the same
    shapes/dtypes. We then:

    1. Load the real upstream model and ``transfer_base_weights`` slices
       its fused projections into the SR base_layer tensors. This
       repopulates every base weight in-place (the ``base_layer`` keypath
       indirection is already handled by ``transfer_base_weights``).
    2. Re-run PEFT's LoRA init recipe on every wrapped projection so
       ``lora_A`` is kaiming-uniform and ``lora_B`` is zeros. Without
       this step, lora_A would be uninitialized garbage from
       ``to_empty(cpu)``; FSDP would broadcast that garbage to other
       ranks and training would diverge or NaN immediately.

    Rationale for this two-phase shape: every rank constructs the SR +
    PEFT module tree on meta with identical traversal order (no
    ``from_pretrained`` divergence). Only rank 0 pays the host-RAM cost
    of the real upstream + materialized SR copy; other ranks stay on
    meta until FSDP's ``sync_module_states`` broadcasts rank 0's values
    onto their (now sharded across GPUs) tensors.
    """
    # Materialize meta → real CPU storage in-place. This preserves the
    # nn.Parameter / nn.Module identity that PEFT's wrappers reference
    # (no lookup tables to update) — only the underlying tensor
    # storage changes. Equivalent to a per-parameter
    # `param.data = torch.empty_like(param, device='cpu')` walk, but
    # safer because it goes through torch's official path.
    peft_model.to_empty(device="cpu")

    # Load the real upstream model. low_cpu_mem_usage avoids the 2x host-
    # RAM transient on the upstream side (init-real → meta-init → load-
    # in-place). Without this, large bases (e.g. granite-4.1-30b, 57.7GB)
    # OOM host RAM even on rank 0 (transient 2x peak crosses 400Gi).
    load_kwargs = {"low_cpu_mem_usage": True}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, **load_kwargs,
    )

    # The peft_model wraps SR base; transfer_base_weights walks the SR
    # state_dict, which already includes the .base_layer. indirection
    # peft adds — and the function already strips that to find the
    # underlying SR keys (see the ``.replace(".base_layer.weight", ...)``
    # branch in weight_transfer.py). drain_src=True frees each upstream
    # tensor immediately after it's been copied/sliced.
    sr_inner = peft_model.base_model.model  # PeftModel → LoraModel → SR causal-LM
    transfer_base_weights(base_model, sr_inner, drain_src=True)
    del base_model

    # Re-initialize LoRA params (overwriting the garbage values left by
    # to_empty). PEFT exposes per-layer reset on every LoraLayer; walk
    # the model and call it. ``init_lora_weights`` defaults are honored
    # via the lora_config the layer was built with.
    from peft.tuners.lora.layer import LoraLayer

    for module in peft_model.modules():
        if isinstance(module, LoraLayer):
            for adapter_name in module.lora_A.keys():
                module.reset_lora_parameters(
                    adapter_name, init_lora_weights=True,
                )

    # Re-initialize every cross-stream tap, of every type, at every configured site.
    # LOAD-BEARING: ``to_empty(cpu)`` above rewrote each tap's tensors to
    # uninitialized GARBAGE (observed abs_sum ~1e28 / inf / nan per layer), and
    # nothing else repopulates them — they are not upstream weights, so
    # transfer_base_weights skips them, and the LoRA reset only touches LoraLayer.
    # A garbage tap injects inf/nan into the adapter stream → NaN logits on the
    # adapter path (the base stream stays finite), and under FSDP
    # sync_module_states broadcasts that garbage to every rank. The model's
    # _init_weights does this on plain construction, but it does not run after
    # to_empty on this meta-materialize path. Same bug class as the RoPE-buffer
    # reinit below.
    #
    # ONE walk covers all types (the CrossStreamTap marker mixin), and it is correct
    # for both halves of a ModulesToSaveWrapper — the frozen ``original_module`` and
    # the trainable copy — precisely because ``reset_parameters()`` is numerics-only
    # and never touches requires_grad (which to_empty preserves).
    n_taps: dict[str, int] = {}
    for module in peft_model.modules():
        if isinstance(module, CrossStreamTap):
            module.reset_parameters()
            kind = type(module).__name__
            n_taps[kind] = n_taps.get(kind, 0) + 1
    logger.info(
        "Re-initialized %d cross-stream tap module(s) after materialize: %s",
        sum(n_taps.values()),
        ", ".join(f"{k}={v}" for k, v in sorted(n_taps.items())) or "none",
    )

    # Re-initialize non-persistent RoPE buffers. The rotary embedding
    # registers ``inv_freq`` (and ``original_inv_freq``) with
    # persistent=False, so they are NOT in the upstream state_dict —
    # transfer_base_weights never copies them, and the LoRA reset above
    # doesn't touch them. After ``to_empty`` they hold uninitialized garbage
    # (verified non-finite on granite-4.1-30b via SR_DIAG_MATERIALIZE).
    # Under FSDP, ``sync_module_states`` then broadcasts that garbage to
    # every rank, producing garbage RoPE frequencies → NaN attention on the
    # very first forward (loss=nan even at learning_rate=0). Recompute them
    # so the buffers are correct before FSDP broadcasts.
    _reinit_rope_buffers(peft_model)

    # Opt-in diagnostic (no-op unless SR_DIAG_MATERIALIZE=1): after the meta
    # → real materialization + base-weight transfer + LoRA reset, every
    # tensor on rank 0 should be finite and "real" (transferred or freshly
    # initialized). Anything still non-finite or all-zeros is a tensor that
    # to_empty() left as garbage and nothing repopulated — and under FSDP
    # sync_module_states it gets broadcast to every rank, the classic NaN
    # source. This dumps the suspects so we don't have to guess.
    if os.environ.get("SR_DIAG_MATERIALIZE") == "1":
        _diagnose_materialized(peft_model)


def _build_sr_peft_model_meta(
    sr_config: ShadowResidualConfig,
    lora_config: LoraConfig,
    *,
    torch_dtype: Optional[torch.dtype],
    adapter_name: str,
):
    """Build SR + PEFT-wrap entirely under meta context.

    Runs identically on every rank. Returned model has meta-device
    tensors for every parameter; storage is allocated only on rank 0
    by the subsequent ``_materialize_and_transfer`` call.
    """
    from accelerate import init_empty_weights

    with init_empty_weights():
        sr_model = ShadowResidualForCausalLM(sr_config)
        if torch_dtype is not None:
            # ``.to(dtype)`` rewrites dtype metadata on meta tensors —
            # no storage alloc.
            sr_model = sr_model.to(dtype=torch_dtype)
        peft_model = get_peft_model(sr_model, lora_config, adapter_name=adapter_name)
        if torch_dtype is not None:
            # PEFT's get_peft_model calls cast_adapter_dtype, which
            # *upcasts* lora_A/lora_B to fp32 (the standard recipe — fp32
            # adapter updates on top of a quantized/bf16 base). That
            # leaves us with mixed-dtype params (bf16 base, fp32 lora)
            # which FSDP refuses to flatten:
            #   "Must flatten tensors with uniform dtype but got
            #    torch.bfloat16 and torch.float32"
            # Re-cast the entire PEFT model to torch_dtype so every
            # tensor (base + lora) is uniform before FSDP wraps. Still
            # all on meta — pure metadata flip, no allocation.
            peft_model = peft_model.to(dtype=torch_dtype)
    return peft_model


def _disable_merge_and_unload(peft_model) -> None:
    """Disable ``merge_and_unload`` on the returned PEFT model.

    Merging the LoRA delta into the base linear would silently fold the
    adapter into the frozen-base path, breaking the shadow-residual
    invariant.  ``save_pretrained`` / ``from_pretrained`` go through a
    separate code path (state-dict round-trip) and are unaffected.
    """
    def _raise(*_args, **_kwargs):
        raise NotImplementedError(
            "merge_and_unload is disabled for shadow-residual PEFT models. "
            "Merging the LoRA delta into the base linear would break the "
            "frozen-base invariant of the shadow-residual architecture. "
            "Use save_pretrained + PeftModel.from_pretrained (on a build_sr_base "
            "base) for checkpoint round-trips instead."
        )

    peft_model.merge_and_unload = _raise
    if hasattr(peft_model, "base_model"):
        peft_model.base_model.merge_and_unload = _raise
        peft_model.base_model.merge_adapter = _raise
        peft_model.base_model.unmerge_adapter = _raise


def _reject_kv_lora(lora_config: LoraConfig) -> None:
    """Forbid LoRA on ``k_proj`` / ``v_proj``.

    SR uses shared base-only K/V: the adapter stream never multiplies
    ``normed_adapt`` by ``W_K`` or ``W_V`` — there is no path through which an
    adapter-side K/V LoRA delta could affect the cache or the attention output.
    Silently dropping a configured K/V adapter would be a footgun, so reject the
    combination at construction time.

    ``"all-linear"`` is forbidden outright: PEFT expands it to every ``nn.Linear``
    in the model, and SR's ``k_proj`` / ``v_proj`` are ``_StreamGatedLinear``
    (an ``nn.Linear`` subclass), so it *would* wrap K/V — the delta would then be
    trained and saved but never fire (base-only K/V calls ``base_layer``). Rejecting
    the string here (rather than only inspecting an explicit name list) closes that
    hole; SR has no legitimate all-linear use, since it always includes forbidden K/V.
    """
    targets = lora_config.target_modules
    if targets is None:
        return
    if isinstance(targets, str):
        if targets == "all-linear":
            raise ValueError(
                'SR forbids target_modules="all-linear" (it expands to every '
                "nn.Linear, including the forbidden K/V projections — shared "
                "base-only K/V has no place for an adapter-side delta to land). "
                "List the SR target modules explicitly (q_proj, o_proj, mlp, "
                "cross_stream, …) instead."
            )
        return
    bad = [name for name in targets if name in {"k_proj", "v_proj"}]
    if bad:
        raise ValueError(
            "SR forbids LoRA on K/V projections (shared base-only K/V — no place "
            "for an adapter-side delta to land; K/V is computed once from "
            f"normed_base). Remove these target_modules entries: {bad}."
        )


def _reject_tap_in_both_selections(lora_config: LoraConfig) -> None:
    """Forbid a cross-stream tap appearing in BOTH PEFT selections.

    A tap's PEFT list IS its type: ``target_modules`` → a frozen-zero
    :class:`CrossStream` made trainable by stock ``lora.Linear``;
    ``modules_to_save`` → a directly-trainable non-LoRA module (``linear`` /
    ``monarch``) wrapped by ``ModulesToSaveWrapper``. Naming the same site in both
    asks for two mutually exclusive types at one address. PEFT would apply both
    wrappers and the resulting module is neither what the config says nor
    reloadable, so reject it here rather than let it surface as an opaque shape
    error inside ``PeftModel.from_pretrained``.
    """
    targets = lora_config.target_modules
    if targets is None or isinstance(targets, str):
        return
    saved = set(getattr(lora_config, "modules_to_save", None) or [])
    both = [n for n in CROSS_STREAM_TAPS if n in set(targets) and n in saved]
    if both:
        raise ValueError(
            "A cross-stream tap must appear in exactly ONE of target_modules "
            "(cross_stream_type='lora') or modules_to_save "
            f"(cross_stream_type='linear'/'monarch'), not both: {both}."
        )


def get_shadow_residual_peft_model(
    base_model_name_or_path: str,
    lora_config: LoraConfig,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    adapter_name: str = "default",
    attn_implementation: Optional[str] = None,
    share_moe_routing: bool = True,
    cross_stream_tap_types: Optional["Mapping[str, Any]"] = None,
    cross_stream_type: Optional[str] = None,
    cross_stream_dim: Optional[int] = None,
):
    """Build a :class:`peft.PeftModel` for shadow-residual + LoRA.

    SR uses a single topology: shared base-only K/V dual-stream. K/V LoRA is
    forbidden (no place for a K/V delta to land).

    Args:
        base_model_name_or_path: HF model id or local path of the base
            Granite model.
        lora_config: a stock :class:`peft.LoraConfig`.  Driving fields:

            * ``target_modules`` — projection names to attach LoRA on, plus one or
              more cross-stream tap names (``"cross_stream"``,
              ``"cross_stream_post_attn"``; see
              :data:`shadow_residual.shadow_residual.cross_stream.CROSS_STREAM_TAPS`)
              to opt into shadow residual (``"lora"``-type cross-stream). Must NOT
              include ``k_proj`` / ``v_proj``.
            * ``r``, ``lora_alpha``, ``lora_dropout`` — defaults.
            * ``rank_pattern`` / ``alpha_pattern`` — per-module rank / alpha
              overrides, e.g. ``rank_pattern={"cross_stream": R}`` (``"lora"``-type
              taps only).
            * ``modules_to_save`` — cross-stream taps whose type is NOT ``"lora"``
              (``"linear"`` / ``"monarch"``): each is a directly-trainable module
              persisted whole, not LoRA-wrapped (see
              config.adapters.to_peft_config). The taps BUILT on the model are
              derived from the union of ``target_modules`` (lora type) and
              ``modules_to_save`` (non-lora types); a tap in BOTH is rejected.
            The adapter is plain LoRA (always active); there is no gated activation.
        torch_dtype: dtype for the base model construction.
        cross_stream_tap_types: the cross-stream layer TYPE per tap — the axis
            ORTHOGONAL to the wiring (the tap NAME is the wiring). A
            ``{tap -> "lora" | "linear" | "monarch"}`` or
            ``{tap -> {"type", "num"}}`` map, so one model can MIX types across
            taps. STRUCTURAL: it decides which module the SR base builds at each
            site, so it is threaded onto the SR config and every rank builds the
            same tree (FSDP-symmetric). Taps omitted from the map default to
            ``"lora"``. ``num`` is the tap's numeric parameter and its meaning is
            type-dependent: unused for ``"lora"`` (the rank comes from
            ``rank_pattern``), provenance-only for ``"linear"`` (the matrix is
            always H×H), and the block count ``b`` for ``"monarch"``.
        cross_stream_type: legacy scalar form of the above — one type applied to
            EVERY built tap. Ignored when ``cross_stream_tap_types`` is given.
        cross_stream_dim: legacy scalar ``num`` paired with ``cross_stream_type``,
            applied to every built non-lora tap.
        adapter_name: PEFT adapter name (default: ``"default"``).
        share_moe_routing: MoE bases only — route once on the base stream and
            reuse that expert partition for the adapter stream (vs. independent
            per-stream routing); on by default. No-op on dense bases. train.py
            records it into the saved ``adapter_config.json`` so the serving path
            can source it back rather than hand-passing.

    Returns:
        :class:`peft.PeftModel`.
    """
    _reject_kv_lora(lora_config)
    _reject_tap_in_both_selections(lora_config)
    # Which cross-stream sites to build is DERIVED from the LoRA config: the tap
    # set is exactly the set of cross_stream* sites the adapter references, taken
    # from BOTH PEFT selections. For the "lora" type those are LoRA targets (in
    # ``target_modules``); for the non-lora types ("linear" / "monarch") they are
    # trainable modules persisted whole (in ``modules_to_save``). Deriving from the
    # union means there is no second config field to drift, and the saved
    # adapter_config.json already records everything the serving path needs
    # (target_modules + modules_to_save).
    cross_stream_taps = cross_stream_taps_from_target_modules(
        lora_config.target_modules,
        getattr(lora_config, "modules_to_save", None),
    )
    # Resolve the TYPE axis once, here, so the log line and the SR config agree.
    tap_types = normalize_cross_stream_tap_types(
        cross_stream_taps,
        cross_stream_tap_types if cross_stream_tap_types is not None else cross_stream_type,
        cross_stream_dim,
    )
    # A tap engages the dual-stream forward once PEFT WRAPS it — as a lora.Linear
    # (target_modules) or a ModulesToSaveWrapper (modules_to_save). An unreferenced
    # tap is a bare frozen zero and stays on the single-stream path, so test both
    # selections here, not just target_modules.
    _referenced = set(lora_config.target_modules or []) | set(
        getattr(lora_config, "modules_to_save", None) or []
    )
    logger.info(
        "Building SR base from upstream HF Granite. target_modules=%s "
        "cross_stream_taps=%s (dual-stream forward %s engage).",
        lora_config.target_modules,
        {t: tap_types[t]["type"] for t in cross_stream_taps},
        "will" if _referenced & set(cross_stream_taps) else "will not",
    )

    # Path-D pipeline: every rank builds SR + PEFT-wrap on meta with an
    # identical traversal order (config-driven, no from_pretrained
    # divergence). Only rank 0 then materializes real CPU storage and
    # loads upstream base weights. FSDP's sync_module_states broadcasts
    # rank-0's values onto the (sharded) tensors of other ranks.
    sr_config = _build_sr_config(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        share_moe_routing=share_moe_routing,
        cross_stream_taps=cross_stream_taps,
        cross_stream_tap_types=tap_types,
    )

    peft_model = _build_sr_peft_model_meta(
        sr_config,
        lora_config,
        torch_dtype=torch_dtype,
        adapter_name=adapter_name,
    )

    # Materialize meta → real weights. Under FSDP, only rank 0 materializes and
    # loads upstream weights; FSDP's sync_module_states broadcasts them to the
    # (still-meta) other ranks after sharding, which keeps peak host RAM to one
    # rank's copy. WITHOUT FSDP (single-GPU or plain DDP), there is no broadcast,
    # so every rank must materialize its own real weights — otherwise non-rank-0
    # processes keep meta tensors and the Trainer's `.to(device)` fails with
    # "Cannot copy out of meta tensor". FSDP is detected via accelerate's
    # ACCELERATE_USE_FSDP env (set by `accelerate launch --config_file <fsdp>`).
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    fsdp_active = os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true"
    if local_rank == 0 or not fsdp_active:
        _materialize_and_transfer(
            peft_model,
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

    # Re-tie lm_head ↔ embed_tokens — CONFIG-DRIVEN, and only when the base
    # ties them (``tie_word_embeddings=True``, e.g. granite-4.1).
    #
    # TIED base (granite-4.1): ``lm_head.weight`` must share storage with
    # ``model.embed_tokens.weight``. The meta build (and rank-0's ``to_empty`` in
    # ``_materialize_and_transfer``) leaves them as two separate tensors; without
    # re-tying, the loss-producing projection is disjoint from the trained
    # embedding → frozen/divergent loss (seen on the DDP path as a collapse to
    # acc 0.0438). ``ShadowResidualForCausalLM.__init__`` sets
    # ``_tied_weights_keys`` accordingly, so ``tie_weights()`` re-aliases lm_head.
    #
    # UNTIED base (granite-4.2, ``tie_word_embeddings=False``): the checkpoint
    # ships a SEPARATELY-TRAINED ``lm_head.weight`` that ``transfer_base_weights``
    # already copied into SR's lm_head. Calling ``tie_weights()`` here would
    # OVERWRITE that real head with the embedding matrix → garbage logits → NaN
    # loss on step 1. So we skip it; the model's ``_tied_weights_keys`` is ``{}``
    # in this case anyway (tie_weights would be a no-op), but we gate explicitly
    # for clarity.
    #
    # FSDP symmetry (why this is safe on all ranks): the tie/untie decision is
    # driven by ``sr_config.tie_word_embeddings``, which every rank sees
    # identically, and the Parameter-sharing structure is settled in
    # ``__init__``/``post_init`` (runs on every rank incl. meta) — NOT introduced
    # only in the rank-0 materialize branch. So the module structure is identical
    # across ranks before the FSDP wrap in both cases. (Tying on only rank 0
    # previously made rank 0 differ from the still-meta ranks → FSDP flatten /
    # sync_module_states mismatch → ``_ALLGATHER_BASE`` hang, Signal 6, no
    # checkpoint. Keeping the decision config-driven and rank-symmetric avoids
    # that.) This call runs on every rank in the tied case; not calling it in the
    # untied case is likewise symmetric.
    if getattr(sr_config, "tie_word_embeddings", True):
        peft_model.base_model.model.tie_weights()

    _disable_merge_and_unload(peft_model)
    return peft_model


__all__ = ["get_shadow_residual_peft_model"]
