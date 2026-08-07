# SPDX-License-Identifier: Apache-2.0
"""Factory: ``get_shadow_residual_peft_model(base_id, lora_config)``.

Builds a :class:`ShadowResidualForCausalLM` and layers PEFT on top.
Under FSDP, every rank constructs the SR + PEFT module tree on meta
with identical traversal order; only rank 0 materializes real CPU
tensors and loads upstream base weights. ``fsdp_sync_module_states``
broadcasts rank 0's values to other ranks after sharding — peak host
RAM is one rank's copy instead of N.

Whether the dual-stream forward actually runs is a runtime decision made
by the SR model itself (see ``modeling_hf.py``): if no :class:`CrossStream`
site has been replaced by a :class:`CrossStreamLora` wrapper (i.e. the
user did not list ``"cross_stream"`` in ``target_modules``), every forward
takes a single-stream early-exit path with compute equivalent to plain
Granite + the wrapped LoRA deltas. This is why there is no factory-level
branch: one architecture, two forward paths.

ALORA-style mid-sequence activation is provided entirely by stock peft —
set ``LoraConfig(alora_invocation_tokens=[...])`` and every wrapped
projection gets gated automatically.  :class:`CrossStreamLora` consumes the
same ``alora_offsets`` (peft injects it via a pre-forward hook because the
class inherits :class:`peft.tuners.lora.layer.LoraLayer`).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.shadow_residual import (
    ShadowResidualForCausalLM,
)
from shadow_residual.shadow_residual._stream_gated_linear import (
    _StreamGatedLinear,
)
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
)
from shadow_residual.shadow_residual.cross_stream import CrossStream
from shadow_residual.shadow_residual.weight_transfer import (
    transfer_base_weights,
)

from .cross_stream_lora import CrossStreamLora
from .stream_gated_lora import ShadowResidualLora

logger = logging.getLogger(__name__)


CROSS_STREAM_NAME = "cross_stream"


def resolve_shared_base_kv(shared_base_kv: Optional[bool], target_modules) -> bool:
    """Resolve the effective ``shared_base_kv`` for a build.

    ``shared_base_kv=None`` means "auto": the flag defaults to the shared
    base-only K/V topology (the SR production variant) when the adapter
    opts into shadow residual (``"cross_stream"`` in ``target_modules``),
    and to disjoint / adapter-computed K/V otherwise (plain LoRA / aLoRA).
    An explicit ``True`` / ``False`` always wins over the auto default.

    A scalar ``target_modules`` ("all-linear", a regex, or ``None``) can't
    name ``cross_stream``, so it resolves to ``False`` — consistent with
    :func:`_reject_kv_lora_when_shared`, which also leaves scalar shapes
    alone.

    When the auto path lands on ``False`` for a LoRA/aLoRA config, a
    warning is logged so the topology choice is visible (K/V is computed
    on the adapter stream, not shared from the frozen base).
    """
    has_cross_stream = (
        target_modules is not None
        and not isinstance(target_modules, str)
        and CROSS_STREAM_NAME in target_modules
    )
    if shared_base_kv is not None:
        return bool(shared_base_kv)
    if has_cross_stream:
        return True
    logger.warning(
        "shared_base_kv unset and 'cross_stream' not in target_modules "
        "(LoRA/aLoRA config) — defaulting to shared_base_kv=False: K/V is "
        "computed on the adapter stream (disjoint topology). Set "
        "shared_base_kv explicitly to silence this warning."
    )
    return False


def _register_cross_stream(lora_config: LoraConfig) -> None:
    """Attach SR custom-module dispatch to ``lora_config``.

    Two registrations:

    - :class:`CrossStream` → :class:`CrossStreamLora`: turns the no-op
      cross-stream site into the rank-R base→adapter projection.
    - :class:`_StreamGatedLinear` → :class:`ShadowResidualLora`: turns
      every Q/K/V/O/gate/up/down projection in the SR decoder into a
      stream-gated LoRA wrapper (LoRA delta gated off in
      ``stream_context("base")``).
    """
    lora_config._register_custom_module(
        {
            CrossStream: CrossStreamLora,
            _StreamGatedLinear: ShadowResidualLora,
        }
    )


def _build_sr_base(
    base_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    shared_base_kv: bool = False,
) -> ShadowResidualForCausalLM:
    """Single-process SR base build (no FSDP / meta machinery).

    Used by :func:`load_shadow_residual_peft_model` for single-GPU
    inference paths where every rank-equivalent process needs a fully
    real SR model and the FSDP cpu-ram-efficient pattern doesn't apply.
    The training-time FSDP path goes through
    :func:`get_shadow_residual_peft_model` instead, which keeps non-rank-0
    processes on meta until the broadcast.
    """
    sr_config = _build_sr_config(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        shared_base_kv=shared_base_kv,
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


def _build_sr_config(
    base_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    shared_base_kv: bool = False,
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
    if "intermediate_size" in config_dict and config_dict.get("shared_intermediate_size") is None:
        config_dict["shared_intermediate_size"] = config_dict["intermediate_size"]
    sr_config = ShadowResidualConfig(**config_dict)
    set_shadow_residual(sr_config, enabled=True, shared_base_kv=shared_base_kv)
    if torch_dtype is not None:
        sr_config.torch_dtype = torch_dtype
    if attn_implementation is not None:
        sr_config._attn_implementation = attn_implementation
    return sr_config


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


def _reinit_rope_buffers(model) -> None:
    """Recompute non-persistent RoPE ``inv_freq`` buffers after meta-materialize.

    Walks every rotary module (anything carrying a real ``inv_freq`` buffer)
    and recomputes its frequencies by re-instantiating the rotary class
    fresh on CPU — its ``__init__`` runs the correct rope-init for whatever
    rope_type the config declares, so we don't reimplement the HF dispatch
    (which varies across transformers versions). The freshly-computed
    ``inv_freq`` / ``original_inv_freq`` are copied in-place into the
    materialized buffers. No-op when the model has no rotary module (e.g.
    configs that disable RoPE). Idempotent and safe on a fully-real model.
    See the caller for why this is required under the FSDP meta-init path.
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


def _diagnose_materialized(peft_model) -> None:
    """Rank-0 finite/zero audit of every parameter and buffer post-materialize."""
    nonfinite, allzero, n_total = [], [], 0
    for name, t in list(peft_model.named_parameters()) + list(peft_model.named_buffers()):
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
        _register_cross_stream(lora_config)
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
            "Use save_pretrained / load_shadow_residual_peft_model for "
            "checkpoint round-trips instead."
        )

    peft_model.merge_and_unload = _raise
    if hasattr(peft_model, "base_model"):
        peft_model.base_model.merge_and_unload = _raise
        peft_model.base_model.merge_adapter = _raise
        peft_model.base_model.unmerge_adapter = _raise


def _reject_kv_lora_when_shared(lora_config: LoraConfig) -> None:
    """Forbid LoRA on ``k_proj`` / ``v_proj`` when ``shared_base_kv=True``.

    In shared-base-K/V mode the adapter stream never multiplies
    ``normed_adapt`` by ``W_K`` or ``W_V`` — there is no path through
    which an adapter-side K/V LoRA delta could affect the cache or the
    attention output. Silently dropping a configured K/V adapter would
    be a footgun, so reject the combination at construction time.
    """
    targets = lora_config.target_modules
    if targets is None or isinstance(targets, str):
        # Scalar shapes ("all-linear", regex, etc.) — leave alone; the
        # config_helpers validator only catches the explicit list form.
        # The user opted into "all linears" intentionally; shared_base_kv
        # is only a meaningful pairing with named target_modules.
        return
    bad = [name for name in targets if name in {"k_proj", "v_proj"}]
    if bad:
        raise ValueError(
            "shared_base_kv=True forbids LoRA on K/V projections (no "
            "place for an adapter-side delta to land — K/V is computed "
            f"once from normed_base). Got target_modules entries: {bad}."
        )


def get_shadow_residual_peft_model(
    base_model_name_or_path: str,
    lora_config: LoraConfig,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    adapter_name: str = "default",
    attn_implementation: Optional[str] = None,
    shared_base_kv: Optional[bool] = None,
):
    """Build a :class:`peft.PeftModel` for shadow-residual + LoRA.

    Args:
        base_model_name_or_path: HF model id or local path of the base
            Granite model.
        lora_config: a stock :class:`peft.LoraConfig`.  Driving fields:

            * ``target_modules`` — projection names to attach LoRA on, plus
              optionally ``"cross_stream"`` to opt into shadow residual.
            * ``r``, ``lora_alpha``, ``lora_dropout`` — defaults.
            * ``rank_pattern={"cross_stream": R}`` — overrides cross-stream
              rank.
            * ``alora_invocation_tokens`` — when set, every wrapped
              projection is gated to positions ≥ the last invocation match.
              :class:`CrossStreamLora` honours the same gate.
        torch_dtype: dtype for the base model construction.
        adapter_name: PEFT adapter name (default: ``"default"``).
        shared_base_kv: when True, the adapter stream attends to a
            single base-only K/V (one cache; K/V computed once from
            ``normed_base``). Forbids K/V LoRA. See
            :mod:`shadow_residual.shadow_residual.attention_hf`. ``None``
            (the default) auto-resolves via :func:`resolve_shared_base_kv`:
            True when ``"cross_stream"`` is in ``target_modules``, else
            False (with a warning).

    Returns:
        :class:`peft.PeftModel`.
    """
    shared_base_kv = resolve_shared_base_kv(shared_base_kv, lora_config.target_modules)
    if shared_base_kv:
        _reject_kv_lora_when_shared(lora_config)
    logger.info(
        "Building SR base from upstream HF Granite; registering CrossStreamLora. "
        "target_modules=%s — cross_stream %s in target_modules. shared_base_kv=%s",
        lora_config.target_modules,
        "is" if CROSS_STREAM_NAME in (lora_config.target_modules or []) else "is not",
        shared_base_kv,
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
        shared_base_kv=shared_base_kv,
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
