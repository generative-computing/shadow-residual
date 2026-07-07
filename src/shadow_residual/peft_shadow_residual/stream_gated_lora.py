# SPDX-License-Identifier: Apache-2.0
"""Custom PEFT LoRA wrapper for shadow-residual stream-gated projections.

Registered with
``LoraConfig._register_custom_module({_StreamGatedLinear: ShadowResidualLora})``.
PEFT's ``LoraModel`` then walks the SR decoder, finds modules whose
attribute name matches ``target_modules`` (e.g. ``q_proj``), and — when
the underlying module is a :class:`_StreamGatedLinear` — replaces it with
this wrapper instead of stock :class:`peft.tuners.lora.layer.Linear`.

The only behavioural difference from stock :class:`Linear` is the
:func:`current_stream` check at the top of :meth:`forward`:

- ``current_stream() == "base"`` → return ``base_layer(x)`` only (no LoRA
  delta, no ALORA, no DoRA).  The decoder calls every projection inside
  a ``stream_context("base")`` block when computing the frozen-base
  stream, which restores the frozen-base invariant from the original
  dual-residual design (Q/O/MLP base contribution is identical to the
  unadapted base model).
- otherwise → delegate to ``super().forward(x, ...)``, i.e. stock peft
  behaviour: LoRA delta, ALORA gating, DoRA, etc.

K/V projections in :class:`ShadowResidualAttention` are called *outside*
any context (default tag ``"adapter"``), so when the user lists
``"k_proj"``/``"v_proj"`` in ``target_modules`` the K/V LoRA delta is
applied normally — the single shared KV cache picks up the delta. This
is the documented K/V exception to the frozen-base invariant.

Inheriting from stock :class:`Linear` (rather than rebuilding the
wrapper from :class:`LoraLayer`) means every PEFT capability that
stock :class:`Linear` supports is automatically inherited:
:class:`ALoraLinearVariant`, DoRA, all init recipes (PiSSA, OLoRA, EVA,
LoftQ, …), rslora, mixed-batch forwards, save/load round-trips through
the standard ``adapter_model.safetensors`` keys.

``merge_and_unload`` is intentionally disabled at the
:class:`peft.PeftModel` level by the SR factory — merging the LoRA
delta into the base linear would silently fold the adapter delta into
the frozen-base path, breaking the invariant.  See ``factory.py``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from peft.tuners.lora.variants import ALoraLinearVariant

from shadow_residual.shadow_residual._stream_context import (
    current_stream,
    offset_recovery_active,
)


class RecomputeStableALoraVariant(ALoraLinearVariant):
    """aLoRA variant whose forward is safe under activation-checkpoint recompute.

    Stock :class:`ALoraLinearVariant.forward` (peft ``variants.py``) applies the
    delta with a boolean mask-select + in-place scatter:
    ``res_flat[mask_flat] += lora_B(lora_A(dropout(x_flat[mask_flat])))``. The
    number of tensors saved for autograd equals ``mask.sum()`` — data-dependent
    on the aLoRA invocation offset. Under FSDP activation-checkpoint recompute
    the mask population differs from the original forward (offsets do not survive
    the checkpoint boundary identically), so a different NUMBER of tensors is
    saved on recompute and ``torch.utils.checkpoint`` raises CheckpointError
    ("a different number of tensors was saved during forward and recomputation").

    This override computes the delta over ALL tokens with fixed shapes, then
    multiplies by a ``[B, T, 1]`` float mask (1.0 from the invocation offset
    onward, 0.0 before). Same ops and identical saved-tensor count on every pass
    regardless of mask population, so recompute cannot diverge. Numerically
    identical to the stock variant: masked-out positions get ``delta * 0``.
    """

    @staticmethod
    def forward(
        module: Any,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        alora_offsets = kwargs.get("alora_offsets", None)
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        x = x.to(lora_A.weight.dtype)
        result_shape = result.shape
        B = result_shape[0]
        T = result_shape[1] if len(result_shape) == 3 else 1
        device = result.device

        # Inference (no autograd) is NOT activation-checkpointed, so there is no
        # recompute to diverge — use the memory-efficient masked-select (stock
        # aLoRA behaviour): compute the delta only on the active tokens. This
        # matters at generation time where the full-token delta on a 32768-wide
        # MLP projection (gate/up_proj) would materialise multi-GB tensors and
        # OOM a single-GPU 30B inference. Only the training path (grad enabled,
        # under gradient checkpointing) needs the fixed-shape full-token form.
        if not torch.is_grad_enabled():
            return ALoraLinearVariant.forward(
                module, active_adapter=active_adapter, x=x, result=result, **kwargs
            )

        # Recover alora_offsets across the activation-checkpoint recompute boundary.
        # Inside the SR checkpointed decoder loop, PEFT's offset hook is gone on the
        # recompute pass so alora_offsets arrives as None — which would wrongly zero
        # the gated delta during recompute and corrupt gradients. Cache the offsets
        # (keyed by batch/seq shape) on the original forward and reuse them on the
        # recompute. Gated on offset_recovery_active() so that OUTSIDE this loop
        # (direct calls, inference) a None still means a genuine "no invocation →
        # zero delta", preserving stock semantics. See _stream_context.py.
        if offset_recovery_active():
            cache_key = (B, T)
            if alora_offsets is not None:
                module._sr_alora_offsets_cache = (cache_key, alora_offsets)
            else:
                cached = getattr(module, "_sr_alora_offsets_cache", None)
                if cached is not None and cached[0] == cache_key:
                    alora_offsets = cached[1]

        if alora_offsets is None:
            # No invocation offsets → adapter inactive everywhere (base only).
            float_mask = torch.zeros((B, T, 1), dtype=result.dtype, device=device)
        else:
            offsets = torch.tensor(
                [0 if o is None else min(int(o), T) for o in alora_offsets],
                device=device, dtype=torch.long,
            )
            pos = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
            bool_mask = pos >= (T - offsets).unsqueeze(1)       # [B, T]
            float_mask = bool_mask.to(result.dtype).unsqueeze(-1)  # [B, T, 1]

        # Full-token delta (fixed shape, recompute-stable), gated by the mask.
        delta = lora_B(lora_A(dropout(x))) * scaling
        return result + delta * float_mask


class ShadowResidualLora(PeftLoraLinear):
    """Stock PEFT :class:`Linear` wrapper plus a per-call stream gate.

    The wrapper is constructed by PEFT's ``_create_and_replace``; the
    constructor signature must therefore match
    :class:`peft.tuners.lora.layer.Linear`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Forward everything to stock peft Linear.__init__. PEFT's
        # _create_and_replace constructs us with the same call shape it
        # uses for stock Linear: (target, adapter_name, config=...,
        # r=..., lora_alpha=..., target_name=..., …).
        super().__init__(*args, **kwargs)

    def resolve_lora_variant(self, *, config: Any, **kwargs: Any) -> Optional[Any]:
        """Use the recompute-stable aLoRA variant instead of stock.

        PEFT selects the aLoRA variant when ``config.alora_invocation_tokens``
        is set. The stock variant's masked-select forward saves a
        data-dependent number of tensors and breaks under activation-checkpoint
        recompute (see :class:`RecomputeStableALoraVariant`). Swap in our
        fixed-shape variant; defer to stock peft for every other case (DoRA,
        Arrow, BDLoRA, plain LoRA).
        """
        if getattr(config, "alora_invocation_tokens", None) is not None:
            return RecomputeStableALoraVariant()
        return super().resolve_lora_variant(config=config, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Stream-gated forward.

        When the active stream is ``"base"`` the LoRA delta is gated off so
        the result equals ``base_layer(x)``; any other tag (``"adapter"`` by
        default, used for K/V calls) applies the delta normally.

        The gate is applied by zeroing the adapter ``scaling`` for the
        duration of the base call rather than by branching to a different
        code path. This is deliberate and load-bearing for activation
        checkpointing: under FSDP, the stream contextvar does not survive
        into the checkpoint recompute, so a ``current_stream()``-driven
        *branch* (base = ``base_layer(x)`` only vs adapter = full LoRA) saves
        a different NUMBER of tensors on forward vs recompute and trips
        ``torch.utils.checkpoint``'s recompute checker (CheckpointError: "a
        different number of tensors was saved"). Running the SAME ops in both
        streams and only varying a scalar (``scaling`` 0.0 vs normal) keeps
        the saved-tensor count and metadata identical regardless of what the
        recompute reads — the checker compares count/metadata, never values —
        so the delta is exactly zero on the base stream (frozen-base invariant
        preserved) without introducing recompute-divergent control flow.
        """
        if current_stream() != "base":
            return super().forward(x, *args, **kwargs)

        # Base stream: run the identical LoRA ops with the delta scaled to
        # zero. Restore scaling afterward so adapter-stream calls (same module,
        # invoked again within this forward) are unaffected.
        saved = {a: self.scaling[a] for a in self.active_adapters if a in self.scaling}
        try:
            for a in saved:
                self.scaling[a] = 0.0
            return super().forward(x, *args, **kwargs)
        finally:
            for a, s in saved.items():
                self.scaling[a] = s

    def __repr__(self) -> str:
        return f"ShadowResidualLora({super().__repr__()})"


__all__ = ["ShadowResidualLora"]
