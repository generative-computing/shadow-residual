# SPDX-License-Identifier: Apache-2.0
"""Custom PEFT LoRA layer for the shadow-residual cross-stream site.

Registered with ``LoraConfig._register_custom_module({CrossStream: CrossStreamLora})``
so PEFT's ``LoraModel`` walks the SR decoder, finds modules whose attribute
name matches ``"cross_stream"`` in ``target_modules``, and replaces the
no-op :class:`~shadow_residual.shadow_residual.cross_stream.CrossStream`
target with this wrapper.

Once installed, the wrapper owns the rank-R LoRA weights ``A`` (Kaiming init)
and ``B`` (zero init).  Its ``forward`` returns ``B(A(x)) * scaling`` — the
cross-stream contribution that the SR decoder adds to ``h_adapt``.

Rank, alpha, dropout, and rslora flow from the user's :class:`peft.LoraConfig`
exactly as for any other LoRA target.  ``rank_pattern={"cross_stream": 64}``
or ``alpha_pattern`` overrides apply automatically — PEFT's
``_create_and_replace`` resolves them before constructing this class.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from peft.tuners.lora import LoraConfig
from peft.tuners.lora.layer import LoraLayer

from shadow_residual.shadow_residual.cross_stream import CrossStream
from shadow_residual.shadow_residual._stream_context import (
    offset_recovery_active,
)


class CrossStreamLora(nn.Module, LoraLayer):
    """LoRA wrapper for a :class:`CrossStream` site.

    The base layer is parameter-free — it returns zeros — so the result of
    ``forward(x)`` is purely the LoRA delta ``B(A(x)) * scaling``.  This is
    exactly the rank-R cross-stream projection from the base stream into
    the adapter stream.

    Layout matches PEFT's :class:`~peft.tuners.lora.layer.Linear`:

    - ``self.lora_A``: ``ModuleDict[adapter_name -> nn.Linear(H, r, bias=False)]``,
      Kaiming-init.
    - ``self.lora_B``: ``ModuleDict[adapter_name -> nn.Linear(r, H, bias=False)]``,
      zero-init.
    - ``self.scaling[adapter_name]``: ``alpha / r`` (or ``alpha / sqrt(r)``
      under ``use_rslora``).
    - ``self.lora_dropout[adapter_name]``: ``nn.Dropout(p)`` or ``nn.Identity``.

    Save / load: the standard PEFT save format (``adapter_model.safetensors``)
    captures these ``ModuleDict``s under the natural state-dict key
    ``...cross_stream.lora_A.<adapter>.weight``, so no SR-specific sidecar is
    needed.
    """

    def __init__(
        self,
        base_layer: CrossStream,
        adapter_name: str,
        config: LoraConfig,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        # LoraLayer.__init__ wires up the LoRA bookkeeping dicts and reads
        # in_features / out_features from the base layer (which CrossStream
        # exposes — see cross_stream.py).
        LoraLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name
        # Track whatever fan_in_fan_out is on the config (LoraConfig default
        # is False).  Cross-stream is always a "fan-in-fan-out=False" linear.
        self.fan_in_fan_out = getattr(config, "fan_in_fan_out", False)

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            config=config,
        )

    # ------------------------------------------------------------------
    # PEFT integration: resolve_lora_variant + update_layer
    # ------------------------------------------------------------------

    def resolve_lora_variant(self, *, config: LoraConfig, **kwargs):  # noqa: D401
        """No specialized variants (DoRA / OLoRA / EVA) supported here."""
        return None

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        config: LoraConfig,
        **kwargs: Any,
    ) -> None:
        """Install the rank-R LoRA pair for ``adapter_name``.

        Idempotent across multiple calls with different adapter names — PEFT
        invokes this once per ``add_adapter`` call.
        """
        if r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {r}")

        lora_dropout = config.lora_dropout
        init_lora_weights = config.init_lora_weights
        use_rslora = config.use_rslora

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            dropout_layer: nn.Module = nn.Dropout(p=lora_dropout)
        else:
            dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        # Cross-stream is a square H → H projection; both lora_A and lora_B
        # share the same in/out hidden size H == base_layer.in_features.
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        self.lora_bias[adapter_name] = False

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        self.use_rslora[adapter_name] = use_rslora
        self.use_dora[adapter_name] = False

        self._reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _reset_lora_parameters(self, adapter_name: str, init_lora_weights: Any) -> None:
        """Initialize ``A`` to Kaiming-uniform and ``B`` to zero."""
        if init_lora_weights is False:
            return
        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1.0 / self.r[adapter_name])
            else:
                # Fall back to default; advanced inits (PiSSA, OLoRA, EVA, …)
                # require access to a base weight matrix that cross-stream
                # does not have.
                warnings.warn(
                    f"CrossStreamLora ignores init_lora_weights={init_lora_weights!r}; "
                    "falling back to Kaiming-uniform.",
                    stacklevel=2,
                )
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute the cross-stream LoRA contribution.

        ``base_layer(x)`` returns zeros (CrossStream is a no-op site).  The
        result is therefore ``Σ_a  B_a(A_a(dropout(x))) * scaling_a`` summed
        over active adapters — i.e. the rank-R cross-stream projection.

        ALORA gating: when ``alora_offsets`` is in ``kwargs`` (peft injects
        it via a pre-forward hook on every ``LoraLayer`` when the parent
        ``LoraConfig`` has ``alora_invocation_tokens`` set), the delta is
        masked to the trailing ``T - offsets[i]`` positions per row.  This
        mirrors :class:`peft.tuners.lora.variants.ALoraLinearVariant.forward`
        and gives bit-identical pre-invocation behaviour to the unadapted
        base.  Receiving ``alora_offsets=None`` (when ALORA is configured
        but no invocation match was found in a row) means "no delta at all"
        for that row — same convention peft uses.
        """
        # Strip variant kwargs so they don't reach ``CrossStream.forward``.
        alora_offsets = kwargs.pop("alora_offsets", None)
        # ``adapter_names`` is the other peft variant kwarg; ignore it for
        # the bare base call.
        kwargs.pop("adapter_names", None)
        result = self.base_layer(x, *args, **kwargs)

        if self.disable_adapters or self.merged:
            return result

        torch_result_dtype = result.dtype

        # Inference (no autograd) is not activation-checkpointed → no recompute
        # to diverge. Use the memory-efficient masked-select (delta computed
        # only on active tokens) to avoid materialising a full [B, T, H] delta
        # at generation time. The fixed-shape full-token form below is only
        # needed under training's gradient checkpointing. Mirrors
        # RecomputeStableALoraVariant.
        if not torch.is_grad_enabled():
            mask = None
            if alora_offsets is not None:
                B = result.shape[0]
                T = result.shape[1] if result.dim() == 3 else 1
                offsets = torch.tensor(
                    [0 if o is None else min(int(o), T) for o in alora_offsets],
                    device=result.device, dtype=torch.long,
                )
                pos = torch.arange(T, device=result.device).unsqueeze(0)
                mask = pos >= (T - offsets).unsqueeze(1)  # [B, T]
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
                delta = lora_B(lora_A(dropout(x_cast))) * scaling
                if mask is not None:
                    delta = delta * mask.unsqueeze(-1).to(delta.dtype)
                result = result + delta
            return result.to(torch_result_dtype)

        # Always build a [B, T, 1] float mask with FIXED shape/ops, regardless
        # of whether alora_offsets is present. This is load-bearing for
        # activation-checkpoint recompute: a branch on `alora_offsets is None`
        # (build-mask vs skip-mask) saves a different NUMBER of tensors on the
        # original forward vs the recompute (under FSDP the offsets do not
        # arrive identically across the checkpoint boundary), tripping
        # torch.utils.checkpoint's recompute checker ("a different number of
        # tensors was saved"). Computing the same ops every pass — full delta
        # times a float mask (all-ones when no offsets, so a plain LoRA add) —
        # keeps the saved-tensor count and metadata identical and is
        # numerically unchanged. Mirrors RecomputeStableALoraVariant.
        B = result.shape[0]
        T = result.shape[1] if result.dim() == 3 else 1
        device = result.device

        # Recover alora_offsets across the checkpoint recompute boundary, scoped to
        # the SR checkpointed decoder loop (offset_recovery_active()). On recompute
        # PEFT's offset hook is gone so alora_offsets=None; without recovery the
        # cross-stream delta would be applied UNGATED (None→ones below), diverging
        # from the gated original forward. Outside the loop, None keeps its plain
        # meaning. Same contract as RecomputeStableALoraVariant.
        if offset_recovery_active():
            cache_key = (B, T)
            if alora_offsets is not None:
                self._sr_alora_offsets_cache = (cache_key, alora_offsets)
            else:
                cached = getattr(self, "_sr_alora_offsets_cache", None)
                if cached is not None and cached[0] == cache_key:
                    alora_offsets = cached[1]

        if alora_offsets is None:
            # No gating → adapter active everywhere (plain LoRA add).
            float_mask = torch.ones((B, T, 1), dtype=torch_result_dtype, device=device)
        else:
            offsets = torch.tensor(
                [0 if o is None else min(int(o), T) for o in alora_offsets],
                device=device, dtype=torch.long,
            )
            pos = torch.arange(T, device=device).unsqueeze(0)   # [1, T]
            bool_mask = pos >= (T - offsets).unsqueeze(1)        # [B, T]
            float_mask = bool_mask.to(torch_result_dtype).unsqueeze(-1)  # [B, T, 1]

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
            delta = lora_B(lora_A(dropout(x_cast))) * scaling
            result = result + delta * float_mask
        return result.to(torch_result_dtype)

    # ------------------------------------------------------------------
    # merge / unmerge — not meaningful for a no-op base layer
    # ------------------------------------------------------------------

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merging cross-stream LoRA into the base layer is a no-op.

        The base ``CrossStream`` module has no weight to merge into.  We
        keep this method for API compatibility with PEFT's merge tooling
        (``model.merge_and_unload``) but treat it as a no-op: the LoRA
        contribution is computed at every forward call regardless.
        """
        del safe_merge, adapter_names  # unused

    def unmerge(self) -> None:
        """No-op (see :meth:`merge`)."""
        return None

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """Return ``B @ A * scaling`` as an ``(H, H)`` matrix."""
        weight_A = self.lora_A[adapter].weight  # (r, H)
        weight_B = self.lora_B[adapter].weight  # (H, r)
        return (weight_B @ weight_A) * self.scaling[adapter]

    def __repr__(self) -> str:
        return f"CrossStreamLora(in={self.in_features}, out={self.out_features}, adapters={list(self.r.keys())})"


__all__ = ["CrossStreamLora"]
