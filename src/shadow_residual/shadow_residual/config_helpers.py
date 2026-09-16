# SPDX-License-Identifier: Apache-2.0
"""Config-side helpers for shadow-residual.

``ShadowResidualConfig`` does not declare ``shadow_residual`` or
``cross_stream_rank`` as class attributes. ``PretrainedConfig`` accepts
unknown ``**kwargs`` and stores them as plain attributes, so passing them
through the config constructor works at runtime — but we lose declarative
validation.

This module reproduces that validation as a free function. Call
:func:`validate_shadow_residual_config` immediately after constructing a
``ShadowResidualConfig`` whenever the SR fields are set.
"""

from typing import Iterable

from shadow_residual.shadow_residual.cross_stream import (
    CROSS_STREAM_TAPS,
    validate_cross_stream_tap_types,
)
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig


def validate_shadow_residual_config(config: ShadowResidualConfig) -> None:
    """Validate shadow-residual related config fields.

    Mirrors the constructor-time checks from the dual-residual branch:

    - ``cross_stream_rank`` requires ``shadow_residual=True``.  Note: the
      HF backend no longer consumes ``cross_stream_rank`` — cross-stream
      LoRA rank is provided by PEFT's ``LoraConfig.r`` /
      ``rank_pattern["cross_stream"]``.  The field is retained on config
      objects for backward compatibility with the vLLM backend and saved
      checkpoints.
    - ``shadow_residual=True`` requires every layer to be of type
      ``"attention"`` (no SSM / mamba layers). Note: sparse-MoE bases (Granite
      5.0 / ``granitemoe``, ``num_local_experts > 0``) are attention-only for
      token mixing — the experts are the FFN, not a layer type — so they pass
      this check.
    - When ``num_local_experts > 0``, ``num_experts_per_tok`` must be a positive
      integer no larger than ``num_local_experts`` (the top-k router cannot select
      more experts than exist).
    - ``cross_stream_taps``, when set, must be a non-empty list of names drawn from
      :data:`shadow_residual.shadow_residual.cross_stream.CROSS_STREAM_TAPS`.
      An unknown name would only surface as a ``KeyError`` deep inside the decoder
      forward, so catch it here.
    - ``cross_stream_tap_types``, when set, must name only registered taps that are
      actually BUILT (a type for an unbuilt tap has no effect and is almost always a
      typo), with a known type name, and — for ``"monarch"`` — a block count that
      divides ``hidden_size``. See
      :func:`shadow_residual.shadow_residual.cross_stream.validate_cross_stream_tap_types`.

    SR uses a single K/V topology (shared base-only K/V), so there is no
    ``shared_base_kv`` toggle to validate.

    Raises:
        ValueError: if any of the above is violated.
    """
    shadow_residual = bool(getattr(config, "shadow_residual", False))
    cross_stream_rank = getattr(config, "cross_stream_rank", None)

    if cross_stream_rank is not None and not shadow_residual:
        raise ValueError("cross_stream_rank requires shadow_residual=True")

    if not shadow_residual:
        return

    layer_types: Iterable[str] = getattr(config, "layer_types", []) or []
    if any(lt != "attention" for lt in layer_types):
        raise ValueError(
            "shadow_residual=True requires all layer_types to be 'attention' "
            "(no SSM / mamba layers)."
        )

    taps = getattr(config, "cross_stream_taps", None)
    if taps is not None:
        if not taps:
            raise ValueError(
                "cross_stream_taps must name at least one cross-stream site; "
                f"valid names: {sorted(CROSS_STREAM_TAPS)}."
            )
        unknown = [name for name in taps if name not in CROSS_STREAM_TAPS]
        if unknown:
            raise ValueError(
                f"Unknown cross_stream_taps entries: {unknown}. "
                f"Valid names: {sorted(CROSS_STREAM_TAPS)}."
            )

    validate_cross_stream_tap_types(
        int(getattr(config, "hidden_size", 0) or 0),
        getattr(config, "cross_stream_tap_types", None),
        taps,
    )

    num_local_experts = int(getattr(config, "num_local_experts", 0) or 0)
    if num_local_experts > 0:
        top_k = getattr(config, "num_experts_per_tok", None)
        if not isinstance(top_k, int) or top_k <= 0 or top_k > num_local_experts:
            raise ValueError(
                "num_experts_per_tok must be a positive integer <= "
                f"num_local_experts ({num_local_experts}); got {top_k!r}."
            )


def set_shadow_residual(
    config: ShadowResidualConfig,
    *,
    enabled: bool = True,
    cross_stream_rank: "int | None" = None,
) -> ShadowResidualConfig:
    """Mutate ``config`` to enable shadow-residual and validate the result.

    SR uses a single K/V topology (shared base-only K/V) — K/V is computed once
    from the base stream and shared with the adapter stream's Q; LoRA on
    ``k_proj`` / ``v_proj`` is forbidden (enforced at PEFT-attach time).

    Returns the same config object for chaining.
    """
    config.shadow_residual = enabled
    config.cross_stream_rank = cross_stream_rank
    validate_shadow_residual_config(config)
    return config


__all__ = [
    "validate_shadow_residual_config",
    "set_shadow_residual",
]
