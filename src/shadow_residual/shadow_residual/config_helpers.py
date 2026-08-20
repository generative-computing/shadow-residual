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
      ``"attention"`` (no SSM / mamba layers).

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
