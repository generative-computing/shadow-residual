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
    - ``shared_base_kv=True`` requires ``shadow_residual=True``. The flag
      is an SR-internal knob (it picks between disjoint per-stream K/V
      and a single base-only K/V shared with the adapter stream).

    The previous check ``num_adapters > 0`` was dropped — SR does not
    use adapter routing or switching scaffolding.

    Raises:
        ValueError: if any of the above is violated.
    """
    shadow_residual = bool(getattr(config, "shadow_residual", False))
    cross_stream_rank = getattr(config, "cross_stream_rank", None)
    shared_base_kv = bool(getattr(config, "shared_base_kv", False))

    if cross_stream_rank is not None and not shadow_residual:
        raise ValueError("cross_stream_rank requires shadow_residual=True")

    if shared_base_kv and not shadow_residual:
        raise ValueError("shared_base_kv=True requires shadow_residual=True")

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
    shared_base_kv: bool = False,
) -> ShadowResidualConfig:
    """Mutate ``config`` to enable shadow-residual and validate the result.

    ``shared_base_kv`` selects between disjoint per-stream K/V (the
    default — adapter K/V is computed from ``normed_adapt`` into a
    second cache) and a single base-only K/V (computed once from
    ``normed_base``, shared with the adapter stream's Q at attention
    time, single cache). The shared mode forbids LoRA on ``k_proj`` /
    ``v_proj`` because there is no place for an adapter-side K/V delta
    to land — that constraint is enforced at PEFT-attach time, not here.

    Returns the same config object for chaining.
    """
    config.shadow_residual = enabled
    config.cross_stream_rank = cross_stream_rank
    config.shared_base_kv = shared_base_kv
    validate_shadow_residual_config(config)
    return config


__all__ = [
    "validate_shadow_residual_config",
    "set_shadow_residual",
]
