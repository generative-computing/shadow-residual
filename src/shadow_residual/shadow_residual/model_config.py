# SPDX-License-Identifier: Apache-2.0
"""Vendored config class for the shadow-residual model.

``ShadowResidualConfig`` is a thin subclass of transformers'
:class:`~transformers.GraniteMoeHybridConfig`. It exists so this package is
self-contained — the shadow-residual model code needs a concrete
``config_class`` to declare, and the fields it reads (``hidden_size``,
``num_attention_heads``, ``embedding_multiplier``, ``shared_intermediate_size``,
``projection_head_dim`` …) all live on the Granite hybrid config.

The SR-specific knobs (``shadow_residual``, ``cross_stream_rank``,
``shared_base_kv``) are *not* declared here on purpose:
:class:`~transformers.PretrainedConfig` stores unknown ``**kwargs`` as plain
attributes, and :func:`shadow_residual.shadow_residual.config_helpers.set_shadow_residual`
sets them dynamically after construction (then validates). Declaring them would
duplicate that path without benefit.

Note: this is a clean break from the previous ``granite_switch`` model type.
Old full-model checkpoints with ``model_type: "granite_switch"`` in their
``config.json`` will no longer auto-resolve via ``AutoConfig``. The adapter
workflow (``build_sr_base`` + ``PeftModel.from_pretrained``) is unaffected —
it builds from the upstream Granite base model.
"""

from __future__ import annotations

from typing import List, Optional

from transformers import GraniteMoeHybridConfig


class ShadowResidualConfig(GraniteMoeHybridConfig):
    """Config for the shadow-residual model.

    Extends the Granite 4 hybrid config with the two structural defaults the
    SR model depends on:

    - ``shared_intermediate_size`` defaults to ``intermediate_size`` when unset.
      Some Granite hybrid configs ship a smaller shared-MLP size (or leave it
      None); SR builds its unfused ``gate_proj`` / ``up_proj`` from the shared
      MLP, so the two must match or the fused→unfused weight transfer fails on
      a shape mismatch and the MLP stays randomly initialized.
    - ``projection_head_dim`` — the per-head Q/K/V projection width, read by
      ``attention_hf`` and ``weight_transfer``. Defaults to an explicit
      ``head_dim`` when the base config carries one, else
      ``hidden_size // num_attention_heads``.

    ``layer_types`` is normalized to all-``"attention"`` when absent — SR is an
    attention-only architecture (no SSM / mamba layers).
    """

    model_type = "shadow_residual"

    def __init__(
        self,
        fused_add_norm: bool = False,
        num_local_experts: int = 0,
        position_embedding_type: str = "rope",
        layer_types: Optional[List[str]] = None,
        **kwargs,
    ):
        # layer_types must have length == num_hidden_layers so the KV cache
        # pre-allocation matches the decoder's global layer indices. SR is
        # attention-only, so default to all-"attention".
        if layer_types is None:
            num_hidden_layers = kwargs.get("num_hidden_layers", 32)
            layer_types = ["attention"] * num_hidden_layers

        super().__init__(
            num_local_experts=num_local_experts,
            position_embedding_type=position_embedding_type,
            layer_types=layer_types,
            **kwargs,
        )

        # All Granite 4 models use shared_mlp naming; for dense models the
        # shared MLP is the full MLP, so shared_intermediate_size must equal
        # intermediate_size when it wasn't given explicitly.
        if getattr(self, "shared_intermediate_size", None) is None:
            self.shared_intermediate_size = self.intermediate_size

        # vLLM residual-norm convention (kept for checkpoint/back-compat).
        self.fused_add_norm = fused_add_norm

        # Per-head projection width. We do NOT set head_dim here because HF's
        # RoPE reads it; use an explicit head_dim from kwargs when present.
        explicit_head_dim = kwargs.get("head_dim")
        self.projection_head_dim = (
            explicit_head_dim
            if explicit_head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )


__all__ = ["ShadowResidualConfig"]
