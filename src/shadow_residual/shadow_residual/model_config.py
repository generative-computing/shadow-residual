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
      a shape mismatch and the MLP stays randomly initialized. This default is
      only load-bearing in **dense** MLP mode; on a MoE base
      (``num_local_experts > 0``) the SR MLP is the frozen expert bank and does
      not read ``shared_intermediate_size``.
    - ``projection_head_dim`` — the per-head Q/K/V projection width, read by
      ``attention_hf`` and ``weight_transfer``. Defaults to an explicit
      ``head_dim`` when the base config carries one, else
      ``hidden_size // num_attention_heads``.

    ``layer_types`` is normalized to all-``"attention"`` when absent — SR is an
    attention-only architecture (no SSM / mamba layers).

    MoE (Granite 5.0 / ``granitemoe``) bases carry a nonzero
    ``num_local_experts`` (and ``num_experts_per_tok``). These flow straight
    through ``**kwargs`` to the hybrid parent, which already stores them; SR no
    longer pins ``num_local_experts=0``. When experts are present the SR decoder
    layer builds a frozen sparse expert bank instead of a dense SwiGLU (see
    ``decoder_hf.ShadowResidualMLP``); ``intermediate_size`` is then the
    **per-expert** FFN width, not a dense MLP width.
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

        # SR is attention-only and NEVER builds a Mamba mixer, but the
        # GraniteMoeHybridConfig parent runs a strict `validate_architecture`
        # requiring the Mamba dims to be self-consistent with hidden_size:
        #   mamba_expand * hidden_size % mamba_n_heads == 0   AND
        #   mamba_d_head * mamba_n_heads == mamba_expand * hidden_size
        # A `granitemoe` (Granite 5.0) base carries NO mamba fields, so the
        # parent injects its own large defaults (mamba_n_heads=128, expand=2),
        # which fail the divisibility check for any hidden_size the defaults
        # weren't tuned for (e.g. the tiny test configs, or any non-4096 base).
        # Since no Mamba layer is ever constructed, coerce the dims to an
        # explicitly-disabled zero set so the validator passes AND a saved
        # config.json reads as "no mamba here" rather than as fabricated dims.
        #
        # The validator divides by mamba_n_heads (`inter % n_heads`), so a full
        # all-zero set would raise ZeroDivisionError. The zero-family that both
        # passes validation and reads as disabled is: the widths go to 0
        # (expand, d_head, d_state) and the divisor/shape-critical counts go to
        # 1 (n_heads, n_groups, d_conv, chunk_size). With expand=0 →
        # mamba_intermediate=0, so `0 % 1 == 0` and `d_head(0) * n_heads(1) == 0`
        # both hold. This ONLY fires when the incoming dims are inconsistent
        # (the granitemoe / tiny-test case); a real dense hybrid base whose
        # mamba dims are already consistent (e.g. granite-4.0-micro) is left
        # untouched, so its authored mamba metadata survives round-trip.
        hidden_size = kwargs.get("hidden_size")
        if hidden_size is not None:
            expand = kwargs.get("mamba_expand", 2) or 2
            n_heads = kwargs.get("mamba_n_heads")
            d_head = kwargs.get("mamba_d_head")
            mamba_inter = expand * hidden_size
            consistent = (
                isinstance(n_heads, int)
                and isinstance(d_head, int)
                and n_heads > 0
                and mamba_inter % n_heads == 0
                and d_head * n_heads == mamba_inter
            )
            if not consistent:
                # Widths → 0 (disabled); divisor/shape-critical counts → 1 to
                # avoid a divide-by-zero in the parent's validate_architecture.
                kwargs["mamba_expand"] = 0
                kwargs["mamba_d_head"] = 0
                kwargs["mamba_d_state"] = 0
                kwargs["mamba_n_heads"] = 1
                kwargs["mamba_n_groups"] = 1
                kwargs["mamba_d_conv"] = 1
                kwargs["mamba_chunk_size"] = 1
                kwargs["mamba_conv_bias"] = False
                kwargs["mamba_proj_bias"] = False

        super().__init__(
            num_local_experts=num_local_experts,
            position_embedding_type=position_embedding_type,
            layer_types=layer_types,
            **kwargs,
        )

        # All Granite 4 models use shared_mlp naming; for dense models the
        # shared MLP is the full MLP, so shared_intermediate_size must equal
        # intermediate_size when it wasn't given explicitly. On a MoE base
        # (num_local_experts > 0) the SR MLP is the frozen expert bank and never
        # reads shared_intermediate_size, so this default is harmless there (it
        # ends up equal to the per-expert width, which nothing consumes).
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
