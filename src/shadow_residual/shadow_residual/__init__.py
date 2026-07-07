# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual architecture for Granite Switch (experimental).

Two parallel hidden-state streams (base + adapter) routed through a single
GQA call with doubled query heads.  See
``docs/SHADOW_RESIDUAL.md`` for an overview.

Public API:

- :class:`ShadowResidualAttention` — HF attention module.
- :class:`ShadowResidualDecoderLayer` — HF decoder layer.
- :class:`ShadowResidualModel`, :class:`ShadowResidualForCausalLM` —
  HF modeling classes (inherit from
  :class:`GraniteMoeHybridPreTrainedModel`).
- :func:`validate_shadow_residual_config`, :func:`set_shadow_residual` —
  config helpers.
- :func:`transfer_base_weights` — fused→unfused projection slicer used
  when constructing an SR model from an upstream HF Granite checkpoint.
"""

from .attention_hf import ShadowResidualAttention
from .config_helpers import set_shadow_residual, validate_shadow_residual_config
from .cross_stream import CrossStream
from .decoder_hf import ShadowResidualDecoderLayer
from .modeling_hf import (
    ShadowResidualForCausalLM,
    ShadowResidualModel,
    ShadowResidualPreTrainedModel,
)
from .weight_transfer import transfer_base_weights

__all__ = [
    "CrossStream",
    "ShadowResidualAttention",
    "ShadowResidualDecoderLayer",
    "ShadowResidualPreTrainedModel",
    "ShadowResidualModel",
    "ShadowResidualForCausalLM",
    "set_shadow_residual",
    "validate_shadow_residual_config",
    "transfer_base_weights",
]
