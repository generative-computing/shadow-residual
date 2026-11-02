# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual architecture (HF backend).

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

Importing this package registers ``model_type="shadow_residual"`` with the HF
auto classes, so a converted SR checkpoint loads with
``AutoModelForCausalLM.from_pretrained``.
"""

from .attention_hf import ShadowResidualAttention
from .config_helpers import set_shadow_residual, validate_shadow_residual_config
from .cross_stream import CrossStream
from .decoder_hf import ShadowResidualDecoderLayer, ShadowResidualMLP
from .modeling_hf import (
    ShadowResidualForCausalLM,
    ShadowResidualModel,
    ShadowResidualPreTrainedModel,
)
from .model_config import ShadowResidualConfig
from .weight_transfer import transfer_base_weights

# Register the SR model type with the HF auto classes so a saved SR checkpoint
# (see ``convert.py``) loads with stock ``AutoConfig`` / ``AutoModelForCausalLM``
# ``.from_pretrained`` once this package is imported — no ``trust_remote_code``.
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register(ShadowResidualConfig.model_type, ShadowResidualConfig, exist_ok=True)
AutoModel.register(ShadowResidualConfig, ShadowResidualModel, exist_ok=True)
AutoModelForCausalLM.register(ShadowResidualConfig, ShadowResidualForCausalLM, exist_ok=True)

__all__ = [
    "CrossStream",
    "ShadowResidualAttention",
    "ShadowResidualDecoderLayer",
    "ShadowResidualMLP",
    "ShadowResidualPreTrainedModel",
    "ShadowResidualModel",
    "ShadowResidualForCausalLM",
    "ShadowResidualConfig",
    "set_shadow_residual",
    "validate_shadow_residual_config",
    "transfer_base_weights",
]
