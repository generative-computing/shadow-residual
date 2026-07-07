# SPDX-License-Identifier: Apache-2.0
"""PEFT integration for the shadow-residual architecture.

Public surface is just two names:

* :class:`CrossStreamLora` — custom PEFT LoRA layer for the SR cross-stream
  site, registered with PEFT via ``LoraConfig._register_custom_module``.
* :func:`get_shadow_residual_peft_model` — factory analogous to
  :func:`peft.get_peft_model`.  Takes a stock :class:`peft.LoraConfig`.  If
  ``"cross_stream"`` is in ``target_modules``, builds the shadow-residual
  base model and wraps it with a PEFT ``LoraModel`` that has
  :class:`CrossStreamLora` registered.  Otherwise builds a plain Granite
  base and returns ``peft.get_peft_model(base, config)`` — bit-identical
  to vanilla PEFT LoRA, no SR overhead.

The user-facing config is therefore stock :class:`peft.LoraConfig`.  All of
PEFT's standard knobs (``r``, ``lora_alpha``, ``lora_dropout``,
``target_modules``, ``rank_pattern``, ``alpha_pattern``, ``use_rslora``,
``layers_to_transform``, ``modules_to_save``, …) work, including against
the cross-stream site.
"""

from .cross_stream_lora import CrossStreamLora
from .factory import get_shadow_residual_peft_model
from .load import load_shadow_residual_peft_model

__all__ = [
    "CrossStreamLora",
    "get_shadow_residual_peft_model",
    "load_shadow_residual_peft_model",
]
