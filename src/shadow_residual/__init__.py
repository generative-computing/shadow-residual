# SPDX-License-Identifier: Apache-2.0
"""Shadow Residual — trainable, distributable adapters for Granite models.

Shadow Residual (SR) runs two parallel hidden-state streams through every
decoder layer: a frozen *base stream* (bit-exact with the unadapted base
model) and a trainable *adapter stream* carrying LoRA deltas plus a per-layer
low-rank cross-stream injection from the base. Only the adapter stream reaches
the LM head.

Subpackages:

- :mod:`shadow_residual.shadow_residual` — the SR HF model
  (:class:`ShadowResidualForCausalLM`). Runs standalone as a causal LM and is
  adaptable with 100% stock ``peft`` (plain ``LoraConfig`` + ``get_peft_model``).
  For serving, build a base with
  :func:`~shadow_residual.shadow_residual.build.build_sr_base` and attach a
  saved adapter with stock :func:`peft.PeftModel.from_pretrained`.
- :mod:`shadow_residual.training` — the unified YAML-driven training driver
  (HF ``Trainer`` + collators + adapter export). Includes
  :func:`~shadow_residual.training.factory.get_shadow_residual_peft_model`, the
  FSDP-aware training factory that meta-inits + PEFT-wraps the SR model.
- :mod:`shadow_residual.config` — the unified training-config schema
  (:class:`TrainingConfig`, :func:`load_training_config`).
"""

__version__ = "0.1.0-dev"

from .config import TrainingConfig, load_training_config

__all__ = ["TrainingConfig", "load_training_config", "__version__"]
