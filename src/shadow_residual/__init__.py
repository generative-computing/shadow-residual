# SPDX-License-Identifier: Apache-2.0
"""Shadow Residual — trainable, distributable adapters for Granite models.

Shadow Residual (SR) runs two parallel hidden-state streams through every
decoder layer: a frozen *base stream* (bit-exact with the unadapted base
model) and a trainable *adapter stream* carrying LoRA deltas plus a per-layer
low-rank cross-stream injection from the base. Only the adapter stream reaches
the LM head.

Subpackages:

- :mod:`shadow_residual.peft_shadow_residual` — PEFT-based SR factory
  (:func:`~shadow_residual.peft_shadow_residual.get_shadow_residual_peft_model`)
  and loader
  (:func:`~shadow_residual.peft_shadow_residual.load_shadow_residual_peft_model`).
  Canonical entry points for training and serving.
- :mod:`shadow_residual.shadow_residual` — the SR HF model
  (:class:`ShadowResidualForCausalLM`) the PEFT factory builds on.
- :mod:`shadow_residual.training` — the unified YAML-driven training driver
  (HF ``Trainer`` + collators + adapter export).
- :mod:`shadow_residual.config` — the unified training-config schema
  (:class:`TrainingConfig`, :func:`load_training_config`).
"""

__version__ = "0.1.0-dev"

from .config import TrainingConfig, load_training_config

__all__ = ["TrainingConfig", "load_training_config", "__version__"]
