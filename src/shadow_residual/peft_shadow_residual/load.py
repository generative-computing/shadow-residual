# SPDX-License-Identifier: Apache-2.0
"""Standalone loader: ``load_shadow_residual_peft_model(base_id, ckpt)``.

Unconditionally builds an SR base and registers
:class:`CrossStreamLora`, then calls :func:`peft.PeftModel.from_pretrained`.

The PEFT save format (``adapter_config.json`` + ``adapter_model.safetensors``)
records whether ``"cross_stream"`` was in ``target_modules``. Either way
the architecture is the same — the SR model's runtime gate (see
``modeling_hf.py``) takes the single-stream early-exit path automatically
when no :class:`CrossStream` site is wrapped, so checkpoints that did not
target ``cross_stream`` pay no dual-stream overhead at inference time.

ALORA-style mid-sequence activation is fully transparent: when the saved
:class:`peft.LoraConfig` carries ``alora_invocation_tokens``, stock peft
attaches ``ALoraLinearVariant`` to every wrapped projection automatically;
:class:`CrossStreamLora` consumes the same ``alora_offsets`` via the
pre-forward hook that peft installs on every :class:`LoraLayer`.

Returns a :class:`peft.PeftModel`.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from peft import LoraConfig, PeftModel

from .factory import (
    CROSS_STREAM_NAME,
    _build_sr_base,
    _disable_merge_and_unload,
    _register_cross_stream,
)

logger = logging.getLogger(__name__)


def load_shadow_residual_peft_model(
    base_model_name_or_path: str,
    checkpoint_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    adapter_name: str = "default",
    attn_implementation: Optional[str] = None,
    shared_base_kv: bool = False,
):
    """Load a PEFT adapter saved by :func:`get_shadow_residual_peft_model`.

    Args:
        base_model_name_or_path: HF id or local path of the base Granite
            model.
        checkpoint_path: directory containing ``adapter_config.json`` +
            ``adapter_model.safetensors``.
        torch_dtype: dtype for the base model construction.
        adapter_name: PEFT adapter slot name to load into.
        attn_implementation: passed through to the upstream HF base load.
        shared_base_kv: must match the value the checkpoint was trained
            with. The flag is not serialized into ``adapter_config.json``
            (it is an architecture choice, not a LoRA hyperparameter), so
            the caller has to supply it.

    Returns:
        :class:`peft.PeftModel`.
    """
    lora_config = LoraConfig.from_pretrained(checkpoint_path)
    targets = lora_config.target_modules or []
    has_cross_stream = (
        CROSS_STREAM_NAME in targets
        if not isinstance(targets, str)
        else CROSS_STREAM_NAME in targets
    )
    logger.info(
        "Loading PEFT checkpoint from %s; cross_stream %s in target_modules. "
        "Building SR base unconditionally.",
        checkpoint_path,
        "is" if has_cross_stream else "is not",
    )
    if not has_cross_stream:
        # LoRA/aLoRA checkpoint loaded onto the SR base. The SR model's
        # runtime gate (modeling_hf._any_cross_stream_wrapped) takes the
        # single-stream early-exit path, which is semantically equivalent
        # to upstream Granite + stock PEFT — but not bit-identical,
        # because SR uses unfused per-projection weights while upstream
        # Granite fuses QKV / gate-up. The reduction order differs;
        # logits drift at the last few decimals. For greedy decoding on
        # discrete classification tasks this is essentially never
        # observable, but flag it so cross-path numerical comparisons
        # don't surprise anyone.
        logger.warning(
            "Loaded LoRA/aLoRA checkpoint onto SR base; outputs are "
            "equivalent but not bit-exact with stock "
            "AutoModelForCausalLM + PeftModel.from_pretrained, due to "
            "SR's unfused QKV / gate-up projections."
        )

    base_model = _build_sr_base(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        shared_base_kv=shared_base_kv,
    )
    _register_cross_stream(lora_config)

    # Pass the registered config explicitly so peft does not re-read
    # adapter_config.json and lose our SR custom-module dispatch.
    peft_model = PeftModel.from_pretrained(
        base_model, checkpoint_path, adapter_name=adapter_name, config=lora_config,
    )
    _disable_merge_and_unload(peft_model)
    return peft_model


__all__ = ["load_shadow_residual_peft_model"]
