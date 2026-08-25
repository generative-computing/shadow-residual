# SPDX-License-Identifier: Apache-2.0
"""Serving/generation glue for loading a trained SR adapter.

Small helpers shared by the adapter-loading paths (``training.generate`` and
``eval.answerability_eval``) that build an SR base and attach a saved PEFT
adapter. Kept out of the model package (``shadow_residual.shadow_residual``),
which is pure model construction and must not depend on PEFT adapter-save
conventions.
"""

from __future__ import annotations

import json
import os


def read_share_moe_routing_from_adapter(adapter_path: str) -> bool:
    """Read the ``share_moe_routing`` mode an adapter was trained under.

    ``share_moe_routing`` is a forward-path choice, not a weight, so it is not
    encoded in ``adapter_model.safetensors``; ``train.py`` records it as an extra
    key in the saved ``adapter_config.json``. Serving must build the SR base with
    the SAME value the adapter was trained under, or the forward path silently
    diverges from training. Use this to source that value from the adapter itself
    instead of hand-passing it::

        from shadow_residual.shadow_residual.build import build_sr_base
        from shadow_residual.training.generation_utils import (
            read_share_moe_routing_from_adapter,
        )

        share = read_share_moe_routing_from_adapter(adapter_path)
        base = build_sr_base(base_id, torch_dtype=..., share_moe_routing=share)
        model = PeftModel.from_pretrained(base, adapter_path)

    Returns ``False`` when the key is absent (older adapters, or dense/non-MoE
    bases where the flag is a no-op) — the safe default that matches the model's
    own ``getattr(config, "share_moe_routing", False)`` fallback.
    """
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    try:
        with open(cfg_path) as f:
            return bool(json.load(f).get("share_moe_routing", False))
    except (FileNotFoundError, ValueError):
        return False


__all__ = ["read_share_moe_routing_from_adapter"]
