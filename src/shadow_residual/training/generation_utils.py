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
from typing import Optional


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


def read_cross_stream_taps_from_adapter(adapter_path: str) -> tuple[str, ...]:
    """Read which cross-stream sites an adapter was trained with.

    The tap set is not a separate saved field — it *is* the set of
    ``cross_stream*`` entries the adapter references. For a ``"lora"`` cross-stream
    those live in ``target_modules``; for a ``"linear"`` cross-stream they are in
    ``modules_to_save`` (not LoRA targets). Deriving from the union covers both.
    Building the SR base with a different tap set leaves the saved tensors with no
    module to attach to, so source it from the adapter::

        from shadow_residual.shadow_residual.build import build_sr_base
        from shadow_residual.training.generation_utils import (
            read_cross_stream_taps_from_adapter,
        )

        taps = read_cross_stream_taps_from_adapter(adapter_path)
        base = build_sr_base(base_id, torch_dtype=..., cross_stream_taps=taps)
        model = PeftModel.from_pretrained(base, adapter_path)

    Returns the historical single ``("cross_stream",)`` topology when the config
    is missing or names no tap — which is what an SR base builds by default, so
    older adapters keep loading unchanged.
    """
    from shadow_residual.shadow_residual.cross_stream import (
        DEFAULT_CROSS_STREAM_TAPS,
        cross_stream_taps_from_target_modules,
    )

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, ValueError):
        return DEFAULT_CROSS_STREAM_TAPS
    sources = list(cfg.get("target_modules") or [])
    sources += list(cfg.get("modules_to_save") or [])
    return cross_stream_taps_from_target_modules(sources)


def read_cross_stream_type_from_adapter(
    adapter_path: str,
) -> tuple[str, Optional[int]]:
    """Read the cross-stream (w-cross) layer type (+ provenance dim) of an adapter.

    The cross-stream type is STRUCTURAL: it decides which module the SR base
    builds at the cross_stream site (``CrossStream`` for ``"lora"``, a trainable
    full-H×H ``CrossStreamLinear`` for ``"linear"``; ``"linear"`` is single-tap-only,
    mutually exclusive with the multi-tap registry). The dim is provenance only —
    it does not shape the H×H matrix. The base must be built with the same type
    the adapter was trained under BEFORE PEFT attaches, or
    ``PeftModel.from_pretrained`` can't bind the saved cross-stream weights.
    ``train.py`` records both as extra keys in ``adapter_config.json``; source
    them from the adapter itself rather than hand-passing::

        from shadow_residual.shadow_residual.build import build_sr_base
        from shadow_residual.training.generation_utils import (
            read_cross_stream_taps_from_adapter,
            read_cross_stream_type_from_adapter,
        )

        taps = read_cross_stream_taps_from_adapter(adapter_path)
        cs_type, cs_dim = read_cross_stream_type_from_adapter(adapter_path)
        base = build_sr_base(
            base_id, torch_dtype=..., cross_stream_taps=taps,
            cross_stream_type=cs_type, cross_stream_dim=cs_dim,
        )
        model = PeftModel.from_pretrained(base, adapter_path)

    Returns ``("lora", None)`` when the keys are absent (older adapters, which
    predate the selectable type and are always the frozen ``"lora"`` site).
    """
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, ValueError):
        return "lora", None
    cs_type = str(cfg.get("cross_stream_type", "lora"))
    cs_dim = cfg.get("cross_stream_dim")
    return cs_type, (int(cs_dim) if cs_dim is not None else None)


__all__ = [
    "read_share_moe_routing_from_adapter",
    "read_cross_stream_taps_from_adapter",
    "read_cross_stream_type_from_adapter",
]
