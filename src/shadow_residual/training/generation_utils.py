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
    those live in ``target_modules``; for a non-lora one (``"linear"`` /
    ``"monarch"``) they are in ``modules_to_save`` (not LoRA targets). Deriving from
    the union covers both.
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
    return cross_stream_taps_from_target_modules(
        cfg.get("target_modules"), cfg.get("modules_to_save"),
    )


def read_cross_stream_tap_types_from_adapter(adapter_path: str) -> dict[str, dict]:
    """Recover each non-lora tap's layer TYPE and numeric parameter from the weights.

    The cross-stream type is STRUCTURAL: it decides which module the SR base builds
    at each tapped site (:class:`CrossStream` for ``"lora"``, a trainable full-H×H
    :class:`CrossStreamLinear` for ``"linear"``, a full-rank
    :class:`MonarchCrossStream` for ``"monarch"``). The base must be built the way it
    was trained BEFORE PEFT attaches, or ``PeftModel.from_pretrained`` has no
    matching module to bind the saved tensors to — and a wrong Monarch block count
    is not a graceful failure but an opaque 3-D shape mismatch deep inside PEFT.

        from shadow_residual.shadow_residual.build import build_sr_base
        from shadow_residual.training.generation_utils import (
            read_cross_stream_taps_from_adapter,
            read_cross_stream_tap_types_from_adapter,
        )

        base = build_sr_base(
            base_id,
            torch_dtype=...,
            cross_stream_taps=read_cross_stream_taps_from_adapter(adapter_path),
            cross_stream_tap_types=(
                read_cross_stream_tap_types_from_adapter(adapter_path)
            ),
        )
        model = PeftModel.from_pretrained(base, adapter_path)

    Reads the safetensors HEADER, not ``adapter_config.json``, on purpose — and
    unlike its siblings, which have no weights to read. Each non-lora type has an
    unmistakable saved signature: a 3-D ``…<tap>.f_in`` of shape ``(b, H/b, H/b)``
    is Monarch with ``b = shape[0]``; a 2-D ``…<tap>.proj.weight`` is linear. Two
    consequences follow. First, the answer cannot disagree with the weights it
    describes. Second, it works on a mid-training ``checkpoint-N/`` dir, where
    ``train.py``'s SR-extras patch has not run (it fires only at the final save) —
    that gap bit us during the wiring ablation's epoch-1 probe.

    Returns ``{}`` for a LoRA-only adapter (every tap is the frozen ``CrossStream``,
    the historical default) and for one saved before this existed. Taps absent from
    the returned map default to ``"lora"`` downstream, so a mixed topology only
    reports the taps that differ.

    Note the saved keys carry **no** ``modules_to_save.<adapter>.`` infix: PEFT's
    ``ModulesToSaveWrapper.adapter_state_dict`` strips it, so a key is just
    ``<prefix>.<tap name>.f_in`` / ``<prefix>.<tap name>.proj.weight``.
    """
    from shadow_residual.shadow_residual.cross_stream import CROSS_STREAM_TAPS

    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        return {}

    from safetensors import safe_open

    found: dict[str, dict] = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".f_in"):
                # `<prefix>.<tap>.f_in` — the tap name is the 2nd-to-last segment.
                tap_name, spec = key.rsplit(".", 2)[-2], {"type": "monarch"}
                spec["num"] = int(f.get_slice(key).get_shape()[0])
            elif key.endswith(".proj.weight"):
                # `<prefix>.<tap>.proj.weight`. The matrix is always H×H, so there
                # is no numeric parameter to recover (the config-side value was
                # provenance only and never shaped anything).
                tap_name, spec = key.rsplit(".", 3)[-3], {"type": "linear", "num": None}
            else:
                continue
            if tap_name in CROSS_STREAM_TAPS:
                found[tap_name] = spec
    # Registry order, matching every other tap-ordered surface in the codebase.
    return {n: found[n] for n in CROSS_STREAM_TAPS if n in found}


def read_cross_stream_type_from_adapter(
    adapter_path: str,
) -> tuple[str, Optional[int]]:
    """The scalar (type, num) of an adapter's DEFAULT ``cross_stream`` tap.

    Back-compat shim over :func:`read_cross_stream_tap_types_from_adapter` for
    callers and adapters that predate the per-tap map. Prefers the saved SHAPES;
    falls back to the ``cross_stream_type`` / ``cross_stream_dim`` extras that
    ``train.py`` writes into ``adapter_config.json`` when there are no weights to
    read. Returns ``("lora", None)`` when neither source says otherwise — an older
    adapter is always the frozen ``"lora"`` site.

    Prefer the per-tap reader in new code: this one cannot express a mixed-type
    topology, and it says nothing about a non-default tap.
    """
    from shadow_residual.shadow_residual.cross_stream import (
        DEFAULT_CROSS_STREAM_TAPS,
    )

    default_tap = DEFAULT_CROSS_STREAM_TAPS[0]
    spec = read_cross_stream_tap_types_from_adapter(adapter_path).get(default_tap)
    if spec is not None:
        return str(spec["type"]), spec.get("num")

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, ValueError):
        return "lora", None
    cs_type = cfg.get("cross_stream_type", "lora")
    # The config-side value may be the per-tap dict form; narrow it to this tap.
    if isinstance(cs_type, dict):
        cs_type = cs_type.get(default_tap, "lora")
    cs_dim = cfg.get("cross_stream_dim")
    return str(cs_type), (int(cs_dim) if cs_dim is not None else None)


__all__ = [
    "read_share_moe_routing_from_adapter",
    "read_cross_stream_taps_from_adapter",
    "read_cross_stream_tap_types_from_adapter",
    "read_cross_stream_type_from_adapter",
]
