# SPDX-License-Identifier: Apache-2.0
"""Per-call stream context for shadow-residual LoRA gating.

The dual-stream forward inside :class:`ShadowResidualDecoderLayer` calls
the same projection (``q_proj``, ``gate_proj``, …) twice — once for the
base stream, once for the adapter stream. PEFT wraps each projection with
a single :class:`ShadowResidualLora`, so both calls reach the same
wrapper. We need that wrapper to apply the LoRA delta on the adapter
call but skip it on the base call, restoring the frozen-base invariant.

The decoder sets a context tag immediately before each call:

- ``"base"`` — wrapper returns ``base_layer(x)`` only (delta gated off).
- ``"adapter"`` — wrapper applies the delta normally.

Outside any context (e.g. ``k_proj`` / ``v_proj`` calls in the SR
attention, where the K/V LoRA SHOULD apply and contaminate the shared
KV cache as a documented exception), the default tag is ``"adapter"`` so
the delta still fires.

Single rule: ``"base"`` is the only tag that gates the delta off.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator

# Default = "adapter": K/V calls (made outside any explicit context) still
# fire their LoRA delta, preserving the documented K/V contamination path.
_stream_tag: contextvars.ContextVar[str] = contextvars.ContextVar(
    "shadow_residual_stream_tag", default="adapter",
)


@contextlib.contextmanager
def stream_context(tag: str) -> Iterator[None]:
    """Set the stream tag for the duration of the with-block."""
    token = _stream_tag.set(tag)
    try:
        yield
    finally:
        _stream_tag.reset(token)


def current_stream() -> str:
    """Read the current stream tag."""
    return _stream_tag.get()


# --- aLoRA-offset recovery across the gradient-checkpoint recompute boundary ---
#
# PEFT injects ``alora_offsets`` onto each LoRA layer via a per-layer forward-pre-
# hook that is only active during the ORIGINAL forward. Gradient-checkpoint
# recompute re-runs the layer in backward, outside that hook, so the variant sees
# ``alora_offsets=None`` on recompute and would zero the gated delta — corrupting
# gradients (training divergence). The variants therefore cache the offsets on the
# original forward and reuse them when ``None`` arrives. But ``None`` can ALSO
# legitimately mean "no invocation in this batch → zero delta", and unit/inference
# calls must keep that semantics. This flag scopes the cache-reuse to the SR
# checkpointed decoder loop ONLY: the model sets it around the loop, and the
# variants consult their cache solely while it is set. Outside it (direct calls,
# inference), ``None`` keeps its plain "zero delta" meaning.
_offset_recovery: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "shadow_residual_offset_recovery", default=False,
)


@contextlib.contextmanager
def offset_recovery_enabled() -> Iterator[None]:
    """Enable aLoRA-offset cache-reuse for the duration of the with-block."""
    token = _offset_recovery.set(True)
    try:
        yield
    finally:
        _offset_recovery.reset(token)


def offset_recovery_active() -> bool:
    """True iff we are inside the SR checkpointed decoder loop."""
    return _offset_recovery.get()


__all__ = [
    "stream_context",
    "current_stream",
    "offset_recovery_enabled",
    "offset_recovery_active",
]
