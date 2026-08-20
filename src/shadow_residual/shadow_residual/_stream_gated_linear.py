# SPDX-License-Identifier: Apache-2.0
"""Marker subclass of :class:`torch.nn.Linear` for the SR decoder projections.

The SR decoder constructs its ``q_proj`` / ``k_proj`` / ``v_proj`` /
``o_proj`` / ``gate_proj`` / ``up_proj`` / ``down_proj`` as
:class:`_StreamGatedLinear`. It is behaviourally identical to
:class:`nn.Linear` — only the type tag differs. Both streams' projections
are wrapped by **stock** :class:`peft.tuners.lora.layer.Linear`; the
base-stream frozen invariant is enforced at the call site by
:func:`_base_only` (which calls ``base_layer(x)``, bypassing the delta),
not by a custom LoRA wrapper.

The tag is retained as a stable, greppable marker for the projections SR
treats as dual-stream, and to keep the door open for a future
name-independent dispatch. It carries no behaviour of its own.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _StreamGatedLinear(nn.Linear):
    """``nn.Linear`` marker for the SR decoder's dual-stream projections.

    Targeted by attribute name (``"q_proj"``, …) like any other LoRA
    target and wrapped by stock ``peft.tuners.lora.layer.Linear``. The
    subclass tag carries no behaviour; it just marks these projections.
    """


def _base_only(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call a projection's frozen base weight, bypassing any LoRA delta.

    A stock ``peft.tuners.lora.layer.Linear`` exposes the raw frozen linear as
    ``.base_layer``; a bare (unwrapped) ``nn.Linear`` is its own base. Calling
    ``base_layer(x)`` is a pure function of ``x`` with no adapter-state mutation —
    the property the checkpoint-pure base-stream gate relies on (see
    ``decoder_hf.ShadowResidualDecoderLayer.forward`` /
    ``attention_hf.ShadowResidualAttention.forward``).
    """
    base = getattr(module, "base_layer", module)
    return base(x)


__all__ = ["_StreamGatedLinear", "_base_only"]
