# SPDX-License-Identifier: Apache-2.0
"""The cross-stream "site" — a targetable nn.Module on each SR decoder layer.

Shadow residual injects information from the base stream into the adapter
stream at the end of each decoder layer:

    h_adapt += W_cross(h_base)            # W_cross is a low-rank linear

This module reifies the cross-stream site as a real ``nn.Linear(H, H)`` named
``cross_stream`` on the decoder so that PEFT's ``LoraModel`` can find it via
``target_modules=["cross_stream"]`` and wrap it with a standard
:class:`peft.tuners.lora.layer.Linear` — no custom module registration needed.

Why a real (frozen, zero-init) linear rather than a parameter-free no-op
---------------------------------------------------------------------------
The base weight is initialised to **zeros** and frozen (``requires_grad=False``),
so ``base_layer(x) = 0`` and the layer output of a stock LoRA wrapper is pure
``B(A(x)) * scaling`` — the rank-R base→adapter injection with no base
contribution. This is expressible with 100% stock PEFT and needs no custom LoRA
class: stock ``lora.Linear`` always adds its delta onto ``base_layer(x)``, which
here is zero. The delta is always active (plain LoRA) — the cross-stream
injection fires on every position, like the other projections.

The frozen ``H×H`` matrix is dead weight (never trained, never nonzero) — the
accepted cost of the stock-PEFT path. On a bare (never-wrapped) model it makes
the cross-stream contribution exactly zero, so the decoder's single-stream
fallback is unaffected.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossStream(nn.Linear):
    """Frozen, zero-initialised ``nn.Linear(H, H)`` cross-stream site.

    A standard square linear that starts as an exact no-op (all-zero weight,
    frozen) so PEFT can wrap it with a stock LoRA layer. Subclasses
    :class:`nn.Linear` so it is a first-class LoRA target — no custom
    ``weight`` property or device-anchor buffer needed.

    Attributes:
        in_features / out_features: hidden size ``H`` (cross-stream is square).
    """

    def __init__(self, hidden_size: int):
        super().__init__(hidden_size, hidden_size, bias=False)
        # Zero-init + freeze: base contribution is exactly zero and never
        # trained. A stock LoRA wrapper's delta is then the sole output.
        nn.init.zeros_(self.weight)
        self.weight.requires_grad_(False)

    def forward(self, h_base: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Return ``W_cross(h_base)`` — zero on a bare (unwrapped) model.

        Extra positional / keyword args are accepted and ignored so the
        wrapper-base call signature stays consistent (e.g. when PEFT strips
        variant kwargs before calling the base layer, or when callers thread
        future per-token kwargs through the layer).
        """
        del args, kwargs
        return super().forward(h_base)


__all__ = ["CrossStream"]
