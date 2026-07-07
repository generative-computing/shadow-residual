# SPDX-License-Identifier: Apache-2.0
"""Marker subclass of :class:`torch.nn.Linear` used to opt projections into
shadow-residual stream gating.

PEFT chooses which module type to install at each target site via the
``custom_module_mapping`` registered on the :class:`peft.LoraConfig`.  By
constructing the SR decoder's ``q_proj`` / ``k_proj`` / ``v_proj`` /
``o_proj`` / ``gate_proj`` / ``up_proj`` / ``down_proj`` as
:class:`_StreamGatedLinear` instead of plain :class:`nn.Linear`, the
factory can dispatch them to :class:`ShadowResidualLora` (the
contextvar-aware wrapper) while leaving any other ``nn.Linear`` in the
model — for example, ones inside the open-source switch — bound to the
stock :class:`peft.tuners.lora.layer.Linear` wrapper.

Behaviourally identical to :class:`nn.Linear`; only the type tag differs.
"""

from __future__ import annotations

import torch.nn as nn


class _StreamGatedLinear(nn.Linear):
    """``nn.Linear`` marker for PEFT custom-module dispatch.

    Targeted by attribute name (``"q_proj"``, …) like any other LoRA
    target; the subclass tag controls which wrapper class PEFT installs.
    """


__all__ = ["_StreamGatedLinear"]
