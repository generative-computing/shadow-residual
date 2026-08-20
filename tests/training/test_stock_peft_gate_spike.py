# SPDX-License-Identifier: Apache-2.0
"""Reproducible spike for the stock-PEFT SR base-stream gate.

Pins the go/no-go finding that shapes the stock-PEFT design so it can't silently
regress under a peft/torch bump:

- Q1: the `disable_adapter()` base-stream gate survives activation checkpointing
  (both reentrant modes). This is the mechanism that replaces the custom
  contextvar + scaling=0.0 stream gate.

(The old Q2/Q2b aLoRA-offset-under-checkpointing findings were removed with the
aLoRA gating — the adapter is now plain LoRA and always active.)

CPU-only; fast.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model

HID, SEQ, BATCH = 16, 8, 2


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HID, HID, bias=False)

    def forward(self, x):
        return self.proj(x)


def _wrap():
    return get_peft_model(_Tiny(), LoraConfig(r=4, lora_alpha=8, target_modules=["proj"]))


def _run_checkpointed(fn, x, use_reentrant):
    x = x.clone().requires_grad_(True)
    out = checkpoint(fn, x, use_reentrant=use_reentrant)
    out.sum().backward()


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_q1_disable_adapter_survives_checkpoint(use_reentrant):
    """Base + adapter stream in one checkpointed step; base via disable_adapter."""
    pm = _wrap()
    x = torch.randn(BATCH, SEQ, HID)

    def step(inp):
        a = pm(inp)
        with pm.disable_adapter():
            b = pm(inp)
        return a + b

    _run_checkpointed(step, x, use_reentrant)  # must not raise
