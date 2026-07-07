# SPDX-License-Identifier: Apache-2.0
"""The cross-stream "site" — a targetable nn.Module on each SR decoder layer.

Shadow residual injects information from the base stream into the adapter
stream at the end of each decoder layer:

    h_adapt += W_cross(h_base)            # W_cross is a low-rank linear

Originally the rank-R weights ``A`` and ``B`` lived as raw ``nn.Parameter``
tensors on the decoder layer and the matmul was inlined in ``forward``.  That
made them invisible to PEFT.  This module reifies the cross-stream site as a
real ``nn.Module`` named ``cross_stream`` on the decoder so that PEFT's
``LoraModel`` can find it via ``target_modules=["cross_stream"]`` and replace
it with a custom :class:`CrossStreamLora` layer (registered via
``LoraConfig._register_custom_module``).

The base ``CrossStream`` module owns no parameters — it is a no-op identity
("returns zero" because cross-stream is purely additive).  All learnable
weight lives on the LoRA wrapper PEFT installs, so the rank flows from
``LoraConfig.r`` or ``LoraConfig.rank_pattern["cross_stream"]`` exactly like
any other PEFT LoRA target.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossStream(nn.Module):
    """No-op cross-stream site.

    Acts as a target for PEFT to attach a :class:`CrossStreamLora` adapter
    onto.  When no adapter is installed (e.g. inference of a non-SR-trained
    model, or eval before training), ``forward`` returns a zero tensor with
    the same shape as the input — a no-op contribution to ``h_adapt``.

    The wrapper PEFT installs (see :class:`CrossStreamLora`) replaces this
    module's behaviour with a low-rank ``B @ A @ x`` projection during
    training and inference.

    Attributes:
        in_features: hidden size of the input ``h_base`` (``H``).
        out_features: same as ``in_features`` — cross-stream is square.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.in_features = hidden_size
        self.out_features = hidden_size
        # PEFT's ``LoraModel._replace_module`` reads ``child.weight`` (or
        # falls back to ``next(child.parameters())``) to learn what device
        # the LoRA adapter weights should live on.  CrossStream is
        # parameter-free in principle, but we expose a single-element
        # buffer-as-parameter so PEFT can discover the device.  The tensor
        # is never read by anything else.
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        """Device-anchor exposed under the conventional ``weight`` name.

        ``LoraModel._replace_module`` checks ``hasattr(child, "weight")``
        before falling back to ``next(child.parameters())``; returning the
        device anchor avoids a ``StopIteration``.
        """
        return self._device_anchor

    def forward(self, h_base: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: D401
        """Return a zero contribution of the right shape and dtype.

        PEFT's LoRA layer wraps this module and *adds* its low-rank delta to
        the base layer's output.  Since the base contribution is zero, the
        sum equals the LoRA delta — which is exactly the cross-stream term.

        Extra positional / keyword arguments are accepted and ignored so the
        wrapper-base call signature stays consistent (e.g. when peft strips
        ``alora_offsets`` before calling us, or when callers thread future
        per-token kwargs through the layer).
        """
        del args, kwargs
        return torch.zeros_like(h_base)


__all__ = ["CrossStream"]
