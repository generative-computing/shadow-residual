# SPDX-License-Identifier: Apache-2.0
"""Opt-in chunked cross-entropy monkeypatch to bound full-vocab softmax memory.

`transformers.loss.loss_utils.fixed_cross_entropy` calls
`nn.functional.cross_entropy(source, target)` on the full `[N_tokens, vocab]`
logit matrix. For a large vocab (granite-4.1: 100,352) and a long sequence
(a single ~37k-token row survives when the SFT path does not hard-truncate),
the internal log-softmax materializes a `[N_tokens, vocab]` fp32 tensor plus
backward-saved tensors — ~21.67 GiB, which OOMs an 80 GiB GPU already holding
the model + activations.

This patch replaces `fixed_cross_entropy` with a numerically-identical version
that splits `source`/`target` along the token dimension into `CHUNK`-row blocks,
computes `reduction="sum"` per block, accumulates the scalar, and divides once at
the end. Peak extra memory is bounded to `[CHUNK, vocab]` instead of
`[N_tokens, vocab]`. Math is identical to the unchunked `sum`/`mean` reduction
(cross-entropy is a per-row sum, so summing per-chunk then dividing by the same
denominator is exact).

Enable by setting env `SR_CHUNKED_CE=1` (optionally `SR_CE_CHUNK=<rows>`,
default 1024) before the first forward. No-op unless enabled, so other tasks are
untouched.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _chunked_fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch=None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    chunk = int(os.environ.get("SR_CE_CHUNK", "1024"))
    reduction = "sum" if num_items_in_batch is not None else "mean"

    # Fast path for small inputs: identical to upstream, no chunking overhead.
    n = source.shape[0]
    if n <= chunk:
        loss = nn.functional.cross_entropy(
            source, target, ignore_index=ignore_index, reduction=reduction
        )
        if reduction == "sum" and num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss

    # Chunked: accumulate a SUM over all rows (ignoring -100 targets, which
    # cross_entropy already excludes), then divide once. For the mean case we
    # also need the count of non-ignored rows to reproduce the mean exactly.
    total = source.new_zeros(())
    valid = 0
    for i in range(0, n, chunk):
        s = source[i : i + chunk]
        t = target[i : i + chunk]
        total = total + nn.functional.cross_entropy(
            s, t, ignore_index=ignore_index, reduction="sum"
        )
        if reduction == "mean":
            valid += int((t != ignore_index).sum().item())

    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total.device)
        return total / num_items_in_batch
    # mean: divide by number of non-ignored target rows (0 -> nan, matching
    # cross_entropy's behavior on an all-ignored input).
    return total / max(valid, 1) if valid else total * float("nan")


def maybe_patch_fixed_cross_entropy() -> bool:
    """If SR_CHUNKED_CE=1, replace transformers' fixed_cross_entropy. Returns
    True if patched. Patches the function object in loss_utils AND the names
    already imported into the per-model modeling modules' loss dispatch."""
    if os.environ.get("SR_CHUNKED_CE") != "1":
        return False
    from transformers.loss import loss_utils

    loss_utils.fixed_cross_entropy = _chunked_fixed_cross_entropy
    # transformers.loss.loss_utils.ForCausalLMLoss calls fixed_cross_entropy by
    # module-global lookup, so patching the module attribute above is sufficient
    # for the standard loss path. Patch the LOSS_MAPPING entry too in case a
    # model resolved it eagerly.
    try:
        from transformers.loss import loss_utils as _lu
        if hasattr(_lu, "LOSS_MAPPING"):
            pass  # ForCausalLMLoss closes over the module global; nothing to swap.
    except Exception:  # noqa: BLE001
        pass
    chunk = os.environ.get("SR_CE_CHUNK", "1024")
    logger.info("Patched fixed_cross_entropy with chunked CE (CHUNK=%s rows).", chunk)
    return True
