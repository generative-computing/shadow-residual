# SPDX-License-Identifier: Apache-2.0
"""The cross-stream "sites" — targetable nn.Modules on each SR decoder layer.

Shadow residual injects information from the base stream into the adapter
stream inside each decoder layer:

    h_adapt += W_cross(h_base)            # W_cross is a low-rank linear

This module reifies each cross-stream site as a real ``nn.Linear(H, H)`` on the
decoder so that PEFT's ``LoraModel`` can find it via ``target_modules`` (e.g.
``["cross_stream"]``) and wrap it with a standard
:class:`peft.tuners.lora.layer.Linear` — no custom module registration needed.

Which sites exist is config-driven — see :data:`CROSS_STREAM_TAPS` below.

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
accepted cost of the stock-PEFT path, and it is paid **per tap**. On a bare
(never-wrapped) model it makes the cross-stream contribution exactly zero, so
the decoder's single-stream fallback is unaffected.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Tap registry
# ---------------------------------------------------------------------------
# A "tap" is one cross-stream site: it reads the base stream at some point in
# the layer and adds its (LoRA) output into the adapter stream at some point.
# The registry maps the site's MODULE NAME — which is also the PEFT
# ``target_modules`` entry and therefore baked into every saved
# ``adapter_model.safetensors`` — to ``(base source point, adapter destination
# point)``.
#
# Naming convention: same-point taps are ``cross_stream_<point>``, with the
# post-MLP one keeping the historical bare name ``cross_stream`` so existing
# YAMLs and checkpoints keep working. Cross-position wirings (destination at a
# different point than the source) get the explicit
# ``cross_stream_<src>_to_<dst>`` form.
#
# Adding a wiring = one row here. No new injection site is needed: the decoder
# injects at every point, resolving taps by destination.
#
# Append new rows — never insert. ``cross_stream_taps_from_target_modules``
# returns names in this dict's order, and that order drives module insertion
# order in the decoder, which the FSDP meta-init path requires to be identical
# on every rank. Appending also keeps existing topologies' module order
# byte-identical.
#
# Which wirings are meaningful: ``adapter_pre_attn[i]`` IS ``adapter_post_mlp[i-1]``
# (the same tensor — the stacked hidden state passes through untouched), and
# likewise ``base_pre_attn[i] == base_post_mlp[i-1]``. So a tap with a
# ``pre_attn`` source and a ``pre_attn`` destination is just ``cross_stream``
# with its weight re-indexed by one layer — the same model family, not a new
# wiring. The two cross-position rows below are not reducible that way:
# ``pre_attn -> post_mlp`` feeds a one-layer-stale base state into the SAME
# destination the default tap uses, and ``post_mlp -> pre_attn`` is resolved
# WITHIN one layer (see ``decoder_hf``), making it the only zero-lag wiring.
CROSS_STREAM_TAPS: dict[str, tuple[str, str]] = {
    "cross_stream": ("post_mlp", "post_mlp"),
    "cross_stream_post_attn": ("post_attn", "post_attn"),
    "cross_stream_pre_attn_to_post_mlp": ("pre_attn", "post_mlp"),
    "cross_stream_post_mlp_to_pre_attn": ("post_mlp", "pre_attn"),
}

#: The historical single-tap topology — what a config that names no tap gets.
DEFAULT_CROSS_STREAM_TAPS: tuple[str, ...] = ("cross_stream",)

#: The tap-able positions within a decoder layer, in execution order.
CROSS_STREAM_POINT_ORDER: dict[str, int] = {
    "pre_attn": 0,
    "post_attn": 1,
    "post_mlp": 2,
}


def is_cross_stream_tap(name: str) -> bool:
    """Is ``name`` a registered cross-stream tap module name?"""
    return name in CROSS_STREAM_TAPS


def cross_stream_taps_need_base_ahead(names: "Iterable[str]") -> bool:
    """Does this tap set require the base sublayer to run AHEAD of the adapter?

    True iff some tap's destination *precedes* its source. The decoder's default
    forward interleaves the two streams sublayer by sublayer, so at the moment
    the adapter reaches such a destination the base stream has not yet reached
    the source point. Serving that wiring requires running the base stream's
    whole layer first — legal and non-circular (``h_base`` never depends on
    ``h_adapt``), but a different execution order, so the decoder keeps it as a
    separate path selected once at construction time.
    """
    return any(
        CROSS_STREAM_POINT_ORDER[dst] < CROSS_STREAM_POINT_ORDER[src]
        for src, dst in (CROSS_STREAM_TAPS[name] for name in names)
    )


def cross_stream_taps_from_target_modules(
    target_modules: "str | Iterable[str] | None",
) -> tuple[str, ...]:
    """Derive the tap set to BUILD from a PEFT ``target_modules`` selection.

    The tap set *is* the set of ``cross_stream*`` LoRA targets: an untargeted
    tap is a frozen zero-init no-op, so building one would be pure dead weight.
    Deriving the set here (rather than carrying a second config field) means the
    two can never drift, and a saved ``adapter_config.json`` already records
    everything the serving path needs to rebuild the same module tree.

    Returns the names in :data:`CROSS_STREAM_TAPS` order (deterministic module
    insertion order — the FSDP meta-init path in ``training.factory`` requires
    every rank to traverse an identically-ordered module tree).

    Falls back to :data:`DEFAULT_CROSS_STREAM_TAPS` when no tap is named, so a
    non-SR target set still yields the same structure it does today (a single
    unwrapped ``cross_stream`` site → the model's bare single-stream path).

    Raises:
        ValueError: on any ``cross_stream``-prefixed name that is not a
            registered tap. PEFT would eventually fail with "Target modules not
            found", but a typo deserves a legible error that lists the valid
            names.
    """
    if target_modules is None or isinstance(target_modules, str):
        # ``"all-linear"`` (the only str form SR could see) is rejected upstream
        # in ``training.factory._reject_kv_lora``; nothing to derive either way.
        return DEFAULT_CROSS_STREAM_TAPS

    names = list(target_modules)
    unknown = [
        n for n in names if n.startswith("cross_stream") and not is_cross_stream_tap(n)
    ]
    if unknown:
        raise ValueError(
            f"Unknown cross-stream tap(s) in target_modules: {unknown}. "
            f"Valid tap names: {sorted(CROSS_STREAM_TAPS)}."
        )

    selected = tuple(n for n in CROSS_STREAM_TAPS if n in names)
    return selected or DEFAULT_CROSS_STREAM_TAPS


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


__all__ = [
    "CrossStream",
    "CROSS_STREAM_POINT_ORDER",
    "CROSS_STREAM_TAPS",
    "DEFAULT_CROSS_STREAM_TAPS",
    "cross_stream_taps_from_target_modules",
    "cross_stream_taps_need_base_ahead",
    "is_cross_stream_tap",
]
