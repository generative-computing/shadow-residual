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

Selectable cross-stream type — ORTHOGONAL to the wiring
-------------------------------------------------------
Two axes compose, and they are independent. **Which** sites exist is the tap
registry (:data:`CROSS_STREAM_TAPS`) — the *wiring*, carried by the tap's module
NAME. **What kind of module** sits at each tapped site is the cross-stream
*type*, chosen in the training config (``adapter.cross_stream_type``, scalar or
per-tap) and threaded onto the SR config so the decoder builds the matching
module at every tap (see :data:`CROSS_STREAM_TAP_TYPES` and ``decoder_hf``):

* ``"lora"`` (default) — :class:`CrossStream`, the frozen H×H site above; stock
  PEFT LoRA supplies the trainable rank-R delta. ``target_modules[tap]`` is the
  LoRA rank ``r``.
* ``"linear"`` — :class:`CrossStreamLinear`, a *directly trainable* single full
  ``H×H`` matrix (one matmul, full-rank — NOT a low-rank bottleneck; that is what
  ``"lora"`` already gives). ``target_modules[tap]`` is provenance only.
* ``"monarch"`` — :class:`MonarchCrossStream`, a two-factor block-diagonal
  butterfly: **full rank** at ``H·(b + H/b)`` params, where ``target_modules[tap]``
  is the block count ``b``.

``"linear"`` and ``"monarch"`` are not LoRA targets (a 3-D block factor cannot be
one, and ``_register_custom_module`` is off-limits — see CLAUDE.md); PEFT persists
them via ``modules_to_save``. Any type is available at ANY tap, and one model may
mix types across taps: the type of a given tap is decided by which PEFT list its
name lands in, so the 4-row registry needs no per-type rows.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

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


def _cross_stream_names_in(
    selection: "str | Iterable[str] | None", field: str
) -> list[str]:
    """The registered tap names in one PEFT module selection, validated.

    ``"all-linear"`` (the only str form SR could see for ``target_modules``) is
    rejected upstream in ``training.factory._reject_kv_lora``; nothing to derive
    from a str either way.

    Raises:
        ValueError: on any ``cross_stream``-prefixed name that is not a
            registered tap. PEFT would eventually fail with "Target modules not
            found", but a typo deserves a legible error that lists the valid
            names.
    """
    if selection is None or isinstance(selection, str):
        return []
    names = list(selection)
    unknown = [
        n for n in names if n.startswith("cross_stream") and not is_cross_stream_tap(n)
    ]
    if unknown:
        raise ValueError(
            f"Unknown cross-stream tap(s) in {field}: {unknown}. "
            f"Valid tap names: {sorted(CROSS_STREAM_TAPS)}."
        )
    return [n for n in names if is_cross_stream_tap(n)]


def cross_stream_taps_from_target_modules(
    target_modules: "str | Iterable[str] | None",
    modules_to_save: "str | Iterable[str] | None" = None,
) -> tuple[str, ...]:
    """Derive the tap set to BUILD from PEFT's two module selections.

    The tap set is the UNION of the ``cross_stream*`` entries in both lists, and
    which list a name lands in is also what fixes its layer TYPE:
    ``target_modules`` → :class:`CrossStream` + stock ``lora.Linear``;
    ``modules_to_save`` → :class:`CrossStreamLinear` or
    :class:`MonarchCrossStream` + ``ModulesToSaveWrapper``. A tap named in
    neither list is a frozen no-op, so building one would be pure dead weight.

    Deriving the set here (rather than carrying a second config field) means the
    two can never drift, and a saved ``adapter_config.json`` records both lists —
    so wiring *and* per-tap layer type are already everything the serving path
    needs to rebuild the same module tree.

    Returns the names in :data:`CROSS_STREAM_TAPS` order (deterministic module
    insertion order — the FSDP meta-init path in ``training.factory`` requires
    every rank to traverse an identically-ordered module tree).

    Falls back to :data:`DEFAULT_CROSS_STREAM_TAPS` when *neither* list names a
    tap, so a non-SR target set still yields the same structure it does today (a
    single unwrapped ``cross_stream`` site → the model's bare single-stream path).

    Raises:
        ValueError: on an unregistered ``cross_stream*`` name in either list.
        ValueError: on a tap named in BOTH lists. LoRA-wrapping and
            ``ModulesToSaveWrapper``-wrapping the same module is not a defined
            combination — a tap has exactly one layer type, not two.
    """
    targeted = _cross_stream_names_in(target_modules, "target_modules")
    saved = _cross_stream_names_in(modules_to_save, "modules_to_save")

    both = [n for n in CROSS_STREAM_TAPS if n in targeted and n in saved]
    if both:
        raise ValueError(
            f"Cross-stream tap(s) named in BOTH target_modules and "
            f"modules_to_save: {both}. A tap has exactly one layer type — list "
            f"it in target_modules for LoRA on a frozen-zero CrossStream, or in "
            f"modules_to_save for a directly-trainable linear / Monarch layer, "
            f"not both."
        )

    named = set(targeted) | set(saved)
    selected = tuple(n for n in CROSS_STREAM_TAPS if n in named)
    return selected or DEFAULT_CROSS_STREAM_TAPS


class CrossStreamTap:
    """Marker mixin for every cross-stream site type.

    A plain mixin, not an ``nn.Module`` base: :class:`CrossStream` subclasses
    ``nn.Linear`` (so stock PEFT dispatches it to ``lora.Linear``) while the other
    types subclass ``nn.Module``, so there is no common module supertype to share.
    Same pattern as ``_stream_gated_linear._StreamGatedLinear``.

    It exists so the three chokepoints that must treat "a BARE tap of any type"
    uniformly need ONE ``isinstance`` test, which also covers types added later:

    * ``modeling_hf._any_cross_stream_wrapped`` — invariant 3: a bare tap of any
      type is an exact no-op, so it must NOT force the dual-stream path.
    * ``modeling_hf._init_weights`` — ``post_init`` would otherwise randomize a
      tap's ``nn.Linear``, or skip a raw ``nn.Parameter`` entirely.
    * ``training.factory._materialize_and_transfer`` — ``to_empty()`` leaves
      garbage, and neither ``transfer_base_weights`` nor PEFT's LoRA reset
      touches a tap.

    Contract: :meth:`reset_parameters` restores the site's initial NUMERICS and
    nothing else. It must never touch ``requires_grad`` — ``to_empty()`` preserves
    it, and one uniform walk has to fix both halves of a ``ModulesToSaveWrapper``
    (the frozen ``original_module`` and the trainable copy) without collapsing the
    distinction. Frozen-ness is each class's ``__init__``'s job.
    """

    def reset_parameters(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class CrossStream(CrossStreamTap, nn.Linear):
    """Frozen, zero-initialised ``nn.Linear(H, H)`` cross-stream site.

    A standard square linear that starts as an exact no-op (all-zero weight,
    frozen) so PEFT can wrap it with a stock LoRA layer. Subclasses
    :class:`nn.Linear` so it is a first-class LoRA target — no custom
    ``weight`` property or device-anchor buffer needed.

    :class:`CrossStreamTap` comes FIRST in the bases so its zeroing
    :meth:`reset_parameters` wins the MRO over ``nn.Linear``'s Kaiming one —
    including the call ``nn.Linear.__init__`` makes itself.

    Attributes:
        in_features / out_features: hidden size ``H`` (cross-stream is square).
    """

    def __init__(self, hidden_size: int):
        super().__init__(hidden_size, hidden_size, bias=False)
        # Zero-init + freeze: base contribution is exactly zero and never
        # trained. A stock LoRA wrapper's delta is then the sole output.
        # (reset_parameters already zeroed the weight via nn.Linear.__init__;
        # this is belt-and-braces and costs nothing.)
        nn.init.zeros_(self.weight)
        self.weight.requires_grad_(False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Zero the weight — numerics only, per the :class:`CrossStreamTap` contract.

        Overrides ``nn.Linear.reset_parameters``' Kaiming init, so a plain
        construction, ``post_init``, and the factory's post-``to_empty()`` walk all
        land on the same exact no-op. Deliberately does NOT re-freeze: see the
        mixin's contract.
        """
        nn.init.zeros_(self.weight)

    def forward(self, h_base: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Return ``W_cross(h_base)`` — zero on a bare (unwrapped) model.

        Extra positional / keyword args are accepted and ignored so the
        wrapper-base call signature stays consistent (e.g. when PEFT strips
        variant kwargs before calling the base layer, or when callers thread
        future per-token kwargs through the layer).
        """
        del args, kwargs
        return super().forward(h_base)


class CrossStreamLinear(CrossStreamTap, nn.Module):
    """Trainable full ``H×H`` matrix cross-stream site — ONE matmul.

    Unlike :class:`CrossStream` (a frozen H×H no-op that stock LoRA makes
    trainable via a low-rank delta), this module *is itself* the trainable
    cross-stream: a single dense ``nn.Linear(H, H, bias=False)`` applied directly
    to ``h_base``. Full-rank — NOT a bottleneck / low-rank factorization (that is
    what the ``"lora"`` type already provides). It is not a LoRA target; PEFT
    persists it via ``modules_to_save`` (each built tap name — a stock mechanism,
    no custom LoRA class), which wraps it, marks it trainable, and round-trips its
    weight through ``save_pretrained`` / ``from_pretrained``.

    The ``dim`` argument (the config's ``cross_stream: d`` value) does NOT shape
    the matrix: the cross-stream maps ``H→H``, so a ``d×d`` matrix only fits when
    ``d == H``. It is accepted for a uniform construction signature with
    :class:`CrossStreamLinear`-style types and recorded for provenance, but the
    weight is always ``H×H``. (See ``decoder_hf`` / ``build.build_sr_config``.)

    The weight is zero-initialised (mirroring LoRA's zero-init ``B``) so the
    injection ``W(h_base)`` is exactly zero at step 0 — the adapter stream starts
    base-identical and the contribution grows only as training moves the weight.
    """

    def __init__(self, hidden_size: int, dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        # Recorded for provenance only; does not shape the (always H×H) matrix.
        self.dim = dim
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Zero-init: the injection starts at zero. Trainable (requires_grad=True).
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Zero the matrix — numerics only, per the :class:`CrossStreamTap` contract.

        Deliberately does NOT touch ``requires_grad``: this module stays trainable,
        and the same call has to re-initialise the frozen ``original_module`` twin
        that ``ModulesToSaveWrapper`` keeps without unfreezing it.
        """
        nn.init.zeros_(self.proj.weight)

    def forward(self, h_base: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Return ``W(h_base)`` — a single full H×H matmul, zero at init.

        Extra positional / keyword args are accepted and ignored to keep the
        call signature identical to :class:`CrossStream` (the decoder calls the
        cross-stream site the same way regardless of type).
        """
        del args, kwargs
        return self.proj(h_base)


class MonarchCrossStream(CrossStreamTap, nn.Module):
    """Full-rank structured-sparse cross-stream site: ``y = f_out(P(f_in(x)))``.

    An alternative REALIZATION of any registered tap (see
    :data:`CROSS_STREAM_TAPS`) — the wiring is the tap name, the type is this
    class. Two-factor Monarch / butterfly factorization (Dao et al., ICML 2022)
    with ``b = num_blocks`` and ``p = H / b``::

        x (..., H) -> view (..., b, p)
                      einsum('bqp,...bp->...bq', f_in)    # f_in : (b, p, p)
                      transpose the (b, q) grid -> (q, b) # the permutation P
                      einsum('qsb,...qb->...qs', f_out)   # f_out: (p, b, b)
                   -> view (..., H)

    Both factors are block-diagonal with SQUARE blocks, so ``f_in`` has ``b``
    blocks of ``p×p`` and ``f_out`` has ``p`` blocks of ``b×b``: the parameter
    count is ``H·(b + H/b)`` (see :func:`monarch_param_count`), *not* ``2H²/b``
    — the two coincide only at ``b = √H``.

    Why this instead of LoRA on a :class:`CrossStream`: LoRA costs ``2rH`` and is
    hard-capped at rank ``r``. This form costs ``H·(b + H/b)`` and is **full
    rank** (a product of invertible block-diagonal matrices and permutations),
    with a **full receptive field for any ``b``** — the grid transpose makes every
    output row read all ``b`` groups. It constrains the sparsity *pattern* rather
    than the rank, which is a different hypothesis about what the cross-stream
    injection needs, and it is not reachable by any LoRA rank.

    Trained via PEFT ``modules_to_save`` (a switch, not a sum) because a 3-D
    block-diagonal factor is not an ``nn.Linear`` and cannot be a LoRA target,
    and because ``_register_custom_module`` is off-limits (see CLAUDE.md).
    ``f_out`` is zero-init so the site is an exact no-op at step 0 and on the
    frozen ``original_module`` twin ``ModulesToSaveWrapper`` keeps — which is what
    preserves the bare-model / ``disable_adapter()`` invariant.
    """

    def __init__(self, hidden_size: int, num_blocks: int):
        super().__init__()
        if num_blocks <= 0 or hidden_size % num_blocks:
            raise ValueError(
                f"Monarch num_blocks must be a positive divisor of hidden_size; "
                f"got num_blocks={num_blocks}, hidden_size={hidden_size}."
            )
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.f_in = nn.Parameter(torch.empty(num_blocks, self.block_size, self.block_size))
        self.f_out = nn.Parameter(torch.empty(self.block_size, num_blocks, num_blocks))
        self.reset_parameters()
        # Frozen by default, exactly like CrossStream: on a bare model the tap
        # must contribute nothing and train nothing. PEFT's ModulesToSaveWrapper
        # unfreezes its own trainable deepcopy.
        self.requires_grad_(False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """PER-BLOCK Kaiming on ``f_in``, zeros on ``f_out`` (the ``lora_B`` analogue).

        The fan-in of a block-diagonal factor is the BLOCK width, not the whole
        slice. ``nn.init.kaiming_uniform_`` on a 3-D tensor infers
        ``fan_in = shape[1] * shape[2] = p²`` from the Conv1d convention, but each
        output coordinate of ``f_in`` sums over only ``p`` inputs — so the stock
        call gives a bound ``sqrt(p)``× too small (8× at ``H=2560, b=40``: 0.0156
        vs 0.125). That is not cosmetic here: a Monarch tap has no ``α/r`` knob, so
        ``f_in``'s variance IS the effective initial scale of the whole site. The
        per-block form matches the Monarch adapter reference implementation
        (SprocketLab/sparse_matrix_fine_tuning, ICML 2024).

        Does NOT touch ``requires_grad``, per the :class:`CrossStreamTap` contract.
        """
        gain = nn.init.calculate_gain("leaky_relu", math.sqrt(5))
        bound = math.sqrt(3.0) * gain / math.sqrt(self.block_size)
        self.f_in.uniform_(-bound, bound)
        nn.init.zeros_(self.f_out)

    def forward(self, h_base: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Return ``W_cross(h_base)`` — zero while ``f_out`` is zero.

        Extra args are accepted and ignored for the same reason as
        :meth:`CrossStream.forward`.
        """
        del args, kwargs
        lead = h_base.shape[:-1]
        x = h_base.reshape(*lead, self.num_blocks, self.block_size)
        x = torch.einsum("bqp,...bp->...bq", self.f_in, x)
        x = x.transpose(-1, -2)  # (..., b, p) -> (..., p, b): the permutation P
        x = torch.einsum("qsb,...qb->...qs", self.f_out, x)
        return x.reshape(*lead, self.hidden_size)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_blocks={self.num_blocks}, "
            f"block_size={self.block_size}, "
            f"params={monarch_param_count(self.hidden_size, self.num_blocks)}"
        )


# ---------------------------------------------------------------------------
# Monarch sizing helpers
# ---------------------------------------------------------------------------


def monarch_param_count(hidden_size: int, num_blocks: int) -> int:
    """Trainable params of one :class:`MonarchCrossStream`: ``H·(b + H/b)``.

    ``f_in`` is ``b`` blocks of ``p×p`` and ``f_out`` is ``p`` blocks of ``b×b``
    (``p = H/b``), giving ``b·p² + p·b² = H·(p + b)``. Exposed so configs and
    docs can state parameter parity with a LoRA rank (``2·r·H``) without
    re-deriving it — e.g. ``H=2560, b=40`` → ``266,240`` == LoRA ``r=52``.
    """
    if num_blocks <= 0 or hidden_size % num_blocks:
        raise ValueError(
            f"Monarch num_blocks must be a positive divisor of hidden_size; "
            f"got num_blocks={num_blocks}, hidden_size={hidden_size}."
        )
    return hidden_size * (num_blocks + hidden_size // num_blocks)


def monarch_blocks_for_hidden_size(hidden_size: int) -> int:
    """The divisor of ``H`` nearest ``√H`` — the minimum-parameter Monarch.

    ``H·(b + H/b)`` is minimized at ``b = √H`` (where it is ``2H^1.5``) and grows
    in both directions, so the best ``b`` is the divisor closest to ``√H``. For
    ``H=2560`` the bracketing divisors are 40 and 64, both giving ``2560·104 =
    266,240``; ties resolve to the smaller ``b``. This value is a FLOOR: no ``b``
    gets a Monarch tap cheaper.
    """
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive; got {hidden_size}.")
    root = math.sqrt(hidden_size)
    divisors = [b for b in range(1, hidden_size + 1) if hidden_size % b == 0]
    return min(divisors, key=lambda b: (abs(b - root), b))


# ---------------------------------------------------------------------------
# Type dispatch — the ONE place that maps a type name to a module
# ---------------------------------------------------------------------------
#: ``type name -> (hidden_size, num) -> tap module``. ``num`` is the tap's numeric
#: parameter from ``target_modules[tap]``, whose meaning is type-dependent: the
#: LoRA rank ``r`` for ``"lora"`` (consumed by PEFT, not by the module),
#: provenance-only ``dim`` for ``"linear"``, and the block count ``b`` for
#: ``"monarch"``. Adding a 4th type is one row here plus a
#: :class:`CrossStreamTap` subclass — no other file changes.
CROSS_STREAM_TAP_TYPES: "dict[str, Callable[[int, Optional[int]], CrossStreamTap]]" = {
    "lora": lambda hidden_size, num: CrossStream(hidden_size),
    "linear": lambda hidden_size, num: CrossStreamLinear(hidden_size, num),
    "monarch": lambda hidden_size, num: MonarchCrossStream(
        hidden_size,
        num if num is not None else monarch_blocks_for_hidden_size(hidden_size),
    ),
}


def build_cross_stream_tap(
    name: str,
    hidden_size: int,
    tap_type: str = "lora",
    num: Optional[int] = None,
) -> CrossStreamTap:
    """Construct the cross-stream module for one tap.

    Args:
        name: the registered tap name — the WIRING (see :data:`CROSS_STREAM_TAPS`).
            Used only for error messages; the wiring is resolved by the decoder.
        hidden_size: ``H``.
        tap_type: a key of :data:`CROSS_STREAM_TAP_TYPES` — the TYPE axis, which is
            orthogonal to the wiring.
        num: the tap's numeric parameter (``target_modules[tap]``), interpreted per
            type. ``None`` for ``"monarch"`` falls back to the minimum-parameter
            block count for ``H``.

    Raises:
        ValueError: on an unregistered type name.
    """
    factory = CROSS_STREAM_TAP_TYPES.get(tap_type)
    if factory is None:
        raise ValueError(
            f"Unknown cross-stream type {tap_type!r} for tap {name!r}. "
            f"Valid types: {sorted(CROSS_STREAM_TAP_TYPES)}."
        )
    return factory(hidden_size, num)


def normalize_cross_stream_tap_types(
    tap_names: "Iterable[str]",
    tap_types: "str | Mapping[str, Any] | None" = None,
    nums: "Mapping[str, int] | int | None" = None,
) -> dict[str, dict[str, Any]]:
    """Resolve the per-tap ``{"type", "num"}`` spec for every built tap.

    The single normalizer shared by ``build.build_sr_config``, the decoder's
    legacy-attribute read, and ``training.factory`` — so the scalar and per-tap
    forms of ``adapter.cross_stream_type`` can never be interpreted two ways.

    Args:
        tap_names: the taps actually being built (registry order).
        tap_types: ``None`` → every tap is ``"lora"``; a str → that type for every
            tap; a mapping ``{tap -> type}`` or ``{tap -> {"type", "num"}}`` →
            per-tap, with taps absent from the mapping defaulting to ``"lora"``.
        nums: the taps' numeric parameters — a mapping ``{tap -> num}`` (from
            ``target_modules``), or a single int applied to every non-lora tap (the
            legacy scalar ``cross_stream_dim``). An explicit ``"num"`` inside
            ``tap_types`` wins.

    Returns:
        ``{tap name -> {"type": str, "num": int | None}}`` for exactly ``tap_names``.
    """
    if isinstance(tap_types, str):
        by_tap: Mapping[str, Any] = {name: tap_types for name in tap_names}
    else:
        by_tap = tap_types or {}

    resolved: dict[str, dict[str, Any]] = {}
    for name in tap_names:
        spec = by_tap.get(name)
        if isinstance(spec, Mapping):
            tap_type = spec.get("type") or "lora"
            num = spec.get("num")
        else:
            tap_type = spec or "lora"
            num = None
        if num is None:
            if isinstance(nums, Mapping):
                num = nums.get(name)
            elif nums is not None and tap_type != "lora":
                # The legacy scalar cross_stream_dim: one number, and it only ever
                # described the non-lora tap.
                num = nums
        resolved[name] = {"type": tap_type, "num": num}
    return resolved


def validate_cross_stream_tap_types(
    hidden_size: int,
    tap_types: "Mapping[str, Any] | None",
    tap_names: "Iterable[str] | None" = None,
) -> None:
    """Raise on a malformed per-tap type spec; warn on a wasteful Monarch ``b``.

    Checks every key is a registered tap, every type name is known, and — when
    ``tap_names`` is given — that a spec exists only for taps actually BUILT (a
    spec for an unbuilt tap is a silent typo, since nothing would consume it).
    For ``"monarch"``, ``b`` must be a positive divisor of ``hidden_size``.

    Raises:
        ValueError: on any of the above.
    """
    built = set(tap_names) if tap_names is not None else None
    for name, spec in (tap_types or {}).items():
        if not is_cross_stream_tap(name):
            raise ValueError(
                f"Unknown cross-stream tap in cross_stream_tap_types: {name!r}. "
                f"Valid tap names: {sorted(CROSS_STREAM_TAPS)}."
            )
        if built is not None and name not in built:
            raise ValueError(
                f"cross_stream_tap_types names tap {name!r}, which is not built "
                f"on this model (built taps: {sorted(built)}). A type for an "
                f"unbuilt tap has no effect — it is almost certainly a typo."
            )
        tap_type = spec.get("type") if isinstance(spec, Mapping) else spec
        tap_type = tap_type or "lora"
        if tap_type not in CROSS_STREAM_TAP_TYPES:
            raise ValueError(
                f"Unknown cross-stream type {tap_type!r} for tap {name!r}. "
                f"Valid types: {sorted(CROSS_STREAM_TAP_TYPES)}."
            )
        if tap_type != "monarch":
            continue
        num = spec.get("num") if isinstance(spec, Mapping) else None
        if num is None:
            continue
        if not isinstance(num, int) or num <= 0:
            raise ValueError(
                f"Monarch num_blocks for {name!r} must be a positive int; "
                f"got {num!r}."
            )
        if hidden_size % num:
            raise ValueError(
                f"Monarch num_blocks for {name!r} must divide hidden_size "
                f"({hidden_size}); got {num}. Nearest usable divisor: "
                f"{monarch_blocks_for_hidden_size(hidden_size)}."
            )
        # H·(b + H/b) is minimized at b ≈ √H; more than 2× that floor means the
        # block count is far off the optimum and the tap is paying for it.
        floor = monarch_param_count(
            hidden_size, monarch_blocks_for_hidden_size(hidden_size)
        )
        params = monarch_param_count(hidden_size, num)
        if params > 2 * floor:
            logger.warning(
                "Monarch tap %r with b=%d costs %d params/layer, over 2x the "
                "%d floor at b=%d. Intentional?",
                name,
                num,
                params,
                floor,
                monarch_blocks_for_hidden_size(hidden_size),
            )


__all__ = [
    "CrossStream",
    "CrossStreamLinear",
    "CrossStreamTap",
    "CROSS_STREAM_POINT_ORDER",
    "CROSS_STREAM_TAP_TYPES",
    "CROSS_STREAM_TAPS",
    "DEFAULT_CROSS_STREAM_TAPS",
    "MonarchCrossStream",
    "build_cross_stream_tap",
    "cross_stream_taps_from_target_modules",
    "cross_stream_taps_need_base_ahead",
    "is_cross_stream_tap",
    "monarch_blocks_for_hidden_size",
    "monarch_param_count",
    "normalize_cross_stream_tap_types",
    "validate_cross_stream_tap_types",
]
