# SPDX-License-Identifier: Apache-2.0
"""Adapter-stream topologies (``config.adapter_stream``) and the LoRA sites each one runs.

``"full"`` is the dual transformer (adapter attention + MLP + cross_stream). Every
other value is ``"<carry>[+branch]"``:

* carry ``readout``  — ``h' = c + cross(hb_out) + branch``, ``c = readout_norm(h)``.
* carry ``layermix`` — ``h' = hb_out + (h - hb_in) + cross(hb_out) + branch``: the adapter
  state is the base state plus a running sum of per-layer taps, so with zero-init
  LoRA it equals the base model exactly. No per-layer adapter nonlinearity.

Branches: ``attn`` (adapter Q/O on the shared base K/V), ``mlp`` (adapter MLP),
``dense`` (``D(c)``), ``pool`` (low-rank map of the uniform causal mean of the base V —
a fixed average, no learned attention), ``heads`` (low-rank map of the base's
per-head attention outputs before o_proj). See docs/sr_readout_stream_design.html.

Pure Python (no torch) so the training-config schema can validate against it.
"""

from __future__ import annotations

from typing import FrozenSet, NamedTuple, Optional

ADAPTER_STREAMS = (
    "full",
    "readout",
    "readout+attn",
    "readout+mlp",
    "readout+dense",
    "readout+pool",
    "readout+heads",
    "layermix",
    "layermix+pool",
    "layermix+heads",
)

CROSS_STREAM = "cross_stream"
BRANCH_SITES = {
    "attn": frozenset({"q_proj", "o_proj"}),
    "mlp": frozenset({"gate_proj", "up_proj", "down_proj"}),
    "dense": frozenset({"readout_dense"}),
    "pool": frozenset({"readout_pool"}),
    "heads": frozenset({"readout_heads"}),
}
# Frozen zero-init sites that are no-ops unless LoRA wraps them.
ZERO_SITE_BRANCHES = {"dense": "readout_dense", "pool": "readout_pool", "heads": "readout_heads"}


class StreamSpec(NamedTuple):
    carry: str
    branches: FrozenSet[str]


def parse_adapter_stream(adapter_stream: str) -> Optional[StreamSpec]:
    """``None`` for ``"full"``; otherwise the carry and its branches."""
    if adapter_stream not in ADAPTER_STREAMS:
        raise ValueError(
            f"unknown adapter_stream {adapter_stream!r}; expected one of {ADAPTER_STREAMS}"
        )
    if adapter_stream == "full":
        return None
    carry, *branches = adapter_stream.split("+")
    return StreamSpec(carry, frozenset(branches))


def lora_sites(adapter_stream: str) -> tuple[frozenset, frozenset]:
    """(allowed, required) LoRA target names for a non-full stream."""
    spec = parse_adapter_stream(adapter_stream)
    if spec is None:
        raise ValueError('"full" has no read-out site restrictions')
    allowed = {CROSS_STREAM}.union(*(BRANCH_SITES[b] for b in spec.branches))
    required = {CROSS_STREAM} | {ZERO_SITE_BRANCHES[b] for b in spec.branches if b in ZERO_SITE_BRANCHES}
    return frozenset(allowed), frozenset(required)


def needs_readout_norm(adapter_stream: str) -> bool:
    spec = parse_adapter_stream(adapter_stream)
    return spec is not None and spec.carry == "readout"
