# SPDX-License-Identifier: Apache-2.0
"""Compute a data-free frozen-A bank from the base weights alone (no calibration data).

``--which minor|principal``: r right singular vectors of W (``frozen_a.weight_svd_a``).
``--which prune|fold|sketch``: W's rows compressed into r rows (``frozen_a.compressed_rows_a``).
gate_proj and up_proj share one A from the stacked ``[W_gate; W_up]``. cross_stream has a
zero base weight, so its A is copied from an existing bank (``--cross-from``, e.g. the
random kaiming bank).

    python -m shadow_residual.training.compute_weight_a --base-model <id> \
        --cross-from <bank dir> --rank 32 --which minor --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from shadow_residual.training.frozen_a import (
    A_FILE,
    COMPRESS_METHODS,
    MANIFEST_FILE,
    compressed_rows_a,
    load_frozen_a,
    row_space_stats,
    weight_svd_a,
)

logger = logging.getLogger(__name__)

SVD_METHODS = ("minor", "principal")


def site_seed(seed: int, layer: int, group: int) -> int:
    """Distinct per (bank seed, layer, group) for up to 10,000 layers × 10 groups."""
    return seed * 100_000 + layer * 10 + group


@torch.no_grad()
def compute_weight_bank(sr_model, rank: int, which: str, cross: dict[str, torch.Tensor], seed: int = 0):
    """One A per (layer, group); returns ``(tensors keyed layers.{l}.{site}, row-space stats)``."""
    tensors: dict[str, torch.Tensor] = {}
    stats: dict[str, dict] = {}
    for i, layer in enumerate(sr_model.model.layers):
        groups = {
            "q": ([layer.self_attn.q_proj.weight], ("q_proj",)),
            "o": ([layer.self_attn.o_proj.weight], ("o_proj",)),
            "mlp_in": ([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], ("gate_proj", "up_proj")),
            "down": ([layer.mlp.down_proj.weight], ("down_proj",)),
        }
        for g, (group, (weights, sites)) in enumerate(groups.items()):
            w = torch.cat(weights, dim=0)
            if which in SVD_METHODS:
                a, _ = weight_svd_a(w, rank, which)
                st = row_space_stats(a, w)
            else:
                a, st = compressed_rows_a(w, rank, which, seed=site_seed(seed, i, g))
            a = a.float().cpu()
            for site in sites:
                tensors[f"layers.{i}.{site}"] = a.clone()
            stats[f"layers.{i}.{group}"] = st
        key = f"layers.{i}.cross_stream"
        tensors[key] = cross[key].float().clone()
    return tensors, stats


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--cross-from", required=True, type=Path, help="bank dir to copy cross_stream A from")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--which", choices=SVD_METHODS + COMPRESS_METHODS, default="minor")
    p.add_argument("--seed", type=int, default=0, help="base seed for fold/sketch (per-site offset added)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    from safetensors.torch import save_file

    from shadow_residual.shadow_residual.build import build_sr_base

    model = build_sr_base(args.base_model, torch_dtype=torch.bfloat16).eval()
    tensors, stats = compute_weight_bank(model, args.rank, args.which, load_frozen_a(args.cross_from), seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / A_FILE))
    summary = {}
    for g in ("q", "o", "mlp_in", "down"):
        vals = [v for k, v in stats.items() if k.endswith(f".{g}")]
        summary[g] = {
            "weight_energy_mean": sum(v["energy"] for v in vals) / len(vals),
            "weight_energy_max": max(v["energy"] for v in vals),
            "rank_min": min(v["rank"] for v in vals),
            "cond_mean": sum(v["cond"] for v in vals) / len(vals),
            "cond_max": max(v["cond"] for v in vals),
        }
    what = (f"{args.which} right singular vectors" if args.which in SVD_METHODS
            else f"{args.which}-compressed rows")
    manifest = {
        "recipe": {"q/o/down": f"{what} of W (data-free)",
                   "gate+up": f"shared {what} of [W_g;W_u]",
                   "cross_stream": f"copied from {args.cross_from}"},
        "rank": args.rank, "which": args.which,
        "seed": args.seed if args.which in ("fold", "sketch") else None,
        "feature_scale": "kaiming row norm (1/sqrt3)",
        "weight_energy": "fraction of ||W||_F^2 inside the row space of A",
        "base_model": args.base_model, "summary": summary, "per_group": stats,
    }
    (args.out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    print("WEIGHT_A_SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
