# SPDX-License-Identifier: Apache-2.0
"""Re-condition a frozen-A bank on calibration inputs, keeping its row space.

With ``C`` the input covariance of a site group and ``s² = tr(C)/(3·in)`` (the feature
variance of a stock kaiming_uniform(a=√5) lora_A on those inputs), three modes:

* ``var``: rescale each row so every feature ``a·x`` has variance ``s²`` (keeps each row's direction).
* ``white``: ``A ← s·(A C Aᵀ)^{-1/2}·A``, features decorrelated with variance ``s²`` (same row space).
* ``energy``: one scalar per group so ``tr(A C Aᵀ) = r·s²`` (keeps the relative spread).

cross_stream is left unchanged unless ``--include-cross``.

    python -m shadow_residual.training.rescale_frozen_a --base-model <path> --data a.jsonl ... \
        --bank <dir> [<dir> ...] --modes var white energy --out-root <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from shadow_residual.training.compute_frozen_a import (
    _batches,
    _calibration_rows,
    _kaiming_scale,
    collect,
)
from shadow_residual.training.frozen_a import A_FILE, GROUP_SITES, MANIFEST_FILE, load_frozen_a

logger = logging.getLogger(__name__)

MODES = ("var", "white", "energy")
_REL_FLOOR = 1e-10
_EIG_FLOOR = 1e-14  # fp64: whitening is well defined up to cond(A C Aᵀ) ≈ 1e14


def _bank_layers(bank: dict[str, torch.Tensor]) -> set[int]:
    return {int(k.split(".")[1]) for k in bank}


def _recondition(a: torch.Tensor, c: torch.Tensor, mode: str, where: str) -> tuple[torch.Tensor, dict]:
    target = _kaiming_scale(c) ** 2
    gram = a @ c @ a.T
    var = gram.diagonal()
    if not bool((var > _REL_FLOOR * var.max()).all()):
        raise ValueError(f"{where}: a feature has (near-)zero variance on the calibration data")
    if mode == "var":
        out = a * (target / var).sqrt().unsqueeze(1)
    elif mode == "energy":
        out = a * (a.shape[0] * target / var.sum()).sqrt()
    elif mode == "white":
        evals, evecs = torch.linalg.eigh(gram)
        if not bool((evals > _EIG_FLOOR * evals.max()).all()):
            raise ValueError(f"{where}: A C Aᵀ is (near-)singular; the rows are linearly dependent on the data")
        out = (evecs @ torch.diag(evals.rsqrt()) @ evecs.T) @ a * target ** 0.5
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    new_var = (out @ c @ out.T).diagonal()
    return out, {
        "var_spread_before": float(var.max() / var.min()),
        "var_spread_after": float(new_var.max() / new_var.min()),
        "energy_ratio_vs_kaiming_before": float(var.sum() / (a.shape[0] * target)),
        "energy_ratio_vs_kaiming_after": float(new_var.sum() / (a.shape[0] * target)),
    }


@torch.no_grad()
def rescale_bank(bank: dict[str, torch.Tensor], collector, mode: str = "var", include_cross: bool = False):
    """Return ``(re-conditioned tensors, per-group stats)``; gate/up keep sharing one A."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    n_layers = len(collector.layers)
    if _bank_layers(bank) != set(range(n_layers)):
        raise ValueError(f"bank layers {sorted(_bank_layers(bank))} != model layers 0..{n_layers - 1}")
    out = {k: v.clone() for k, v in bank.items()}
    stats: dict[str, dict] = {}
    for i in range(n_layers):
        if not torch.equal(bank[f"layers.{i}.gate_proj"], bank[f"layers.{i}.up_proj"]):
            raise ValueError(f"layers.{i}: gate_proj and up_proj A differ; the bank must share one A")
        for group, sites in GROUP_SITES.items():
            if group == "cross" and not include_cross:
                continue
            c = collector.covariance(i, group)
            a = bank[f"layers.{i}.{sites[0]}"].to(device=c.device, dtype=torch.float64)
            new, st = _recondition(a, c, mode, f"layers.{i}.{group}")
            new = new.float().cpu().contiguous()
            for site in sites:
                out[f"layers.{i}.{site}"] = new.clone()
            stats[f"layers.{i}.{group}"] = st
    return out, stats


def _summary(stats: dict[str, dict]) -> dict:
    out = {}
    for g in GROUP_SITES:
        vals = [v for k, v in stats.items() if k.endswith(f".{g}")]
        if not vals:
            continue
        before = sorted(v["var_spread_before"] for v in vals)
        out[g] = {
            "spread_before_median": before[len(before) // 2], "spread_before_max": before[-1],
            "spread_after_max": max(v["var_spread_after"] for v in vals),
            "energy_before_median": sorted(v["energy_ratio_vs_kaiming_before"] for v in vals)[len(vals) // 2],
            "energy_after_median": sorted(v["energy_ratio_vs_kaiming_after"] for v in vals)[len(vals) // 2],
        }
    return out


def out_name(bank_dir: Path, mode: str) -> str:
    base = bank_dir.name.removesuffix("_v1")
    return f"{base}_{mode}_v1"


def plan_outputs(bank_dirs: list[Path], modes: list[str], only: list[str] | None) -> list[tuple[Path, str]]:
    """(bank, mode) pairs to write; every ``--only`` entry must match one pair."""
    pairs = [(b, m) for b in bank_dirs for m in modes]
    if only is None:
        return pairs
    wanted = set(only)
    known = {f"{b.name}:{m}" for b, m in pairs}
    if wanted - known:
        raise ValueError(f"--only entries match no --bank/--modes pair: {sorted(wanted - known)}")
    return [(b, m) for b, m in pairs if f"{b.name}:{m}" in wanted]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--bank", required=True, type=Path, nargs="+")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--data", required=True, nargs="+")
    p.add_argument("--rows-per-file", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-cross", action="store_true")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--only", nargs="*", default=None, help="bank:mode pairs to write (default: all)")
    args = p.parse_args(argv)
    plan = plan_outputs(args.bank, args.modes, args.only)

    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from shadow_residual.shadow_residual.build import build_sr_base

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = build_sr_base(args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    rows = _calibration_rows(args.data, args.rows_per_file, args.seed)
    collector = collect(model, _batches(tok, rows, args.max_tokens, args.batch_size, device))
    tokens = collector.count[(0, "q")]
    logger.info("calibrated on %d rows, %d tokens", len(rows), tokens)

    failed = []
    for bank_dir, mode in plan:
        bank = load_frozen_a(bank_dir)
        src = json.loads((bank_dir / MANIFEST_FILE).read_text())
        try:
            tensors, stats = rescale_bank(bank, collector, mode, args.include_cross)
        except ValueError as e:
            print(f"RESCALE_FAILED {bank_dir.name}:{mode} Error: {e}", flush=True)
            failed.append(f"{bank_dir.name}:{mode}")
            continue
        out_dir = args.out_root / out_name(bank_dir, mode)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_dir / A_FILE))
        summary = _summary(stats)
        manifest = {
            "recipe": {"directions": f"row space of {bank_dir}", "mode": mode,
                       "target_feature_var": "tr(C)/(3 in) (stock kaiming)",
                       "cross_stream": "re-conditioned" if args.include_cross else "unchanged"},
            "rank": src.get("rank"), "feature_scale": f"rescaled: {mode} (target tr(C)/(3 in))",
            "source_manifest": src, "base_model": args.base_model,
            "data": args.data, "rows": len(rows), "rows_per_file": args.rows_per_file,
            "tokens": tokens, "max_tokens": args.max_tokens, "seed": args.seed,
            "summary": summary, "per_group": stats,
        }
        (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        print(f"RESCALE_SUMMARY {out_dir.name} " + json.dumps(summary), flush=True)
    print(f"ALL_RESCALED written={len(plan) - len(failed)} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
