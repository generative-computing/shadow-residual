# SPDX-License-Identifier: Apache-2.0
"""Precompute the frozen LoRA A matrices of SR-FrozenA from the base model.

Runs the bare SR model (bit-identical to the base, and to the adapter stream at
step 0) on calibration rows, accumulates the fp64 input covariance of every LoRA
site, and derives one A per (layer, input group) with :mod:`frozen_a`'s recipes.

    python -m shadow_residual.training.compute_frozen_a --base-model <path> \
        --data a/train.jsonl b/train.jsonl ... --rows-per-file 512 --rank 32 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import torch

from shadow_residual.training.frozen_a import (
    A_FILE,
    GROUP_SITES,
    MANIFEST_FILE,
    pca_a,
    whitened_svd_a,
)

logger = logging.getLogger(__name__)


class CovarianceCollector:
    """Forward hooks that accumulate ``XᵀX`` (fp64) of every SR LoRA input.

    Hooks the bare model: q_proj / o_proj / gate_proj / down_proj inputs, and the
    layer output (the cross_stream input ``h_base``) via the next layer's
    ``input_layernorm`` or the final norm. Only positions with ``mask`` set count.
    """

    def __init__(self, sr_model):
        self.layers = sr_model.model.layers
        self.cov: dict[tuple[int, str], torch.Tensor] = {}
        self.count: dict[tuple[int, str], int] = {}
        self.mask: torch.Tensor | None = None
        self._handles = []
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            self._hook(layer.self_attn.q_proj, i, "q")
            self._hook(layer.self_attn.o_proj, i, "o")
            self._hook(layer.mlp.gate_proj, i, "mlp_in")
            self._hook(layer.mlp.down_proj, i, "down")
            out_reader = self.layers[i + 1].input_layernorm if i + 1 < n else sr_model.model.norm
            self._hook(out_reader, i, "cross")

    def _hook(self, module, layer: int, group: str):
        def pre(_mod, args):
            x = args[0]
            x = x[self.mask] if self.mask is not None else x.reshape(-1, x.shape[-1])
            x = x.to(torch.float64)
            key = (layer, group)
            if key not in self.cov:
                self.cov[key] = torch.zeros(x.shape[-1], x.shape[-1], dtype=torch.float64, device=x.device)
                self.count[key] = 0
            self.cov[key].addmm_(x.T, x)
            self.count[key] += x.shape[0]

        self._handles.append(module.register_forward_pre_hook(pre))

    def remove(self):
        for h in self._handles:
            h.remove()

    def covariance(self, layer: int, group: str) -> torch.Tensor:
        return self.cov[(layer, group)] / self.count[(layer, group)]


@torch.no_grad()
def collect(sr_model, batches) -> CovarianceCollector:
    """``batches``: iterable of ``(input_ids, attention_mask)`` on the model's device."""
    collector = CovarianceCollector(sr_model)
    try:
        for input_ids, attention_mask in batches:
            collector.mask = attention_mask.bool()
            sr_model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        collector.remove()
    return collector


def _kaiming_scale(cov: torch.Tensor) -> float:
    """Per-feature std a stock kaiming_uniform(a=√5) lora_A gives on inputs with ``cov``."""
    return math.sqrt(float(cov.diagonal().sum()) / (3 * cov.shape[0]))


@torch.no_grad()
def compute_a(sr_model, collector: CovarianceCollector, rank: int, *, damp: float = 1e-6,
              feature_scale: str = "kaiming") -> tuple[dict[str, torch.Tensor], dict]:
    """One A per (layer, group); returns ``(tensors keyed layers.{l}.{site}, stats)``."""
    tensors: dict[str, torch.Tensor] = {}
    stats: dict[str, dict] = {}
    for i, layer in enumerate(collector.layers):
        weights = {
            "q": [layer.self_attn.q_proj.weight],
            "o": [layer.self_attn.o_proj.weight],
            "mlp_in": [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight],
            "down": [layer.mlp.down_proj.weight],
        }
        for group, sites in GROUP_SITES.items():
            cov = collector.covariance(i, group)
            if group == "cross":
                a, energy = pca_a(cov, rank, damp=damp)
            else:
                w = torch.cat([x.to(cov.device) for x in weights[group]], dim=0)
                a, energy = whitened_svd_a(w, cov, rank, damp=damp)
            scale = _kaiming_scale(cov) if feature_scale == "kaiming" else 1.0
            a = a * scale
            gram = a @ cov @ a.T
            whiteness = float((gram / scale ** 2 - torch.eye(rank, dtype=gram.dtype, device=gram.device)).abs().max())
            a = a.float().cpu()
            for site in sites:
                tensors[f"layers.{i}.{site}"] = a.clone()
            stats[f"layers.{i}.{group}"] = {
                "energy": energy, "scale": scale, "whiteness_err": whiteness,
                "tokens": collector.count[(i, group)], "in_features": cov.shape[0],
            }
    return tensors, stats


def _calibration_rows(paths: list[str], rows_per_file: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for p in paths:
        with open(p) as f:
            lines = [ln for ln in f if ln.strip()]
        take = rng.sample(range(len(lines)), min(rows_per_file, len(lines)))
        rows += [json.loads(lines[j]) for j in sorted(take)]
    return rows


def _batches(tok, rows, max_tokens: int, batch_size: int, device):
    from shadow_residual.training.chat_render import render_chat

    ids = []
    for r in rows:
        text = render_chat(tok, r["messages"], documents=r.get("documents"), tools=r.get("tools"),
                           add_generation_prompt=False, enable_thinking=False)
        ids.append(tok(text, add_special_tokens=False).input_ids[:max_tokens])
    ids.sort(key=len)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(ids), batch_size):
        chunk = ids[s:s + batch_size]
        width = max(len(x) for x in chunk)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, x in enumerate(chunk):
            input_ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        yield input_ids.to(device), mask.to(device)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--data", required=True, nargs="+", help="calibration jsonl files (rows with messages)")
    p.add_argument("--rows-per-file", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--damp", type=float, default=1e-6)
    p.add_argument("--feature-scale", choices=["kaiming", "unit"], default="kaiming")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from shadow_residual.shadow_residual.build import build_sr_base

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = build_sr_base(args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    rows = _calibration_rows(args.data, args.rows_per_file, args.seed)
    logger.info("calibrating on %d rows from %d files", len(rows), len(args.data))
    collector = collect(model, _batches(tok, rows, args.max_tokens, args.batch_size, device))
    tensors, stats = compute_a(model, collector, args.rank, damp=args.damp, feature_scale=args.feature_scale)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / A_FILE))
    summary = {}
    for g in GROUP_SITES:
        vals = [v for k, v in stats.items() if k.endswith(f".{g}")]
        summary[g] = {
            "energy_mean": sum(v["energy"] for v in vals) / len(vals),
            "energy_min": min(v["energy"] for v in vals),
            "whiteness_err_max": max(v["whiteness_err"] for v in vals),
        }
    manifest = {
        "recipe": {"q/o/down": "SVD-LLM whitened SVD", "gate+up": "shared whitened SVD of [W_g;W_u]",
                   "cross_stream": "unit-variance PCA of h_base"},
        "rank": args.rank, "damp": args.damp, "feature_scale": args.feature_scale,
        "base_model": args.base_model, "data": args.data, "rows": len(rows),
        "rows_per_file": args.rows_per_file, "max_tokens": args.max_tokens, "seed": args.seed,
        "summary": summary, "per_group": stats,
    }
    (args.out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    print("FROZEN_A_SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
