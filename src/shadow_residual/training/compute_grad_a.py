# SPDX-License-Identifier: Apache-2.0
"""Task-gradient frozen-A banks from one backward pass over training rows.

The SR model is wrapped with a probe LoRA (B = 0, so it equals the base). Hooks on
every LoRA module see the adapter-stream input ``x`` and the gradient ``g`` of the
site output, and accumulate the full fine-tuning gradient ``G = Σ g·xᵀ`` (out×in)
and the per-row Fisher estimate ``F = Σ_batches rowsum(G_b²)``. Two banks follow:

* ``grad``: A = top-r right singular vectors of G (LoRA-One / LoRA-GA, frozen).
* ``fwsvd``: A = top-r right singular vectors of ``diag(√F)·W`` (Fisher-weighted SVD
  compression of the base weight, Hsu et al. 2022). cross_stream (W ≡ 0) is copied
  from ``--cross-from``.

gate and up share one A from their stacked matrices. Rows are scaled to the stock
kaiming row norm 1/√3. One bank per task (``<out-root>/frozen_a_<kind>_<task>_r32_v1``)
and one for the union of tasks, where each task's G and F are averaged over its
supervised tokens first (``..._<kind>_union_r32_v1``).

    python -m shadow_residual.training.compute_grad_a --base-model <path> --config <yaml> \
        --data <dir with <task>/train.jsonl> --tasks answerability ... --rows-per-task 256 \
        --cross-from <random bank> --out-root <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import torch

from shadow_residual.training.frozen_a import A_FILE, GROUP_SITES, MANIFEST_FILE, load_frozen_a, site_key

logger = logging.getLogger(__name__)

RANK_DEFAULT = 32


class GradCollector:
    """Accumulate ``G = Σ g xᵀ`` and row Fisher per LoRA site of the adapter stream."""

    def __init__(self, peft_model):
        from peft.tuners.lora.layer import LoraLayer

        self.G: dict[str, torch.Tensor] = {}
        self.F: dict[str, torch.Tensor] = {}
        self._batch: dict[str, torch.Tensor] = {}
        self._handles = []
        for name, module in peft_model.named_modules():
            key = site_key(name)
            if key is not None and isinstance(module, LoraLayer):
                self._handles.append(module.register_forward_hook(self._hook(key)))

    def _hook(self, key: str):
        def fwd(_mod, inputs, output):
            x = inputs[0].detach()

            def bwd(g):
                gb = g.reshape(-1, g.shape[-1]).float().T @ x.reshape(-1, x.shape[-1]).float()
                if key in self._batch:
                    self._batch[key] += gb
                else:
                    self._batch[key] = gb

            if output.requires_grad:
                output.register_hook(bwd)

        return fwd

    def end_batch(self):
        for key, gb in self._batch.items():
            if key in self.G:
                self.G[key] += gb
                self.F[key] += gb.pow(2).sum(dim=1)
            else:
                self.G[key] = gb.clone()
                self.F[key] = gb.pow(2).sum(dim=1)
        self._batch = {}

    def reset(self):
        self.G, self.F, self._batch = {}, {}, {}

    def remove(self):
        for h in self._handles:
            h.remove()


def _top_right_singular(m: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
    _, s, vh = torch.linalg.svd(m.float(), full_matrices=False)
    energy = float((s[:rank] ** 2).sum() / (s ** 2).sum().clamp_min(1e-30))
    return (vh[:rank] / math.sqrt(3)).contiguous(), energy


def _site_weight(peft_model, layer: int, site: str) -> torch.Tensor:
    layers = peft_model.get_base_model().model.layers
    mod = getattr(layers[layer].self_attn if site in ("q_proj", "o_proj") else layers[layer].mlp, site)
    return getattr(mod, "base_layer", mod).weight


@torch.no_grad()
def build_banks(peft_model, G: dict, F: dict, rank: int, cross: dict[str, torch.Tensor]):
    """Return ``{"grad": (tensors, stats), "fwsvd": (tensors, stats)}``."""
    n_layers = len(peft_model.get_base_model().model.layers)
    dev = next(peft_model.parameters()).device
    out = {"grad": ({}, {}), "fwsvd": ({}, {})}
    for i in range(n_layers):
        for group, sites in GROUP_SITES.items():
            keys = [f"layers.{i}.{s}" for s in sites]
            a, e = _top_right_singular(torch.cat([G[k].to(dev) for k in keys], dim=0), rank)
            for k in keys:
                out["grad"][0][k] = a.cpu().clone()
            out["grad"][1][f"layers.{i}.{group}"] = {"grad_energy_top_r": e}
            if group == "cross":
                for k in keys:
                    out["fwsvd"][0][k] = cross[k].float().clone()
                continue
            w = torch.cat([_site_weight(peft_model, i, s).to(dev).float() for s in sites], dim=0)
            f = torch.cat([F[k].to(dev) for k in keys], dim=0)
            a, e = _top_right_singular(f.clamp_min(0).sqrt().unsqueeze(1) * w, rank)
            for k in keys:
                out["fwsvd"][0][k] = a.cpu().clone()
            out["fwsvd"][1][f"layers.{i}.{group}"] = {"weighted_energy_top_r": e}
    return out


def collect_task(peft_model, collector: GradCollector, batches) -> int:
    """Run forward+backward on labelled batches; returns the number of supervised tokens."""
    n_tok = 0
    peft_model.zero_grad(set_to_none=True)
    for batch in batches:
        out = peft_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        out.loss.backward()
        collector.end_batch()
        peft_model.zero_grad(set_to_none=True)
        n_tok += int((batch["labels"] != -100).sum())
    return n_tok


def _labelled_batches(tok, cfg, path: Path, n_rows: int, seed: int, max_length: int, device):
    from shadow_residual.training.data import ResponseOnlyCollator, load_jsonl_dataset

    ds = load_jsonl_dataset(path, tok, enable_thinking=cfg.data.enable_thinking, last_token=cfg.adapter.last_token,
                            instruction_as_user_message=cfg.data.instruction_as_user_message)
    idx = random.Random(seed).sample(range(len(ds)), min(n_rows, len(ds)))
    collate = ResponseOnlyCollator(tokenizer=tok, response_template=cfg.data.assistant_marker)
    skipped = 0
    for j in idx:
        ids = tok(ds[j]["text"], add_special_tokens=False).input_ids
        if len(ids) > max_length:
            skipped += 1
            continue
        b = collate([{"input_ids": ids}])
        if int((b["labels"] != -100).sum()) == 0:
            skipped += 1
            continue
        yield {k: v.to(device) for k, v in b.items()}
    if skipped:
        logger.info("%s: skipped %d rows (too long or unsupervised)", path, skipped)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--config", required=True, help="training YAML (data markers, last_token)")
    p.add_argument("--data", required=True, type=Path, help="dir with <task>/train.jsonl")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--rows-per-task", type=int, default=256)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--rank", type=int, default=RANK_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cross-from", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    args = p.parse_args(argv)

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from shadow_residual.config.training_config import load_training_config
    from shadow_residual.shadow_residual.build import build_sr_base

    cfg = load_training_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    base = build_sr_base(args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    sites = [s for g in GROUP_SITES.values() for s in g]
    model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=sites))
    model.train()
    cross = load_frozen_a(args.cross_from)
    collector = GradCollector(model)
    union_G: dict[str, torch.Tensor] = {}
    union_F: dict[str, torch.Tensor] = {}

    def save(kind: str, name: str, tensors: dict, stats: dict, extra: dict):
        out_dir = args.out_root / f"frozen_a_{kind}_{name}_r{args.rank}_v1"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_dir / A_FILE))
        summary = {g: sum(next(iter(v.values())) for k, v in stats.items() if k.endswith(f".{g}"))
                   / max(1, sum(1 for k in stats if k.endswith(f".{g}"))) for g in GROUP_SITES
                   if any(k.endswith(f".{g}") for k in stats)}
        manifest = {"recipe": {"kind": kind, "A": "top-r right singular vectors" + (" of G" if kind == "grad" else " of diag(sqrt F) W"),
                               "cross_stream": "grad" if kind == "grad" else f"copied from {args.cross_from}"},
                    "rank": args.rank, "feature_scale": "row norm 1/sqrt3 (stock kaiming)", "base_model": args.base_model,
                    "rows_per_task": args.rows_per_task, "max_length": args.max_length, "seed": args.seed,
                    "summary_mean_top_r_energy": summary, "per_group": stats, **extra}
        (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        print(f"GRAD_A_WROTE {out_dir.name} " + json.dumps(summary), flush=True)

    for t_i, task in enumerate(args.tasks):
        collector.reset()
        if device == "cuda":
            torch.cuda.empty_cache()
        n_tok = collect_task(model, collector, _labelled_batches(tok, cfg, args.data / task / "train.jsonl",
                                                                 args.rows_per_task, args.seed + t_i, args.max_length, device))
        logger.info("%s: %d supervised tokens, %d sites", task, n_tok, len(collector.G))
        banks = build_banks(model, collector.G, collector.F, args.rank, cross)
        for kind, (tensors, stats) in banks.items():
            save(kind, task, tensors, stats, {"tasks": [task], "supervised_tokens": n_tok})
        for k in collector.G:
            g, f = (collector.G[k] / n_tok).cpu(), (collector.F[k] / n_tok ** 2).cpu()
            union_G[k] = union_G[k] + g if k in union_G else g
            union_F[k] = union_F[k] + f if k in union_F else f
    banks = build_banks(model, union_G, union_F, args.rank, cross)
    for kind, (tensors, stats) in banks.items():
        save(kind, "union", tensors, stats, {"tasks": args.tasks})
    collector.remove()
    print("ALL_GRAD_BANKS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
