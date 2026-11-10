# SPDX-License-Identifier: Apache-2.0
"""Convert task source rows into tool-formatted rows (mode-independent).

    # training data (gold tool response kept, shuffled train/val split)
    python -m shadow_residual.training.tool_data --task answerability \\
        --input src.jsonl --out-train train.jsonl --out-val val.jsonl

    # eval prompts (gold tool response dropped; gold stays in ground_truth)
    python -m shadow_residual.training.tool_data --task guardian_core \\
        --input evaluation.jsonl --eval --out eval.jsonl

The output rows are plain ``messages`` + ``tools`` (+ ``documents``) rows; the
trainer renders them for whichever ``data.tool_formatting`` mode it runs, so the
same file trains either mode. No tokenizer is needed here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from shadow_residual.training.tool_data import answerability, intrinsics

TASKS = ("answerability", *intrinsics.INTRINSICS)


def convert(task: str, row: dict, *, include_response: bool) -> dict | None:
    if task == "answerability":
        return answerability.to_tool_row(row, include_response=include_response)
    return intrinsics.to_tool_row(task, row, include_response=include_response)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m shadow_residual.training.tool_data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--input", required=True, help="source JSONL")
    p.add_argument("--eval", action="store_true",
                   help="write gold-free eval rows to --out instead of a train/val split")
    p.add_argument("--out", help="output JSONL (with --eval)")
    p.add_argument("--out-train", help="training split JSONL")
    p.add_argument("--out-val", help="validation split JSONL")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="cap the number of output rows, taken after shuffling (0 = all)")
    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)
    if args.eval and not args.out:
        p.error("--eval requires --out")
    if not args.eval and not (args.out_train and args.out_val):
        p.error("--out-train and --out-val are required (or pass --eval --out)")

    rows = []
    with Path(args.input).open() as f:
        for line in f:
            if line.strip():
                r = convert(args.task, json.loads(line), include_response=not args.eval)
                if r is not None:
                    rows.append(r)
    n_usable = len(rows)
    if not args.eval:
        random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    def _write(path: str, data: list[dict]) -> None:
        with Path(path).open("w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.eval:
        _write(args.out, rows)
        print(f"wrote {len(rows)} eval rows (from {n_usable} usable)")
        return 0
    n_val = max(1, int(len(rows) * args.val_frac))
    _write(args.out_val, rows[:n_val])
    _write(args.out_train, rows[n_val:])
    print(f"wrote train={len(rows) - n_val} val={n_val} (from {n_usable} usable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
