# SPDX-License-Identifier: Apache-2.0
"""Evaluate one SR adapter on one intrinsic with the adapter benchmark's own code.

Builds the job exactly as ``benchmarks/adapter_eval/reference.py`` does for the
``sr`` column on granite-4.2-3b (documents ``tool_json_after_question``,
``enable_thinking=False``, the intrinsic's ``max_new_tokens``, max_model_len
32768), runs ``benchmarks.adapter_eval.hf_generate`` in-process, and scores with
``benchmarks.adapter_eval.scorers``. Writes predictions.jsonl, status.json and
score.json into --out.

    python scripts/bench_eval.py --bench-harness <dir> --base-model <path> \
        --adapter <adapter dir> --intrinsic answerability \
        --eval-root /danieloh_cos/granite-switch/adapter-bench/granite-4.2-3b/v1/eval --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INTRINSICS = {
    "answerability": ("answerability", 200, "accuracy"),
    "guardian_core": ("guardian", 24, "accuracy"),
    "query_clarification": ("query_clarification", 160, "overall_accuracy"),
    "requirement_check": ("requirements", 200, "balanced_accuracy"),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-harness", required=True, type=Path)
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--intrinsic", required=True, choices=sorted(INTRINSICS))
    p.add_argument("--eval-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.bench_harness))
    from benchmarks.adapter_eval import hf_generate
    from benchmarks.adapter_eval.scorers import ScoreContext, get_scorer
    from benchmarks.adapter_eval.staged import read_jsonl

    scorer, max_new, headline = INTRINSICS[args.intrinsic]
    eval_dir = args.eval_root / args.intrinsic
    args.out.mkdir(parents=True, exist_ok=True)
    job = {
        "key": f"{args.intrinsic}/sr",
        "column": "sr",
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter),
        "eval_path": str(eval_dir / "evaluation.jsonl"),
        "out_path": str(args.out / "predictions.jsonl"),
        "status_path": str(args.out / "status.json"),
        "limit": args.limit,
        "max_new_tokens": max_new,
        "max_model_len": 32768,
        "token_budget": 262144,
        "max_batch": 64,
        "documents": "tool_json_after_question",
        "chat_template_kwargs": {"enable_thinking": False},
        "instruction": None,
    }
    (args.out / "job.json").write_text(json.dumps(job, indent=2))
    hf_generate.main([str(args.out / "job.json")])
    status = json.loads((args.out / "status.json").read_text())
    if not status.get("ok"):
        print(f"GENERATION FAILED: {status}", flush=True)
        return 1
    rows = read_jsonl(args.out / "predictions.jsonl")
    result = get_scorer(scorer)(rows, ScoreContext(eval_dir=eval_dir, env={}))
    (args.out / "score.json").write_text(
        json.dumps({"metrics": result.metrics, "details": result.details}, indent=2)
    )
    print(f"SCORE {args.intrinsic} {headline}={result.metrics[headline]:.4f} "
          f"metrics={json.dumps(result.metrics)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
