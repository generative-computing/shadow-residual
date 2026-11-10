# SPDX-License-Identifier: Apache-2.0
"""Evaluate an external-tool-formatting SR adapter on an adapter-bench eval set.

Each benchmark eval row becomes the external prompt (tool_intrinsics.py: the
conversation, the base's tool call, the env-written `<tool_response>\\n`). The
adapter generates greedily in bf16 until EOS; the generation is the tool's
payload, scored by the adapter-bench harness's own scorer for the intrinsic
(loaded from --harness, a checkout holding benchmarks/adapter_eval/scorers).

Usage:
  python scripts/eval_tool_intrinsics.py --intrinsic guardian_core \\
    --base-model <4.2> --adapter <ckpt> --eval <bench>/eval/guardian_core/evaluation.jsonl \\
    --harness <harness checkout> --out <run>/eval.json --dump <run>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tool_intrinsics import EOS, INTRINSICS, external_context, split_row  # noqa: E402

SCORERS = {
    "guardian_core": "guardian",
    "query_clarification": "query_clarification",
    "requirement_check": "requirements",
}
HEADLINE = {
    "guardian_core": "accuracy",
    "query_clarification": "overall_accuracy",
    "requirement_check": "balanced_accuracy",
}


def load_scorer(harness: Path, intrinsic: str):
    sys.path.insert(0, str(harness))
    from benchmarks.adapter_eval.scorers import ScoreContext, get_scorer

    return get_scorer(SCORERS[intrinsic]), ScoreContext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsic", required=True, choices=INTRINSICS)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True, help="benchmark evaluation.jsonl")
    ap.add_argument("--harness", required=True, help="checkout with benchmarks/adapter_eval/scorers")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="write prediction rows (eval row + generated_content)")
    args = ap.parse_args()

    score_fn, ScoreContext = load_scorer(Path(args.harness), args.intrinsic)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from shadow_residual.shadow_residual.build import build_sr_base

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(build_sr_base(args.base_model, torch_dtype=dtype), args.adapter)
    model.to(device).eval()

    rows = [json.loads(l) for l in Path(args.eval).open() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    prompts = []
    for r in rows:
        convo, call_args, _ = split_row(args.intrinsic, r)
        prompts.append(external_context(args.intrinsic, convo, call_args, tok))

    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    raw = [""] * len(rows)
    eos_stop = [False] * len(rows)
    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        width = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            text = tok.decode(out[j, width:], skip_special_tokens=False)
            eos_stop[i] = EOS in text
            raw[i] = text.split(EOS, 1)[0].replace(tok.pad_token, "").strip()
        print(f"[eval] {min(start + args.batch_size, len(order))}/{len(order)}", flush=True)

    preds = [{**r, "generated_content": g} for r, g in zip(rows, raw)]
    result = score_fn(preds, ScoreContext(eval_dir=Path(args.eval).parent))
    res = {
        "intrinsic": args.intrinsic,
        "tool_formatting": "external_tool_formatting",
        "scorer": SCORERS[args.intrinsic],
        "headline": HEADLINE[args.intrinsic],
        "headline_value": result.metrics.get(HEADLINE[args.intrinsic]),
        "eos_stop_rate": round(sum(eos_stop) / max(len(rows), 1), 4),
        "metrics": result.metrics,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({**res, "details": result.details}, indent=2))
    if args.dump:
        with Path(args.dump).open("w") as f:
            for p, s in zip(preds, eos_stop):
                f.write(json.dumps({**p, "eos_stop": s}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
