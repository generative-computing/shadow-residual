# SPDX-License-Identifier: Apache-2.0
"""Evaluate a PLAIN SR answerability adapter: render each val row's prompt with
add_generation_prompt=True, generate, and score first-word verdict accuracy
against the gold label (normalized: lowercased, non-alpha stripped — matching
the standalone answerability scorer)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from shadow_residual.shadow_residual.build import build_sr_base

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = build_sr_base(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()

    rows = [json.loads(l) for l in Path(args.val).open() if l.strip()][: args.limit]
    n = correct = 0
    from collections import Counter
    conf = Counter()
    dump = open(args.dump, "w") if args.dump else None
    for row in rows:
        gold = "answerable" if row["metadata"]["verdict"] in ("answerable", "partial") else "unanswerable"
        # prompt = messages WITHOUT the final assistant turn, + generation prompt
        prompt_msgs = row["messages"][:-1]
        text = tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred_word = gen.split()[0] if gen.split() else ""
        pred = "answerable" if norm(pred_word) == "answerable" else (
            "unanswerable" if norm(pred_word) == "unanswerable" else pred_word)
        n += 1
        if norm(pred_word) == norm(gold):
            correct += 1
        conf[(gold, "answerable" if norm(pred_word) == "answerable" else
              "unanswerable" if norm(pred_word) == "unanswerable" else "other")] += 1
        if dump:
            dump.write(json.dumps({"gold": gold, "pred": pred, "gen": gen[:80]}) + "\n")
    if dump:
        dump.close()

    res = {"n": n, "accuracy": round(correct / max(n, 1), 4),
           "confusion": {f"{g}->{p}": c for (g, p), c in sorted(conf.items())}}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
