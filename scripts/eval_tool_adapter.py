# SPDX-License-Identifier: Apache-2.0
"""Evaluate an SR-adapter-as-tool answerability adapter (experiment script).

Rows are tool rows from `python -m shadow_residual.training.tool_data --task
answerability` (train/val rows keep the gold tool message, which is dropped
from the prompt; gold verdict comes from `ground_truth`). The mode is read from
the adapter's adapter_config.json (`tool_formatting`). Each prompt is rendered
up to the mode's handoff point; the adapter generates greedily until EOS and
the tool-response payload is extracted and parsed as {"answerable": bool}.

Reports structural_rate (payload parses with a boolean `answerable`),
verdict_accuracy, a confusion table and eos_stop_rate.

Usage:
  python scripts/eval_tool_adapter.py --base-model ibm-granite/granite-4.2-3b \
    --adapter runs/<run>/checkpoints --val data/tool_adapter/val.jsonl \
    --limit 200 --out runs/<run>/eval.json --dump runs/<run>/eval_dump.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shadow_residual.training.tool_data import extract_tool_payload, render_tool_prompt
from shadow_residual.training.tool_data.formatting import EOS


def parse_verdict(payload: str) -> bool | None:
    try:
        obj = json.loads(payload.strip())
    except ValueError:
        return None
    val = obj.get("answerable") if isinstance(obj, dict) else None
    return val if isinstance(val, bool) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="write per-row generations jsonl")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from shadow_residual.shadow_residual.build import build_sr_base
    from shadow_residual.training.generation_utils import read_tool_formatting_from_adapter

    mode = read_tool_formatting_from_adapter(args.adapter)
    if mode is None:
        raise SystemExit(f"{args.adapter}/adapter_config.json records no tool_formatting")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(build_sr_base(args.base_model, torch_dtype=dtype), args.adapter)
    model.to(device).eval()

    rows = [json.loads(l) for l in Path(args.val).open() if l.strip()][: args.limit]
    n = struct_ok = acc_ok = stop_ok = 0
    conf: dict[str, int] = {}
    dump = open(args.dump, "w") if args.dump else None
    for row in rows:
        prompt = render_tool_prompt(tok, row["messages"], tools=row.get("tools"),
                                    documents=row.get("documents"), mode=mode)
        gold = row["ground_truth"] in ("answerable", "partial")
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        payload = extract_tool_payload(gen, mode)
        pred = parse_verdict(payload)
        n += 1
        stop_ok += EOS in gen
        key = f"{'answerable' if gold else 'unanswerable'}->" + (
            "invalid" if pred is None else "answerable" if pred else "unanswerable")
        conf[key] = conf.get(key, 0) + 1
        if pred is not None:
            struct_ok += 1
            acc_ok += pred == gold
        if dump:
            dump.write(json.dumps({"gold_answerable": gold, "payload": payload, "raw": gen[:600]}) + "\n")
    if dump:
        dump.close()

    res = {
        "tool_formatting": mode,
        "n": n,
        "structural_valid": struct_ok,
        "structural_rate": round(struct_ok / max(n, 1), 4),
        "verdict_accuracy": round(acc_ok / max(n, 1), 4),
        "verdict_accuracy_of_valid": round(acc_ok / max(struct_ok, 1), 4),
        "eos_stop_rate": round(stop_ok / max(n, 1), 4),
        "confusion": dict(sorted(conf.items())),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
