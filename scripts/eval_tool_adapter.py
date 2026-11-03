# SPDX-License-Identifier: Apache-2.0
"""Evaluate an SR-adapter-as-tool answerability adapter.

Builds, for each val row, the inference prompt that ends right after the trigger
`<tool_call>` close (so the adapter continues from the switch point), generates
with the adapter attached, then scores:

  - structural: the generated `<tool_response>` body parses as JSON with an
    `answerable` boolean key
  - accuracy:   parsed verdict == gold verdict
  - consistency: for answerable the answer is non-refusal; for unanswerable the
    answer refuses (polarity of the answer matches the emitted verdict)

Usage:
  python scripts/eval_tool_adapter.py \
    --base-model ibm-research/granite-4.2-3b-prerelease-r260622a \
    --adapter runs/<run>/checkpoints/checkpoint-XXXX \
    --val data/tool_adapter/val.jsonl \
    --limit 200 --out runs/<run>/eval.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# The answer turn is the canonical quoted verdict label ("answerable" /
# "unanswerable"); consistency = answer label matches the emitted verdict bool.


def build_context(tokenizer, row: dict) -> tuple[str, bool]:
    """Return (prompt_text, gold_answerable). The prompt ends right at the
    trigger `</tool_call>` so generation resumes where the adapter would.

    Rows are pre-rendered `text` (the SR-adapter-as-tool shape) — slice at
    `</tool_call>`. Falls back to rendering `messages` if present (legacy)."""
    gold = row["metadata"]["verdict"]
    gold_answerable = gold in ("answerable", "partial")
    if "text" in row:
        text = row["text"]
    else:
        msgs = row["messages"]
        trigger_idx = next(i for i, m in enumerate(msgs) if m.get("tool_calls"))
        text = tokenizer.apply_chat_template(
            msgs[: trigger_idx + 1], tools=row.get("tools"), tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
    cut = text.rindex("</tool_call>") + len("</tool_call>")
    return text[:cut], gold_answerable


def parse_verdict(generated: str) -> dict | None:
    """Extract the first <tool_response>...</tool_response> JSON body."""
    m = re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", generated, re.DOTALL)
    blob = m.group(1) if m else generated
    # try to find a JSON object with "answerable"
    jm = re.search(r"\{.*?\"answerable\".*?\}", blob, re.DOTALL)
    if not jm:
        return None
    try:
        return json.loads(jm.group(0))
    except Exception:
        return None


def extract_answer(generated: str) -> str:
    """The answer-turn body: everything after the last <|im_start|>assistant,
    stripped of a leading think block, up to <|im_end|>.

    Thinking-mode agnostic. The adapter's answer turn opens with
    <|im_start|>assistant\\n then, depending on --thinking: nothing (off),
    <think> (enabled — an open, possibly-unclosed reasoning block), or
    <think></think> (disabled). We take the text after the assistant marker,
    then drop any leading <think>...</think> or a dangling <think> so the
    extracted answer is the user-facing payload regardless of mode."""
    m = re.search(r"<\|im_start\|>assistant\s*(.*?)(?:<\|im_end\|>|$)",
                  generated, re.DOTALL)
    body = m.group(1) if m else generated
    # Strip a closed think block, or a dangling open <think> (enabled mode).
    body = re.sub(r"^\s*<think>.*?</think>", "", body, flags=re.DOTALL)
    body = re.sub(r"^\s*<think>", "", body)
    return body.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="write per-row generations jsonl")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from shadow_residual.shadow_residual.build import build_sr_base

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = build_sr_base(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.to(device).eval()

    rows = [json.loads(l) for l in Path(args.val).open() if l.strip()][: args.limit]
    n = struct_ok = acc_ok = cons_ok = 0
    dump = open(args.dump, "w") if args.dump else None
    for row in rows:
        prompt, gold_answerable = build_context(tok, row)
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        verdict = parse_verdict(gen)
        answer = extract_answer(gen)
        n += 1
        if verdict is not None and isinstance(verdict.get("answerable"), bool):
            struct_ok += 1
            pred = verdict["answerable"]
            if pred == gold_answerable:
                acc_ok += 1
            # Consistency: the answer-turn label matches the verdict bool.
            ans_l = answer.lower()
            answer_label = (
                True if "unanswerable" not in ans_l and "answerable" in ans_l
                else False if "unanswerable" in ans_l else None
            )
            if answer_label is not None and answer_label == pred:
                cons_ok += 1
        if dump:
            dump.write(json.dumps({"gold_answerable": gold_answerable, "verdict": verdict,
                                   "answer": answer[:300], "raw": gen[:600]}) + "\n")
    if dump:
        dump.close()

    res = {
        "n": n,
        "structural_valid": struct_ok,
        "structural_rate": round(struct_ok / max(n, 1), 4),
        "verdict_accuracy": round(acc_ok / max(n, 1), 4),
        "verdict_accuracy_of_valid": round(acc_ok / max(struct_ok, 1), 4),
        "consistency_rate": round(cons_ok / max(n, 1), 4),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
