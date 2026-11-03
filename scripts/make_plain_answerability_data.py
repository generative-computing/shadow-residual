# SPDX-License-Identifier: Apache-2.0
"""Build PLAIN SR answerability training data (label-only, the standard way).

No tool-call framing: the model reads the question + retrieved documents and
answers directly with the verdict label. This is the canonical answerability
adapter flow (gold = the quoted label, matching the standalone answerability
adapter's completion `'"answerable"'`).

Output rows are the standard `messages`(+`documents`) shape consumed by
training/data.py, rendered by the stock chat template. The supervised completion
is the final assistant turn = the verdict label; the g42 config's
`assistant_marker` / `last_context_token` (`</think>`) handle the boundary.

Input: answerability jsonl with `messages` / `documents` / `ground_truth`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM = (
    "You are a helpful assistant with access to retrieved documents. Decide "
    "whether the user's question is answerable from them. Respond with exactly "
    'one word: "answerable" or "unanswerable".'
)


def _last_user_question(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content")
    return None


def _normalize_documents(docs) -> list[dict]:
    out = []
    for i, d in enumerate(docs or []):
        if isinstance(d, dict):
            content = d.get("content", d.get("text", ""))
            doc_id = str(d.get("document_id", d.get("doc_id", i)))
        else:
            content, doc_id = str(d), str(i)
        out.append({"source": "knowledge_base", "document_id": doc_id, "content": content})
    return out


def build_row(obj: dict) -> dict | None:
    gt = obj.get("ground_truth")
    if gt in ("conversational", None):
        return None
    answerable = gt in ("answerable", "partial")
    label = "answerable" if answerable else "unanswerable"

    question = _last_user_question(obj.get("messages", []))
    if not question:
        return None
    docs = _normalize_documents(obj.get("documents"))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "tool", "content": json.dumps(docs)},
        {"role": "assistant", "content": json.dumps(label)},  # '"answerable"'
    ]
    return {"messages": messages, "metadata": {"verdict": gt}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with Path(args.input).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = build_row(json.loads(line))
            if r is not None:
                rows.append(r)

    import random

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    for path, data in [(args.out_train, train), (args.out_val, val)]:
        with Path(path).open("w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
    print(f"wrote train={len(train)} val={len(val)} (from {len(rows)} usable rows)")


if __name__ == "__main__":
    main()
