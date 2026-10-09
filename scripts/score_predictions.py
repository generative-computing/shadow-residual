# SPDX-License-Identifier: Apache-2.0
"""Score a predictions.jsonl produced by shadow_residual.training.generate.

Reproduces the adapter-team scorers (see the user's spec):

  Answerability : first-word accuracy, norm(pred.split()[0]) == norm(gold.split()[0])
                  with norm = re.sub(r'[^a-z]', '', s.lower()).
  DROP / WTQ    : SQuAD-style best-window token-F1 with alias support (_aliases
                  field in each eval row; fall back to [ground_truth] if absent).

Each prediction row is expected to carry `generated_content` (model output) and
either `ground_truth` + optional `_aliases`, or (answerability) the gold embedded
as the final assistant message. The generate path copies through row fields it
doesn't consume, so `_aliases` / `ground_truth` survive from the eval JSONL.

Usage:
    python scripts/score_predictions.py --task {answerability,drop,wtq} \
        --predictions <run_dir>/checkpoints/predictions.jsonl \
        [--out <run_dir>/score.json]
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter


# --- normalization -----------------------------------------------------------

def _norm_firstword(s: str) -> str:
    """Answerability first-word norm: lowercase, strip all non-[a-z]."""
    return re.sub(r"[^a-z]", "", (s or "").lower())


_ARTICLES = {"a", "an", "the"}


def _squad_normalize(s: str) -> list[str]:
    """SQuAD normalization → token list: lowercase, drop punctuation and
    articles, collapse whitespace."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    toks = [t for t in s.split() if t not in _ARTICLES]
    return toks


# --- scorers -----------------------------------------------------------------

def _first_word_correct(pred: str, gold: str) -> bool:
    p = pred.strip().split()
    g = gold.strip().split()
    if not g:
        return False
    if not p:
        return False
    return _norm_firstword(p[0]) == _norm_firstword(g[0])


def _token_f1(pred: str, gold: str) -> float:
    """SQuAD token-F1 for a single (pred, gold) pair."""
    p = _squad_normalize(pred)
    g = _squad_normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def _best_window_f1(pred: str, golds: list[str]) -> float:
    """Best token-F1 over all gold aliases, each scored with a best-window
    slice of the prediction (SQuAD 'best window' = scan every contiguous
    pred-token window of the gold's length and keep the max). Mirrors the
    adapter-team scorer: alias support + windowing so a long generated answer
    that CONTAINS the gold span isn't penalized for extra tokens."""
    p_tokens = _squad_normalize(pred)
    best = 0.0
    for gold in golds:
        g_tokens = _squad_normalize(gold)
        if not g_tokens:
            best = max(best, 1.0 if not p_tokens else 0.0)
            continue
        if not p_tokens:
            continue
        w = len(g_tokens)
        # whole-pred F1 (standard SQuAD) ...
        best = max(best, _token_f1(pred, gold))
        # ... plus best contiguous window of the gold's length (handles verbose gen)
        for i in range(0, max(1, len(p_tokens) - w + 1)):
            window = " ".join(p_tokens[i : i + w])
            best = max(best, _token_f1(window, gold))
    return best


# --- row field extraction ----------------------------------------------------

def _extract_gold_answerability(row: dict) -> str | None:
    gt = row.get("ground_truth")
    if gt:
        return gt
    # Fall back to the final assistant message (answerability data_v4 shape).
    msgs = row.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip().strip('"')
    return None


def _extract_golds_f1(row: dict) -> list[str]:
    aliases = row.get("_aliases")
    if aliases:
        return [a for a in aliases if a]
    gt = row.get("ground_truth")
    if gt:
        return [gt]
    msgs = row.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return [(m.get("content") or "").strip()]
    return []


def _extract_pred(row: dict) -> str:
    return (
        row.get("generated_content")
        or row.get("generated")
        or row.get("prediction")
        or ""
    )


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["answerability", "drop", "wtq"])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    if n == 0:
        print(f"ERROR: no rows in {args.predictions}")
        return 1

    if args.task == "answerability":
        correct = 0
        scored = 0
        for r in rows:
            gold = _extract_gold_answerability(r)
            if gold is None:
                continue
            scored += 1
            if _first_word_correct(_extract_pred(r), gold):
                correct += 1
        metric = correct / scored if scored else 0.0
        result = {
            "task": args.task,
            "metric": "first_word_accuracy",
            "score": metric,
            "n_scored": scored,
            "n_total": n,
        }
    else:
        f1s = []
        for r in rows:
            golds = _extract_golds_f1(r)
            if not golds:
                continue
            f1s.append(_best_window_f1(_extract_pred(r), golds))
        metric = sum(f1s) / len(f1s) if f1s else 0.0
        result = {
            "task": args.task,
            "metric": "squad_token_f1_bestwindow_aliases",
            "score": metric,
            "n_scored": len(f1s),
            "n_total": n,
        }

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
