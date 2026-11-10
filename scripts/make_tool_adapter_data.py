# SPDX-License-Identifier: Apache-2.0
"""Build SR-adapter-as-tool training data from answerability-labeled rows.

Input rows (answerability eval/train shape):
    {"messages": [{"role":"user","content": "..."}, ...],
     "documents": [{"document_id": .., "content": ".."}, ...],
     "ground_truth": "answerable" | "unanswerable" | "partial" | "conversational"}

Output rows (the SR-adapter-as-tool conversation shape consumed by
training/data.py — `messages` + `tools`, rendered by the stock Granite 4.2
ChatML template):

    system (with the `answerability` tool defined)
    user  : the question (the last user turn of the input)
    tool  : the retrieved documents (RAG context)
    assistant(tool_calls=[answerability(question=...)])   <- the TRIGGER
    tool  : {"answerable": <bool>, "rationale": "..."}      <- adapter verdict
    assistant: <grounded answer | refusal>                  <- adapter answer

The supervised completion (set by the collator marker `</tool_call>`) is the
verdict tool_response + the assistant answer. See
docs/sr_tool_adapter/SR_TOOL_ADAPTER_BOOTSTRAP.md.

Gold-answer note: the answerability set carries a verdict label but no gold
answer text. For this bootstrap we synthesize a templated answer (grounded
phrasing for answerable rows, a standard refusal for unanswerable) so the
adapter learns the verdict->answer coupling and the structural scaffolding. The
verdict label is the real supervised signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tool_intrinsics import INTRINSICS, build_training_row  # noqa: E402

ANSWERABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "answerability",
        "description": (
            "Decide whether the user's question is answerable from the retrieved "
            "documents currently in context. Returns a verdict and a short "
            "rationale. Call this before answering any grounded/RAG question so "
            "the assistant can refuse unanswerable questions instead of "
            "hallucinating."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The user's question to assess for answerability against "
                        "the in-context documents."
                    ),
                }
            },
            "required": ["question"],
        },
    },
}

SYSTEM = (
    "You are a helpful enterprise assistant. Answer only from the retrieved "
    "documents. If a question cannot be answered from them, say so."
)


def _last_user_question(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content")
    return None


def _verdict_from_messages(messages: list[dict]) -> str | None:
    """Derive the answerability verdict from the FINAL assistant turn.

    The unified_compare answerability data (data_v4) carries no `ground_truth`
    field — it is already-rendered multi-turn chat whose last assistant turn is
    the gold verdict, a quoted label like '"answerable"' / '"unanswerable"'
    (optionally wrapped in quotes/whitespace). Returns a normalized
    "answerable" / "unanswerable" / "partial" / "conversational", or None if the
    last assistant turn isn't a recognizable verdict label."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            raw = (m.get("content") or "").strip().strip('"').strip().lower()
            if raw in ("answerable", "unanswerable", "partial", "conversational"):
                return raw
            return None
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


# What follows `<|im_start|>assistant\n` in the adapter's answer turn, by
# --thinking mode:
#   off      -> nothing (no think block at all)
#   enabled  -> "<think>"        (open think tag; model reasons before answering)
#   disabled -> "<think></think>" (empty, closed think block)
THINK_PREFIX = {"off": "", "enabled": "<think>", "disabled": "<think></think>"}

AUTOREGRESSIVE = "autoregressive_tool_formatting"
EXTERNAL = "external_tool_formatting"
_SENTINEL = "__SR_TOOL_RESPONSE_SENTINEL__"


def _external_context(context_msgs: list[dict], tokenizer) -> str:
    """Stock-template render through the env-written "<tool_response>\\n".

    The env closes the base's tool-call turn and opens the tool-response turn
    exactly as the chat template would, so render with a sentinel tool message
    and cut right before it."""
    rendered = tokenizer.apply_chat_template(
        context_msgs + [{"role": "tool", "content": _SENTINEL}],
        tools=[ANSWERABILITY_TOOL], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    ctx = rendered[: rendered.rindex(_SENTINEL)]
    if not ctx.endswith("<tool_response>\n"):
        raise ValueError(f"unexpected tool-response opener: {ctx[-40:]!r}")
    return ctx


def build_row(
    obj: dict, tokenizer, thinking: str = "off", tool_formatting: str = EXTERNAL
) -> dict | None:
    # The gold verdict is either an explicit `ground_truth` field (raw
    # answerability_evaluation.jsonl shape) or, failing that, the final assistant
    # turn of an already-rendered conversation (unified_compare/data_v4 shape).
    gt = obj.get("ground_truth") or _verdict_from_messages(obj.get("messages", []))
    # Keep the clean binary decision: map partial->answerable, drop conversational
    # (no grounding decision to learn) and anything unrecognized.
    if gt in ("conversational", None):
        return None
    answerable = gt in ("answerable", "partial")
    label = "answerable" if answerable else "unanswerable"

    question = _last_user_question(obj.get("messages", []))
    if not question:
        return None
    docs = _normalize_documents(obj.get("documents"))

    # The adapter's output is driven entirely by `ground_truth`:
    #   - tool_response: the verdict as a JSON object (the tool's structured result)
    #   - answer turn: the canonical quoted verdict label ("answerable" /
    #     "unanswerable") — exactly the gold completion the standalone
    #     answerability adapter emits (answerability_test.jsonl completion = '"answerable"').
    # No synthesized prose answer — the answerability tool's job IS the verdict.
    verdict = {"answerable": bool(answerable)}
    answer = json.dumps(label)  # '"answerable"' / '"unanswerable"'

    # Context: system + user + docs(tool) + the trigger tool_call. Rendered via
    # the stock template, then TRIMMED at </tool_call> so the turn-closing
    # <|im_end|> (the EOS/stop token) is NOT part of the row.
    context_msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "tool", "content": json.dumps(docs)},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "answerability",
                        "arguments": {"question": question},
                    },
                }
            ],
        },
    ]
    metadata = {"tool_name": "answerability", "verdict": gt, "tool_formatting": tool_formatting}

    if tool_formatting == EXTERNAL:
        # The adapter emits ONLY the tool-response dict, then EOS (<|im_end|>, the
        # stock generate() stop). The env drops the EOS and appends
        # "\n</tool_response>\n<|im_end|>\n" itself.
        ctx = _external_context(context_msgs, tokenizer)
        return {"text": ctx + json.dumps(verdict) + "<|im_end|>", "metadata": metadata}

    ctx = tokenizer.apply_chat_template(
        context_msgs, tools=[ANSWERABILITY_TOOL], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    ctx = ctx[: ctx.rindex("</tool_call>") + len("</tool_call>")]

    # Completion: the adapter's output with EXACTLY ONE <|im_end|> (the terminal
    # stop). The intermediate turn-closing <|im_end|> is intentionally omitted so
    # stock generate() doesn't halt before the answer. The tool_response block and
    # the chat-template scaffolding are built through `<|im_start|>assistant\n`;
    # what follows is the --thinking prefix (off/enabled/disabled), then the
    # answer. See docs/sr_tool_adapter/.
    completion = (
        "\n<|im_start|>user\n<tool_response>\n"
        + json.dumps(verdict)
        + "\n</tool_response>\n<|im_start|>assistant\n"
        + THINK_PREFIX[thinking]
        + answer
        + "<|im_end|>"
    )
    return {"text": ctx + completion, "metadata": metadata}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="source jsonl (answerability: messages/documents/ground_truth)")
    ap.add_argument(
        "--intrinsic",
        choices=["answerability", *INTRINSICS],
        default="answerability",
        help="Which tool to build rows for. The adapter-bench intrinsics "
             f"({', '.join(INTRINSICS)}) support {EXTERNAL} only.",
    )
    ap.add_argument("--base-model", required=True, help="tokenizer source (for rendering the context)")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--thinking",
        choices=["off", "enabled", "disabled"],
        default="off",
        help=(
            "What the adapter's answer turn emits after '<|im_start|>assistant\\n': "
            "off = nothing (default); enabled = '<think>'; disabled = '<think></think>'. "
            "Autoregressive mode only (external mode has no answer turn)."
        ),
    )
    ap.add_argument(
        "--tool-formatting",
        choices=[EXTERNAL, AUTOREGRESSIVE],
        default=EXTERNAL,
        help=(
            f"{EXTERNAL} (default): env writes '<tool_response>\\n'; adapter emits only "
            f"the dict, then EOS. {AUTOREGRESSIVE}: adapter starts after '</tool_call>' "
            "and emits the tool_response turn + final answer."
        ),
    )
    args = ap.parse_args()
    if args.tool_formatting == EXTERNAL and args.thinking != "off":
        ap.error(f"--thinking applies only to {AUTOREGRESSIVE}")
    if args.intrinsic != "answerability" and args.tool_formatting != EXTERNAL:
        ap.error(f"--intrinsic {args.intrinsic} supports {EXTERNAL} only")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    rows = []
    with Path(args.input).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if args.intrinsic == "answerability":
                r = build_row(
                    obj, tokenizer,
                    thinking=args.thinking, tool_formatting=args.tool_formatting,
                )
            else:
                r = build_training_row(args.intrinsic, obj, tokenizer)
            if r is not None:
                rows.append(r)
    if args.limit:
        rows = rows[: args.limit]

    # Deterministic shuffle without Math.random — index-based rotation by seed.
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
