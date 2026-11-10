# SPDX-License-Identifier: Apache-2.0
"""Answerability as a tool: source rows -> tool-formatted training/eval rows.

Source rows (either shape):
    {"messages": [...], "documents": [...], "ground_truth": "answerable" | ...}
    {"messages": [..., {"role": "assistant", "content": '"answerable"'}], "documents": [...]}

The gold verdict is ``ground_truth`` or, failing that, the final assistant turn
(a quoted label). ``partial`` counts as answerable; ``conversational`` and
unrecognized rows are dropped (no grounding decision to learn).

Output row: the base calls ``answerability(question=...)`` and the adapter's
tool response is ``{"answerable": <bool>}``::

    {"messages": [system, user(question),
                  assistant(tool_calls=[answerability]), tool('{"answerable": true}')],
     "tools": [ANSWERABILITY_TOOL], "documents": [...], "ground_truth": "answerable"}
"""

from __future__ import annotations

import json

ANSWERABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "answerability",
        "description": (
            "Decide whether the user's question is answerable from the retrieved "
            "documents currently in context. Call this before answering any "
            "grounded/RAG question so the assistant can refuse unanswerable "
            'questions instead of hallucinating. Returns {"answerable": true} or '
            '{"answerable": false}.'
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

_LABELS = ("answerable", "unanswerable", "partial", "conversational")


def _last_user_question(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content")
    return None


def _verdict(row: dict) -> str | None:
    gt = row.get("ground_truth")
    if gt is None:
        final = next((m for m in reversed(row.get("messages", [])) if m.get("role") == "assistant"), None)
        gt = (final or {}).get("content")
    if not isinstance(gt, str):
        return None
    gt = gt.strip().strip('"').strip().lower()
    return gt if gt in _LABELS else None


def to_tool_row(row: dict, *, include_response: bool = True) -> dict | None:
    """Convert one source row; ``None`` if it has no usable verdict or question.

    ``include_response=False`` drops the gold tool message, giving the eval
    shape (generation prompts; the gold stays in ``ground_truth``).
    """
    verdict = _verdict(row)
    question = _last_user_question(row.get("messages", []))
    if verdict in (None, "conversational") or not question:
        return None
    answerable = verdict in ("answerable", "partial")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "answerability", "arguments": {"question": question}},
            }],
        },
    ]
    if include_response:
        messages.append({"role": "tool", "content": json.dumps({"answerable": answerable})})
    out = {
        "messages": messages,
        "tools": [ANSWERABILITY_TOOL],
        "ground_truth": "answerable" if answerable else "unanswerable",
    }
    if row.get("documents"):
        out["documents"] = row["documents"]
    return out
