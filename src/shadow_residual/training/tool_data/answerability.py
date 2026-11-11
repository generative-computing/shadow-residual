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


def _is_document_turn(m: dict) -> bool:
    return m.get("role") in ("tool", "documents")


def documents(row: dict) -> list:
    """The row's ``documents`` plus its inline document turns (``tool`` /
    ``documents`` messages, the adapter-bench g42-alora-4.1-split shape), the
    latter as ``{"document_id": <index>, "content": ...}`` like the benchmark's
    own training-set renderer."""
    docs = list(row.get("documents") or [])
    for m in row.get("messages", []):
        if _is_document_turn(m):
            docs.append({"document_id": len(docs), "content": m.get("content")})
    return docs


def to_vanilla_row(row: dict) -> dict:
    """The source row as a plain chat row: inline document turns move into
    ``documents``; the final assistant turn (the quoted label) is the gold, built
    from ``ground_truth`` when the row has none."""
    messages = [m for m in row.get("messages", []) if not _is_document_turn(m)]
    if messages and messages[-1].get("role") != "assistant" and _verdict(row):
        messages.append({"role": "assistant", "content": f'"{_verdict(row)}"'})
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("a vanilla answerability row must end with the gold assistant turn")
    out = {"messages": messages}
    docs = documents(row)
    if docs:
        out["documents"] = docs
    return out


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
    docs = documents(row)
    if docs:
        out["documents"] = docs
    return out
