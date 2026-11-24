# SPDX-License-Identifier: Apache-2.0
"""Answerability as a tool.

Source rows (either shape):
    {"messages": [...], "documents": [...], "ground_truth": "answerable" | ...}
    {"messages": [..., {"role": "assistant", "content": '"answerable"'}], "documents": [...]}

The gold verdict is ``ground_truth`` or, failing that, the final assistant turn
(a quoted label). ``partial`` counts as answerable; ``conversational`` and
unrecognized rows are dropped (no grounding decision to learn). Documents may
also arrive as inline ``tool`` / ``documents`` turns (the adapter-bench
g42-alora-4.1-split shape); they are moved into ``documents`` as
``{"document_id": <index>, "content": ...}``, the records the benchmark's own
renderer builds. The tool row keeps the documents.

Tool row: the base calls ``answerability(question=...)`` and the adapter's
tool response is ``{"answerable": <bool>}``. The conversation is rebuilt as the
fixed ``SYSTEM`` text and the last user question (the chat template appends
``TOOL`` to that system turn)::

    {"messages": [system(SYSTEM), user(question),
                  assistant(tool_calls=[answerability]), tool('{"answerable": true}')],
     "tools": [TOOL], "ground_truth": "answerable", "documents": [...]}

A vanilla row keeps the source label as is (``partial`` stays ``"partial"``).
"""

from __future__ import annotations

from shadow_residual.training.tool_data.tasks.base import Row, ToolDataRenderer

TOOL = {
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


def _verdict(row: Row) -> str | None:
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


def documents(row: Row) -> list:
    """The row's ``documents`` plus its inline document turns, the latter as
    ``{"document_id": <index>, "content": ...}`` like the benchmark's own
    training-set renderer."""
    docs = list(row.get("documents") or [])
    for m in row.get("messages", []):
        if _is_document_turn(m):
            docs.append({"document_id": len(docs), "content": m.get("content")})
    return docs


class Answerability(ToolDataRenderer[Row]):
    tool_definition = TOOL

    def keep(self, row: Row) -> bool:
        return _verdict(row) not in (None, "conversational") and bool(_last_user_question(row.get("messages", [])))

    def to_tool_call(self, row: Row) -> list[dict]:
        question = _last_user_question(row["messages"])
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            self.tool_call({"question": question}),
        ]

    def format_response(self, row: Row) -> dict:
        return {"answerable": _verdict(row) in ("answerable", "partial")}

    def carried_fields(self, row: Row) -> dict:
        out = {"ground_truth": "answerable" if self.format_response(row)["answerable"] else "unanswerable"}
        docs = documents(row)
        if docs:
            out["documents"] = docs
        return out

    def vanilla_gold(self, row: Row) -> str | None:
        verdict = _verdict(row)
        return f'"{verdict}"' if verdict else None

    def to_vanilla_row(self, row: Row) -> Row:
        messages = [m for m in row.get("messages", []) if not _is_document_turn(m)]
        return super().to_vanilla_row({**row, "messages": messages, "documents": documents(row)})


__all__ = ["Answerability", "SYSTEM", "TOOL", "documents"]
