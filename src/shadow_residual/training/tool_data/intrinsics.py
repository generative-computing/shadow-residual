# SPDX-License-Identifier: Apache-2.0
"""Adapter-bench intrinsics as tools: source rows -> tool-formatted rows.

Each intrinsic is exposed to the base as a tool. A source row (adapter-bench
training row, or benchmark eval row) is split into the conversation the base
saw, the tool call's arguments (the intrinsic's instruction, moved out of the
``<guardian>`` / ``<requirements>`` user turn into the call), and the gold
tool-response dict (exactly what the harness scorer parses).

Gold comes from the training row's final assistant turn when present, else
from ``ground_truth`` (eval shape; query-clarification eval rows carry no gold
dict, so they convert only with ``include_response=False``).
"""

from __future__ import annotations

import json
import re

_GUARDIAN = re.compile(r"### Criteria: (.*?)\n\n### Scoring Schema: (.*)$", re.S)
_REQUIREMENTS = re.compile(r"<requirements>(.*?)\nPlease verify", re.S)


def _tool(name: str, description: str, params: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS = {
    "guardian_core": _tool(
        "guardian_core",
        "Judge whether a turn of the conversation so far meets a risk criterion. "
        'Returns {"score": "yes"} if it does and {"score": "no"} otherwise.',
        {
            "criteria": "The risk criterion to judge the text against.",
            "target": "Whose text to judge: the last 'user' turn or the last 'assistant' turn.",
        },
    ),
    "requirement_check": _tool(
        "requirement_check",
        "Check whether the assistant's last response satisfies a list of "
        'requirements. Returns {"score": "yes"} if all are satisfied and '
        '{"score": "no"} otherwise.',
        {"requirements": "The requirements the assistant's last response must satisfy."},
    ),
    "query_clarification": _tool(
        "query_clarification",
        "Decide whether the user's latest query is clear enough to answer from "
        'the conversation and documents. Returns {"clarification": "CLEAR"} if '
        "it is, otherwise a clarifying question to ask the user.",
        {"query": "The user's latest query."},
    ),
}

INTRINSICS = tuple(TOOLS)


def _drop_final_assistant(messages: list[dict]) -> tuple[list[dict], str | None]:
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1], messages[-1].get("content")
    return messages, None


def _score_gold(final: str | None, row: dict) -> dict:
    if final is not None:
        return json.loads(final)
    return {"score": str(row["ground_truth"]).strip().lower()}


def split_row(intrinsic: str, row: dict) -> tuple[list[dict], dict, dict | None]:
    """Return (conversation the base saw, tool-call arguments, gold dict or None)."""
    if intrinsic not in TOOLS:
        raise KeyError(f"unknown intrinsic {intrinsic!r}; known: {INTRINSICS}")
    messages, final = _drop_final_assistant(list(row["messages"]))

    if intrinsic == "query_clarification":
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not users:
            raise ValueError("query_clarification row has no user turn")
        args = {"query": messages[users[-1]]["content"]}
        return messages, args, json.loads(final) if final else None

    if not messages or messages[-1].get("role") != "user":
        raise ValueError(f"{intrinsic} row does not end in its instruction turn")
    instruction = messages[-1]["content"]
    convo = messages[:-1]
    if intrinsic == "guardian_core":
        m = _GUARDIAN.search(instruction)
        if not m:
            raise ValueError("guardian instruction has no criteria/schema")
        target = "user" if "last user's text" in m.group(2) else "assistant"
        args = {"criteria": m.group(1).strip(), "target": target}
    else:
        m = _REQUIREMENTS.match(instruction)
        if not m:
            raise ValueError("requirements instruction has no requirement list")
        args = {"requirements": m.group(1).strip()}
    return convo, args, _score_gold(final, row)


def to_tool_row(intrinsic: str, row: dict, *, include_response: bool = True) -> dict:
    """Convert one source row to a tool-formatted row.

    Documents are kept only for ``query_clarification`` (the other intrinsics
    judge the conversation itself); they land after the last user turn.
    """
    convo, args, gold = split_row(intrinsic, row)
    messages = [*convo, {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": intrinsic, "arguments": args}}],
    }]
    if include_response:
        if gold is None:
            raise ValueError(f"{intrinsic} row has no gold tool response")
        messages.append({"role": "tool", "content": json.dumps(gold, ensure_ascii=False)})
    out = {"messages": messages, "tools": [TOOLS[intrinsic]]}
    if intrinsic == "query_clarification" and row.get("documents"):
        out["documents"] = row["documents"]
    if "ground_truth" in row:
        out["ground_truth"] = row["ground_truth"]
    return out
