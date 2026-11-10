# SPDX-License-Identifier: Apache-2.0
"""External-tool-formatting specs for the adapter-bench intrinsics.

Each intrinsic is exposed to the base as a tool. A source row (adapter-bench
training row or benchmark eval row) is split into:

  - the conversation the base saw (normal user / assistant turns),
  - the tool call's arguments (the intrinsic's instruction, moved out of the
    `<guardian>` / `<requirements>` user turn into the call),
  - the gold tool-response dict, exactly what the harness scorer parses.

The same split builds the training rows (make_tool_adapter_data.py) and the
eval prompts (eval_tool_intrinsics.py), so train and eval prompts match.
"""

from __future__ import annotations

import json
import re

_SENTINEL = "__SR_TOOL_RESPONSE_SENTINEL__"
OPENER = "<tool_response>\n"
EOS = "<|im_end|>"

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


def _documents_payload(documents) -> str:
    """The harness's SR document form (prompts.py, tool_json_after_question)."""
    records = []
    for i, d in enumerate(documents):
        d = d if isinstance(d, dict) else {"text": str(d)}
        records.append({
            "source": d.get("source", "knowledge_base"),
            "document_id": str(d.get("document_id", d.get("doc_id", i))),
            "content": d.get("content", d.get("text", "")),
        })
    return json.dumps(records, indent=2)


def _drop_final_assistant(messages: list[dict]) -> tuple[list[dict], str | None]:
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1], messages[-1].get("content")
    return messages, None


def split_row(intrinsic: str, row: dict) -> tuple[list[dict], dict, dict | None]:
    """Return (conversation, tool-call arguments, gold dict or None).

    Gold comes from the training row's final assistant turn when present (the
    training shape), else from `ground_truth` (the eval shape; query
    clarification eval rows carry no gold dict and return None)."""
    messages, final = _drop_final_assistant(list(row["messages"]))

    if intrinsic == "query_clarification":
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not users:
            raise ValueError("query_clarification row has no user turn")
        convo = messages
        if row.get("documents"):
            at = users[-1] + 1
            convo = [*messages[:at], {"role": "tool", "content": _documents_payload(row["documents"])},
                     *messages[at:]]
        args = {"query": messages[users[-1]]["content"]}
        gold = json.loads(final) if final else None
        return convo, args, gold

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
        gold_raw = final if final is not None else json.dumps({"score": str(row["ground_truth"]).strip().lower()})
    elif intrinsic == "requirement_check":
        m = _REQUIREMENTS.match(instruction)
        if not m:
            raise ValueError("requirements instruction has no requirement list")
        args = {"requirements": m.group(1).strip()}
        gold_raw = final if final is not None else json.dumps({"score": str(row["ground_truth"]).strip().lower()})
    else:
        raise KeyError(f"unknown intrinsic {intrinsic!r}")
    gold = json.loads(gold_raw)
    return convo, args, gold


def external_context(intrinsic: str, convo: list[dict], args: dict, tokenizer) -> str:
    """Stock-template render of the conversation, the base's tool call, and the
    env-written tool-response opener: the prompt the adapter continues from."""
    msgs = [*convo, {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": intrinsic, "arguments": args}}],
    }, {"role": "tool", "content": _SENTINEL}]
    rendered = tokenizer.apply_chat_template(
        msgs, tools=[TOOLS[intrinsic]], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    ctx = rendered[: rendered.rindex(_SENTINEL)]
    if not ctx.endswith(OPENER):
        raise ValueError(f"unexpected tool-response opener: {ctx[-40:]!r}")
    return ctx


def build_training_row(intrinsic: str, row: dict, tokenizer) -> dict:
    convo, args, gold = split_row(intrinsic, row)
    if gold is None:
        raise ValueError(f"{intrinsic} training row has no gold tool response")
    ctx = external_context(intrinsic, convo, args, tokenizer)
    return {
        "text": ctx + json.dumps(gold, ensure_ascii=False) + EOS,
        "metadata": {"tool_name": intrinsic, "tool_formatting": "external_tool_formatting"},
    }
