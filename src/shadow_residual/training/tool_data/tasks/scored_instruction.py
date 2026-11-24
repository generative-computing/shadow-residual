# SPDX-License-Identifier: Apache-2.0
"""Tasks whose source row ends with an instruction user turn and a yes/no score.

Source row: ``[...conversation, user(<instruction>), assistant('{"score": "yes" | "no"}')]``.
The instruction turn's content becomes the tool call's arguments and is removed
from the conversation; the tool response is the ``{"score": "yes" | "no"}``
dict of the final assistant turn, or ``{"score": <ground_truth>}`` for eval
rows. These tasks judge the conversation itself, so their tool rows carry no
documents (their vanilla rows keep them).

- ``guardian_core``: the ``<guardian>`` instruction's criterion and target
  become ``{"criteria", "target"}``.
- ``requirement_check``: the ``<requirements>`` instruction's requirement list
  becomes ``{"requirements"}``.
"""

from __future__ import annotations

import json
import re
from abc import abstractmethod

from shadow_residual.training.tool_data.tasks.base import Row, ToolDataRenderer, final_assistant, make_tool


class ScoredInstruction(ToolDataRenderer[Row]):
    @abstractmethod
    def parse_instruction(self, instruction: str) -> dict:
        """The tool call's arguments, lifted out of the instruction turn."""

    def to_tool_call(self, row: Row) -> list[dict]:
        messages = row["messages"]
        if not messages or messages[-1].get("role") != "user":
            raise ValueError(f"{self.name} row does not end in its instruction turn")
        arguments = self.parse_instruction(messages[-1]["content"])
        return [*messages[:-1], self.tool_call(arguments)]

    def format_response(self, row: Row) -> dict:
        final = final_assistant(row)
        if final is not None:
            return json.loads(final)
        return {"score": str(row["ground_truth"]).strip().lower()}

    def carried_fields(self, row: Row) -> dict:
        return {"ground_truth": row["ground_truth"]} if "ground_truth" in row else {}

    def vanilla_gold(self, row: Row) -> str | None:
        if row.get("ground_truth") is None:
            return None
        return json.dumps({"score": str(row["ground_truth"]).strip().lower()})


GUARDIAN_TOOL = make_tool(
    "guardian_core",
    "Judge whether a turn of the conversation so far meets a risk criterion. "
    'Returns {"score": "yes"} if it does and {"score": "no"} otherwise.',
    {
        "criteria": "The risk criterion to judge the text against.",
        "target": "Whose text to judge: the last 'user' turn or the last 'assistant' turn.",
    },
)

_GUARDIAN = re.compile(r"### Criteria: (.*?)\n\n### Scoring Schema: (.*)$", re.S)


class GuardianCore(ScoredInstruction):
    tool_definition = GUARDIAN_TOOL

    def parse_instruction(self, instruction: str) -> dict:
        m = _GUARDIAN.search(instruction)
        if not m:
            raise ValueError("guardian instruction has no criteria/schema")
        target = "user" if "last user's text" in m.group(2) else "assistant"
        return {"criteria": m.group(1).strip(), "target": target}


REQUIREMENT_TOOL = make_tool(
    "requirement_check",
    "Check whether the assistant's last response satisfies a list of "
    'requirements. Returns {"score": "yes"} if all are satisfied and '
    '{"score": "no"} otherwise.',
    {"requirements": "The requirements the assistant's last response must satisfy."},
)

_REQUIREMENTS = re.compile(r"<requirements>(.*?)\nPlease verify", re.S)


class RequirementCheck(ScoredInstruction):
    tool_definition = REQUIREMENT_TOOL

    def parse_instruction(self, instruction: str) -> dict:
        m = _REQUIREMENTS.match(instruction)
        if not m:
            raise ValueError("requirements instruction has no requirement list")
        return {"requirements": m.group(1).strip()}


__all__ = ["GUARDIAN_TOOL", "GuardianCore", "REQUIREMENT_TOOL", "RequirementCheck", "ScoredInstruction"]
