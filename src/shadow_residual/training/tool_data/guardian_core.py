# SPDX-License-Identifier: Apache-2.0
"""Guardian (risk criterion judge) as a tool.

The ``<guardian>`` instruction turn's criterion and target become the call's
arguments ``{"criteria", "target"}``; the tool response is ``{"score": "yes" | "no"}``.
"""

from __future__ import annotations

import re

from shadow_residual.training.tool_data.renderer import make_tool
from shadow_residual.training.tool_data.scored_instruction import ScoredInstruction

TOOL = make_tool(
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
    tool_definition = TOOL

    def parse_instruction(self, instruction: str) -> dict:
        m = _GUARDIAN.search(instruction)
        if not m:
            raise ValueError("guardian instruction has no criteria/schema")
        target = "user" if "last user's text" in m.group(2) else "assistant"
        return {"criteria": m.group(1).strip(), "target": target}


__all__ = ["GuardianCore", "TOOL"]
