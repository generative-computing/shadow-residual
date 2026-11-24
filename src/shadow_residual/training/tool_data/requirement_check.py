# SPDX-License-Identifier: Apache-2.0
"""Requirement check as a tool.

The ``<requirements>`` instruction turn's requirement list becomes the call's
argument ``{"requirements"}``; the tool response is ``{"score": "yes" | "no"}``.
"""

from __future__ import annotations

import re

from shadow_residual.training.tool_data.renderer import make_tool
from shadow_residual.training.tool_data.scored_instruction import ScoredInstruction

TOOL = make_tool(
    "requirement_check",
    "Check whether the assistant's last response satisfies a list of "
    'requirements. Returns {"score": "yes"} if all are satisfied and '
    '{"score": "no"} otherwise.',
    {"requirements": "The requirements the assistant's last response must satisfy."},
)

_REQUIREMENTS = re.compile(r"<requirements>(.*?)\nPlease verify", re.S)


class RequirementCheck(ScoredInstruction):
    tool_definition = TOOL

    def parse_instruction(self, instruction: str) -> dict:
        m = _REQUIREMENTS.match(instruction)
        if not m:
            raise ValueError("requirements instruction has no requirement list")
        return {"requirements": m.group(1).strip()}


__all__ = ["RequirementCheck", "TOOL"]
