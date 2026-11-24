# SPDX-License-Identifier: Apache-2.0
"""Task converters: the :class:`ToolDataRenderer` base class and the shipped
example tasks (one renderer per ``data.tool_task``; registered in
:mod:`shadow_residual.training.tool_data.convert`)."""

from shadow_residual.training.tool_data.tasks.answerability import Answerability
from shadow_residual.training.tool_data.tasks.base import Row, ToolDataRenderer, final_assistant, make_tool
from shadow_residual.training.tool_data.tasks.query_clarification import QueryClarification
from shadow_residual.training.tool_data.tasks.scored_instruction import GuardianCore, RequirementCheck, ScoredInstruction

__all__ = [
    "Answerability",
    "GuardianCore",
    "QueryClarification",
    "RequirementCheck",
    "Row",
    "ScoredInstruction",
    "ToolDataRenderer",
    "final_assistant",
    "make_tool",
]
