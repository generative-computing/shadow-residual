# SPDX-License-Identifier: Apache-2.0
"""The task registry: dispatch a task source row to its renderer.

Used by the trainer / generator when ``data.tool_task`` is set (conversion on
load, no intermediate file).
"""

from __future__ import annotations

from shadow_residual.training.tool_data.answerability import Answerability
from shadow_residual.training.tool_data.guardian_core import GuardianCore
from shadow_residual.training.tool_data.query_clarification import QueryClarification
from shadow_residual.training.tool_data.renderer import Row, ToolDataRenderer
from shadow_residual.training.tool_data.requirement_check import RequirementCheck

RENDERERS: dict[str, ToolDataRenderer] = {
    r.name: r for r in (Answerability(), GuardianCore(), RequirementCheck(), QueryClarification())
}
TASKS = tuple(RENDERERS)


def renderer(task: str) -> ToolDataRenderer:
    if task not in RENDERERS:
        raise ValueError(f"unknown tool task {task!r}; known: {TASKS}")
    return RENDERERS[task]


def tool_name(task: str) -> str:
    return RENDERERS[task].name


def convert(task: str, row: Row, *, include_response: bool) -> Row | None:
    """Convert one source row; ``None`` if the renderer drops it as unusable.

    Raises if ``row`` is already a tool row for ``task`` (it defines the task's
    tool), so a converted file can't be converted a second time.
    """
    r = renderer(task)
    if any((t.get("function") or {}).get("name") == r.name for t in row.get("tools") or []):
        raise ValueError(
            f"row already defines the {r.name!r} tool, so it is a converted tool row, "
            "not a source row; unset data.tool_task (or --tool-task) to train on it as-is"
        )
    return r.to_tool_row(row, include_response=include_response)


def to_vanilla(task: str, row: Row) -> Row:
    """The source row as a plain chat training row (``data.vanilla_frac``):
    ``messages`` ending with the gold assistant turn, plus ``documents``."""
    return renderer(task).to_vanilla_row(row)


__all__ = ["RENDERERS", "TASKS", "convert", "renderer", "to_vanilla", "tool_name"]
