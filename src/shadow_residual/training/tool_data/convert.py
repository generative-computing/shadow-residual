# SPDX-License-Identifier: Apache-2.0
"""Dispatch a task source row to its tool-row converter.

Used by the trainer / generator when ``data.tool_task`` is set (conversion on
load, no intermediate file).
"""

from __future__ import annotations

from shadow_residual.training.tool_data import answerability, intrinsics

TASKS = ("answerability", *intrinsics.INTRINSICS)


def tool_name(task: str) -> str:
    if task == "answerability":
        return answerability.ANSWERABILITY_TOOL["function"]["name"]
    return intrinsics.TOOLS[task]["function"]["name"]


def convert(task: str, row: dict, *, include_response: bool) -> dict | None:
    """Convert one source row; ``None`` if the converter drops it as unusable.

    Raises if ``row`` is already a tool row for ``task`` (it defines the task's
    tool), so a converted file can't be converted a second time.
    """
    if task not in TASKS:
        raise ValueError(f"unknown tool task {task!r}; known: {TASKS}")
    name = tool_name(task)
    if any((t.get("function") or {}).get("name") == name for t in row.get("tools") or []):
        raise ValueError(
            f"row already defines the {name!r} tool, so it is a converted tool row, "
            "not a source row; unset data.tool_task (or --tool-task) to train on it as-is"
        )
    if task == "answerability":
        return answerability.to_tool_row(row, include_response=include_response)
    return intrinsics.to_tool_row(task, row, include_response=include_response)


def to_vanilla(task: str, row: dict) -> dict:
    """The source row as a plain chat training row (``data.vanilla_frac``):
    ``messages`` ending with the gold assistant turn, plus ``documents``."""
    if task not in TASKS:
        raise ValueError(f"unknown tool task {task!r}; known: {TASKS}")
    if task == "answerability":
        return answerability.to_vanilla_row(row)
    return intrinsics.to_vanilla_row(task, row)


__all__ = ["TASKS", "convert", "to_vanilla", "tool_name"]
