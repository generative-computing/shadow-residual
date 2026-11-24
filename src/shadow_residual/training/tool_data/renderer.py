# SPDX-License-Identifier: Apache-2.0
"""Task source rows -> SR-adapter-as-tool rows: the per-task renderer interface.

A task subclasses :class:`ToolDataRenderer`, sets ``tool_definition`` and
implements :meth:`~ToolDataRenderer.to_tool_call` and
:meth:`~ToolDataRenderer.format_response`; the other hooks have defaults. The
base class assembles the tool row and the vanilla row, so a task never builds
either itself.

A task that reads extra source-row keys declares them on a :class:`Row`
subclass and parametrizes the renderer with it (``ToolDataRenderer[MyRow]``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar


class Row(TypedDict, total=False):
    """One JSONL line: a task source row, or the tool / vanilla row built from it."""

    messages: list[dict]
    documents: list[Any]
    tools: list[dict]
    ground_truth: Any


R = TypeVar("R", bound=Row)


def make_tool(name: str, description: str, params: dict[str, str]) -> dict:
    """An OpenAI-style function schema whose parameters are all required strings."""
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


def final_assistant(row: Row) -> str | None:
    """Content of the row's final turn if it is an assistant turn (the gold), else None."""
    messages = row["messages"]
    if messages and messages[-1].get("role") == "assistant":
        return messages[-1].get("content")
    return None


class ToolDataRenderer(ABC, Generic[R]):
    """Per-task conversion of source rows into tool rows and vanilla rows."""

    tool_definition: dict

    @property
    def name(self) -> str:
        """The task name: the tool's name, which the trigger tool call calls."""
        return self.tool_definition["function"]["name"]

    def keep(self, row: R) -> bool:
        """False drops the row as unusable (counted in the loader's log)."""
        return True

    @abstractmethod
    def to_tool_call(self, row: R) -> list[dict]:
        """The conversation the base saw, ending with the trigger turn (see
        :meth:`tool_call`). ``row["messages"]`` arrives without the gold
        assistant turn. Raise ``ValueError`` for a malformed row."""

    @abstractmethod
    def format_response(self, row: R) -> dict | None:
        """The gold tool response, from the row's final assistant turn
        (:func:`final_assistant`) or ``ground_truth``; None if the row carries no
        gold (allowed for eval rows only)."""

    def carried_fields(self, row: R) -> dict:
        """Source-row fields the tool row keeps after ``messages`` and ``tools``:
        ``documents`` if non-empty, and ``ground_truth`` (for the scorer) if present."""
        out = {}
        if row.get("documents"):
            out["documents"] = row["documents"]
        if "ground_truth" in row:
            out["ground_truth"] = row["ground_truth"]
        return out

    def vanilla_gold(self, row: R) -> str | None:
        """The gold assistant turn of a vanilla row that has none, built from
        ``ground_truth``; None leaves such a row invalid."""
        return None

    def tool_call(self, arguments: dict) -> dict:
        """The trigger turn: an empty assistant turn carrying one call of this tool."""
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": self.name, "arguments": arguments}}],
        }

    def to_tool_row(self, row: R, *, include_response: bool = True) -> Row | None:
        """The tool row: the trigger conversation, then (if ``include_response``)
        the gold ``tool`` message; None if :meth:`keep` drops the row.

        ``include_response=False`` is the eval shape (generation prompts). The
        gold is still read, so a malformed one fails in either shape.
        """
        if not self.keep(row):
            return None
        messages = row["messages"]
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        messages = self.to_tool_call({**row, "messages": list(messages)})
        response = self.format_response(row)
        if include_response:
            if response is None:
                raise ValueError(f"{self.name} row has no gold tool response")
            messages.append({"role": "tool", "content": json.dumps(response, ensure_ascii=False)})
        return {"messages": messages, "tools": [self.tool_definition], **self.carried_fields(row)}

    def to_vanilla_row(self, row: R) -> Row:
        """The source row as a plain chat row (``data.vanilla_frac``): ``messages``
        ending with the gold assistant turn, plus ``documents`` if any."""
        messages = list(row["messages"])
        if messages and messages[-1].get("role") != "assistant":
            gold = self.vanilla_gold(row)
            if gold is not None:
                messages.append({"role": "assistant", "content": gold})
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"a vanilla {self.name} row must end with the gold assistant turn")
        out: Row = {"messages": messages}
        if row.get("documents"):
            out["documents"] = row["documents"]
        return out


__all__ = ["R", "Row", "ToolDataRenderer", "final_assistant", "make_tool"]
