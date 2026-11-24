# SPDX-License-Identifier: Apache-2.0
"""Tasks whose source row ends with an instruction user turn and a yes/no score.

The instruction turn (e.g. ``<guardian>...`` or ``<requirements>...``) becomes
the tool call's arguments and is removed from the conversation; the gold is the
``{"score": "yes" | "no"}`` dict of the final assistant turn, or
``ground_truth`` for eval rows. These tasks judge the conversation itself, so
their tool rows carry no documents.
"""

from __future__ import annotations

import json
from abc import abstractmethod

from shadow_residual.training.tool_data.renderer import Row, ToolDataRenderer, final_assistant


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


__all__ = ["ScoredInstruction"]
