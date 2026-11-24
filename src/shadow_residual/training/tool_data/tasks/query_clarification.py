# SPDX-License-Identifier: Apache-2.0
"""Query clarification as a tool.

The user's latest query becomes the call's argument ``{"query"}`` (the turn
stays in the conversation); the tool response is ``{"clarification": "CLEAR" |
<question>}``. The tool row keeps the row's documents. Eval rows carry no gold dict, so they convert only without the
response.
"""

from __future__ import annotations

import json

from shadow_residual.training.tool_data.tasks.base import Row, ToolDataRenderer, final_assistant, make_tool

TOOL = make_tool(
    "query_clarification",
    "Decide whether the user's latest query is clear enough to answer from "
    'the conversation and documents. Returns {"clarification": "CLEAR"} if '
    "it is, otherwise a clarifying question to ask the user.",
    {"query": "The user's latest query."},
)


class QueryClarification(ToolDataRenderer[Row]):
    tool_definition = TOOL

    def to_tool_call(self, row: Row) -> list[dict]:
        messages = row["messages"]
        users = [m for m in messages if m.get("role") == "user"]
        if not users:
            raise ValueError("query_clarification row has no user turn")
        return [*messages, self.tool_call({"query": users[-1]["content"]})]

    def format_response(self, row: Row) -> dict | None:
        final = final_assistant(row)
        return json.loads(final) if final else None


__all__ = ["QueryClarification", "TOOL"]
