# SPDX-License-Identifier: Apache-2.0
"""SR-adapter-as-tool data: mode rendering + task converters.

- :mod:`.formatting` renders tool rows for the two ``data.tool_formatting``
  modes (training text, generation prompts, payload extraction) and checks the
  tokenizer supports them (Granite 4.2 only).
- :mod:`.tasks` holds the converters: :class:`Row` and
  :class:`ToolDataRenderer` (:mod:`.tasks.base`), the per-task interface that
  converts task source rows into tool rows and vanilla rows, and the shipped
  example tasks that implement it.
- :mod:`.convert` is the task registry (used on load when ``data.tool_task`` is
  set).

See ``docs/tool_formatting.md``.
"""

from shadow_residual.training.tool_data.formatting import (
    AUTOREGRESSIVE,
    EXTERNAL,
    TOOL_FORMATTING_MARKERS,
    VANILLA_MARKERS,
    ToolMarkers,
    check_tool_formatting_support,
    extract_tool_payload,
    render_tool_example,
    render_tool_prompt,
)
from shadow_residual.training.tool_data.tasks.base import Row, ToolDataRenderer

__all__ = [
    "AUTOREGRESSIVE",
    "EXTERNAL",
    "Row",
    "TOOL_FORMATTING_MARKERS",
    "ToolDataRenderer",
    "ToolMarkers",
    "VANILLA_MARKERS",
    "check_tool_formatting_support",
    "extract_tool_payload",
    "render_tool_example",
    "render_tool_prompt",
]
