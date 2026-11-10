# SPDX-License-Identifier: Apache-2.0
"""SR-adapter-as-tool data: mode rendering + task converters.

- :mod:`.formatting` renders tool rows for the two ``data.tool_formatting``
  modes (training text, generation prompts, payload extraction) and checks the
  tokenizer supports them (Granite 4.2 only).
- :mod:`.answerability` / :mod:`.intrinsics` convert task source rows into
  tool rows; :mod:`.convert` dispatches by task (used on load when
  ``data.tool_task`` is set); ``python -m shadow_residual.training.tool_data``
  is the CLI.

See ``docs/tool_formatting.md``.
"""

from shadow_residual.training.tool_data.formatting import (
    AUTOREGRESSIVE,
    EXTERNAL,
    TOOL_FORMATTING_MARKERS,
    ToolMarkers,
    check_tool_formatting_support,
    extract_tool_payload,
    render_tool_example,
    render_tool_prompt,
)

__all__ = [
    "AUTOREGRESSIVE",
    "EXTERNAL",
    "TOOL_FORMATTING_MARKERS",
    "ToolMarkers",
    "check_tool_formatting_support",
    "extract_tool_payload",
    "render_tool_example",
    "render_tool_prompt",
]
