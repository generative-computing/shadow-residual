# SPDX-License-Identifier: Apache-2.0
"""Render SR-adapter-as-tool rows for the two ``data.tool_formatting`` modes.

An SR adapter can be exposed to the base model as a *tool*: the base emits a
``<tool_call>`` for it, the serving environment routes to the adapter, and the
adapter produces the ``<tool_response>`` payload. The adapter never writes the
assistant answer — after the tool response, the base (or another adapter)
resumes and answers.

A tool row is a stock chat row whose ``messages`` end with the trigger and the
response::

    [..., {"role": "assistant", "content": "", "tool_calls": [<one call>]},
          {"role": "tool", "content": "<payload the adapter must produce>"}]

with the called tool defined in the row's ``tools`` (and optional RAG
``documents``, injected after the last user turn as for every ChatML row). The
same row serves both modes; only where the adapter takes over differs:

- ``external_tool_formatting`` — the environment owns the template. The
  adapter's prompt ends with the env-written ``<tool_response>\\n``; it emits
  only the payload and stops on EOS (``<|im_end|>``). The environment drops the
  EOS and closes the turn with ``\\n</tool_response>\\n<|im_end|>\\n``. The
  resulting transcript is byte-identical to the stock template.
- ``autoregressive_tool_formatting`` — the adapter's prompt ends right after the
  base's ``</tool_call>``; it emits the whole tool-response turn,
  ``\\n<|im_start|>user\\n<tool_response>\\n{payload}\\n</tool_response>\\n<|im_end|>``,
  stopping on that turn's ``<|im_end|>``. The base turn's own ``<|im_end|>``
  after ``</tool_call>`` is omitted (it is the stop token, so an adapter
  trained to emit it would halt immediately) — the one off-template seam.

Granite 4.2 (ChatML, e.g. ``ibm-granite/granite-4.2-3b``) only: the seams above
are validated against the tokenizer's real template on every row, and
:func:`check_tool_formatting_support` probes them up front. See
``docs/tool_formatting.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from shadow_residual.config.training_config import ToolFormatting
from shadow_residual.training.chat_render import _is_chatml_tokenizer, render_chat

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

AUTOREGRESSIVE = ToolFormatting.AUTOREGRESSIVE.value
EXTERNAL = ToolFormatting.EXTERNAL.value

EOS = "<|im_end|>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>\n"

# Stock Granite 4.2 rendering around a tool response:
#   ...</tool_call>{_STOCK_GLUE}{payload}{_STOCK_TAIL}
_STOCK_GLUE = "\n<|im_end|>\n<|im_start|>user\n" + TOOL_RESPONSE_OPEN
_STOCK_TAIL = "\n</tool_response>\n<|im_end|>\n"
# What the autoregressive adapter emits around the payload: the stock glue minus
# the base turn's <|im_end|>, and the stock tail up to (and including) the EOS.
_AR_OPEN = "\n<|im_start|>user\n" + TOOL_RESPONSE_OPEN
_AR_CLOSE = "\n</tool_response>\n" + EOS

_SENTINEL = "__SR_TOOL_RESPONSE_SENTINEL__"
_ROW_SHAPE = (
    "a tool-formatted row's messages must end with an assistant turn carrying "
    "exactly one tool_call (the trigger), optionally followed by a tool message "
    "holding the response the adapter produces"
)


class ToolMarkers(NamedTuple):
    """Boundary markers a mode implies (set by ``train.apply_tool_formatting``)."""

    assistant_marker: str
    last_context_token: str
    last_token: str


TOOL_FORMATTING_MARKERS: dict[str, ToolMarkers] = {
    AUTOREGRESSIVE: ToolMarkers(TOOL_CALL_CLOSE, TOOL_CALL_CLOSE, EOS),
    EXTERNAL: ToolMarkers(TOOL_RESPONSE_OPEN, "\n", EOS),
}

# Vanilla rows (``data.vanilla_frac``): plain Granite 4.2 SR chat rows, thinking
# off. The adapter answers right after the empty think block.
VANILLA_MARKERS = ToolMarkers("<|im_start|>assistant\n<think></think>", "</think>", EOS)


def _mode(mode: str) -> str:
    return ToolFormatting(mode).value


def _split_trigger(
    messages: list[dict], tools: list[dict] | None
) -> tuple[list[dict], str | None]:
    """Return (conversation through the trigger turn, gold payload or None)."""
    msgs = list(messages)
    gold = None
    if msgs and msgs[-1].get("role") == "tool":
        gold = msgs.pop().get("content")
        if not isinstance(gold, str):
            raise ValueError(f"the tool response must be a string, got {type(gold).__name__}")
    trigger = msgs[-1] if msgs else {}
    calls = trigger.get("tool_calls") or []
    if trigger.get("role") != "assistant" or len(calls) != 1:
        raise ValueError(_ROW_SHAPE)
    name = (calls[0].get("function") or {}).get("name")
    defined = sorted({(t.get("function") or {}).get("name") for t in tools or []} - {None})
    if name not in defined:
        raise ValueError(
            f"the trigger tool_call names {name!r}, which is not defined in the "
            f"row's tools ({defined}); a tool-formatted row must carry the "
            "definition of the tool it calls"
        )
    return msgs, gold


def _render_contexts(
    tokenizer: "PreTrainedTokenizerBase",
    convo: list[dict],
    tools: list[dict],
    documents: list[Any] | None,
) -> tuple[str, str]:
    """Return (autoregressive prompt, external prompt) for a conversation that
    ends with the trigger turn, validating the Granite 4.2 seams."""
    rendered = render_chat(
        tokenizer,
        [*convo, {"role": "tool", "content": _SENTINEL}],
        documents=documents,
        tools=tools,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    head, sep, tail = rendered.rpartition(_SENTINEL)
    if not sep or tail != _STOCK_TAIL or not head.endswith(TOOL_CALL_CLOSE + _STOCK_GLUE):
        seam = head[-60:] + "<payload>" + tail if sep else rendered[-60:]
        raise ValueError(
            "the chat template does not render tool turns the Granite 4.2 way "
            f"(expected '...</tool_call>{_STOCK_GLUE}<payload>{_STOCK_TAIL}', "
            f"got {seam!r})"
        )
    return head[: -len(_STOCK_GLUE)], head


def render_tool_example(
    tokenizer: "PreTrainedTokenizerBase",
    messages: list[dict],
    *,
    tools: list[dict] | None,
    mode: str,
    documents: list[Any] | None = None,
) -> str:
    """Render a training row: the mode's prompt followed by the supervised span.

    The row's final ``tool`` message is the payload the adapter learns to
    produce. ``enable_thinking`` / ``instruction_as_user_message`` do not apply:
    the prompt never ends in a generation prompt, and documents always land
    after the last user turn.
    """
    mode = _mode(mode)
    convo, gold = _split_trigger(messages, tools)
    if gold is None:
        raise ValueError(f"{_ROW_SHAPE}; training rows need the tool message")
    ar_ctx, ext_ctx = _render_contexts(tokenizer, convo, tools, documents)
    if mode == EXTERNAL:
        return ext_ctx + gold + EOS
    return ar_ctx + _AR_OPEN + gold + _AR_CLOSE


def render_tool_prompt(
    tokenizer: "PreTrainedTokenizerBase",
    messages: list[dict],
    *,
    tools: list[dict] | None,
    mode: str,
    documents: list[Any] | None = None,
) -> str:
    """Render the prompt the adapter continues from in ``mode``.

    A trailing ``tool`` message (the gold response) is dropped, so a training
    or validation row renders the same prompt as its gold-free eval twin.
    """
    mode = _mode(mode)
    convo, _ = _split_trigger(messages, tools)
    ar_ctx, ext_ctx = _render_contexts(tokenizer, convo, tools, documents)
    return ext_ctx if mode == EXTERNAL else ar_ctx


def extract_tool_payload(generated: str, mode: str) -> str:
    """Return the tool-response payload from text generated after the prompt.

    ``generated`` must be decoded with special tokens kept. Everything from the
    first EOS on is dropped. In autoregressive mode the payload is unwrapped
    from the emitted tool-response turn; a generation that doesn't have that
    structure is returned as-is, so it scores as malformed rather than vanishing.
    """
    mode = _mode(mode)
    text = generated.split(EOS, 1)[0]
    if mode == AUTOREGRESSIVE:
        close = _AR_CLOSE[: -len(EOS)]
        if text.startswith(_AR_OPEN) and text.endswith(close):
            return text[len(_AR_OPEN) : -len(close)]
    return text


_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "probe",
        "description": "Probe tool.",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string", "description": "query"}},
            "required": ["q"],
        },
    },
}
_PROBE_MESSAGES = [
    {"role": "user", "content": "hi"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "probe", "arguments": {"q": "hi"}}}],
    },
]


def check_tool_formatting_support(
    tokenizer: "PreTrainedTokenizerBase", model_name: str
) -> None:
    """Raise unless ``tokenizer`` renders tool turns the way both modes need.

    Tool formatting is Granite 4.2-only: it relies on 4.2's ChatML tool-turn
    seams and on ``</tool_call>`` / ``<tool_response>`` / ``</tool_response>`` /
    ``<|im_end|>`` being single tokens. Checked by rendering a probe
    conversation through the real template, not by matching the model name.
    """
    try:
        if not _is_chatml_tokenizer(tokenizer):
            raise ValueError("its chat template is not ChatML")
        for tok in (EOS, TOOL_CALL_CLOSE, "<tool_response>", "</tool_response>"):
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{tok!r} is not a single token (got {ids})")
        _render_contexts(tokenizer, _PROBE_MESSAGES, [_PROBE_TOOL], None)
    except Exception as e:
        raise ValueError(
            f"data.tool_formatting requires a Granite 4.2 base model (ChatML tool "
            f"turns, e.g. ibm-granite/granite-4.2-3b); {model_name!r} is not "
            f"supported: {e}"
        ) from e


__all__ = [
    "AUTOREGRESSIVE",
    "EXTERNAL",
    "TOOL_FORMATTING_MARKERS",
    "ToolMarkers",
    "VANILLA_MARKERS",
    "check_tool_formatting_support",
    "extract_tool_payload",
    "render_tool_example",
    "render_tool_prompt",
]
