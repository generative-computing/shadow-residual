# SPDX-License-Identifier: Apache-2.0
"""ToolDataRenderer: the per-task interface, its shared row assembly, and the registry.

A test-only renderer with its own Row subclass exercises the contract a new task
relies on; the shipped tasks' conversions are covered in test_tool_data.py.
"""

from __future__ import annotations

import json

import pytest

from shadow_residual.training.tool_data import Row, ToolDataRenderer
from shadow_residual.training.tool_data.convert import RENDERERS, TASKS, convert, renderer, to_vanilla
from shadow_residual.training.tool_data.renderer import final_assistant, make_tool

G42 = "ibm-granite/granite-4.2-3b"


class EchoRow(Row, total=False):
    tool_input: dict[str, str]


ECHO_TOOL = make_tool("echo", "Echo the input.", {"text": "Text to echo."})


class Echo(ToolDataRenderer[EchoRow]):
    """Arguments come from the row's own ``tool_input`` key; gold is the final turn's text."""

    tool_definition = ECHO_TOOL

    def __init__(self):
        self.seen: list[EchoRow] = []

    def keep(self, row: EchoRow) -> bool:
        return row.get("ground_truth") != "skip"

    def to_tool_call(self, row: EchoRow) -> list[dict]:
        self.seen.append(row)
        return [*row["messages"], self.tool_call(row["tool_input"])]

    def format_response(self, row: EchoRow) -> dict | None:
        final = final_assistant(row)
        return {"echo": final} if final is not None else None

    def vanilla_gold(self, row: EchoRow) -> str | None:
        return row.get("ground_truth")


def _row(**extra) -> EchoRow:
    return {
        "messages": [{"role": "user", "content": "say hi"}, {"role": "assistant", "content": "hi ✓"}],
        "tool_input": {"text": "hi"},
        **extra,
    }


# --- shared tool-row assembly ---------------------------------------------------


def test_tool_row_assembly():
    echo = Echo()
    out = echo.to_tool_row(_row(documents=[{"text": "d"}], ground_truth="hi"))
    assert out["messages"] == [
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "echo", "arguments": {"text": "hi"}}}]},
        {"role": "tool", "content": '{"echo": "hi ✓"}'},
    ]
    assert out["tools"] == [ECHO_TOOL]
    assert list(out) == ["messages", "tools", "documents", "ground_truth"]
    assert "tool_input" not in out


def test_to_tool_call_gets_the_row_without_its_gold_turn():
    echo = Echo()
    src = _row()
    echo.to_tool_row(src)
    assert echo.seen[-1]["messages"] == [{"role": "user", "content": "say hi"}]
    assert echo.seen[-1]["tool_input"] == {"text": "hi"}
    assert len(src["messages"]) == 2


def test_eval_shape_has_no_tool_message():
    out = Echo().to_tool_row(_row(), include_response=False)
    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][-1]["tool_calls"]


def test_missing_gold_is_an_error_only_with_the_response():
    src = {"messages": [{"role": "user", "content": "say hi"}], "tool_input": {"text": "hi"}}
    assert Echo().to_tool_row(src, include_response=False)["messages"][-1]["role"] == "assistant"
    with pytest.raises(ValueError, match="echo row has no gold tool response"):
        Echo().to_tool_row(src)


def test_malformed_gold_fails_in_both_shapes():
    class Strict(Echo):
        def format_response(self, row):
            return json.loads(final_assistant(row))

    for include_response in (True, False):
        with pytest.raises(json.JSONDecodeError):
            Strict().to_tool_row(_row(), include_response=include_response)


def test_keep_drops_the_row():
    assert Echo().to_tool_row(_row(ground_truth="skip")) is None


def test_carried_fields_default():
    echo = Echo()
    assert echo.carried_fields(_row()) == {}
    assert echo.carried_fields(_row(documents=[])) == {}
    assert echo.carried_fields(_row(ground_truth=None)) == {"ground_truth": None}


# --- shared vanilla-row assembly --------------------------------------------------


def test_vanilla_row_is_the_source_row():
    assert Echo().to_vanilla_row(_row(documents=[{"text": "d"}], ground_truth="x")) == {
        "messages": _row()["messages"], "documents": [{"text": "d"}]}


def test_vanilla_gold_fills_a_missing_gold_turn():
    src = {"messages": [{"role": "user", "content": "say hi"}], "ground_truth": "hi"}
    assert Echo().to_vanilla_row(src)["messages"][-1] == {"role": "assistant", "content": "hi"}
    with pytest.raises(ValueError, match="a vanilla echo row must end with the gold assistant turn"):
        Echo().to_vanilla_row({"messages": [{"role": "user", "content": "say hi"}]})


# --- base-class contract ----------------------------------------------------------


def test_name_comes_from_the_tool_definition():
    assert Echo().name == "echo"


def test_required_hooks_are_abstract():
    class Incomplete(ToolDataRenderer[Row]):
        tool_definition = ECHO_TOOL

        def to_tool_call(self, row):
            return []

    with pytest.raises(TypeError, match="format_response"):
        Incomplete()


def test_make_tool_schema():
    assert ECHO_TOOL == {"type": "function", "function": {
        "name": "echo", "description": "Echo the input.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to echo."}},
                       "required": ["text"]}}}


def test_final_assistant():
    assert final_assistant(_row()) == "hi ✓"
    assert final_assistant({"messages": [{"role": "user", "content": "q"}]}) is None
    assert final_assistant({"messages": []}) is None


# --- registry -----------------------------------------------------------------------


def test_registry():
    assert TASKS == ("answerability", "guardian_core", "requirement_check", "query_clarification")
    for name, r in RENDERERS.items():
        assert isinstance(r, ToolDataRenderer)
        assert r.name == name == r.tool_definition["function"]["name"]
    with pytest.raises(ValueError, match="unknown tool task 'nope'"):
        renderer("nope")
    with pytest.raises(ValueError, match="unknown tool task"):
        convert("nope", {"messages": []}, include_response=True)
    with pytest.raises(ValueError, match="unknown tool task"):
        to_vanilla("nope", {"messages": []})


@pytest.mark.parametrize("task, keys", [
    ("answerability", ["messages", "tools", "ground_truth", "documents"]),
    ("query_clarification", ["messages", "tools", "documents", "ground_truth"]),
    ("guardian_core", ["messages", "tools", "ground_truth"]),
])
def test_tool_row_key_order(task, keys):
    from tests.training.test_tool_data import ADAPTER_SRC, ANS_SRC
    src = ANS_SRC[0] if task == "answerability" else {**ADAPTER_SRC[task], "documents": [{"text": "d"}],
                                                      "ground_truth": "g"}
    assert list(convert(task, src, include_response=True)) == keys


# --- a new task renders through the shared Granite 4.2 path -------------------------


@pytest.mark.parametrize("mode", ["external_tool_formatting", "autoregressive_tool_formatting"])
def test_custom_task_renders(mode):
    from tests._tokenizers import load_tokenizer
    from shadow_residual.training.tool_data import render_tool_example, render_tool_prompt
    tok = load_tokenizer(G42)
    row = Echo().to_tool_row(_row())
    text = render_tool_example(tok, row["messages"], tools=row["tools"], mode=mode)
    prompt = render_tool_prompt(tok, row["messages"], tools=row["tools"], mode=mode)
    assert text.startswith(prompt)
    assert '{"echo": "hi ✓"}' in text[len(prompt):]
