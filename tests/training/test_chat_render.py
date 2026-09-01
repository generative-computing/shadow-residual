# SPDX-License-Identifier: Apache-2.0
"""Tests for render_chat: RAG documents must reach the prompt on both Granite
4.1 (native documents=) and Granite 4.2 (ChatML → injected as a tool response)."""

from __future__ import annotations

import json

import pytest

from shadow_residual.training.chat_render import render_chat


class _FakeTokenizer:
    """Minimal tokenizer stub with a controllable vocab + a template mimicking
    each family's document handling.

    - chatml=True: <|im_start|> is in the vocab AND apply_chat_template IGNORES
      the documents= kwarg (like Granite 4.2). It renders a `tool` message as a
      <tool_response>…</tool_response> block (like 4.2's template) and otherwise
      concatenates message contents.
    - chatml=False: no <|im_start|>; apply_chat_template RENDERS documents=
      natively (like Granite 4.1).
    """

    def __init__(self, chatml: bool):
        self.chatml = chatml
        self._vocab = {"<|im_start|>": 100256, "<|im_end|>": 100257} if chatml else {}

    def get_vocab(self):
        return dict(self._vocab)

    def apply_chat_template(self, messages, *, documents=None, tools=None,
                            tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        parts = []
        if not self.chatml and documents:  # 4.1-style native rendering
            for d in documents:
                parts.append("NATIVE_DOC:" + str(d))
        for m in messages:
            role = m["role"]
            content = m.get("content", "")
            if self.chatml and role == "tool":
                parts.append(f"[tool] <tool_response>\n{content}\n</tool_response>")
            else:
                parts.append(f"[{role}] {content}")
        if add_generation_prompt:
            parts.append("[assistant]")
        return "\n".join(parts)


DOCS = [{"text": "SENTINEL_DOC_ONE"}, {"text": "SENTINEL_DOC_TWO"}]
MSGS = [{"role": "user", "content": "Is it answerable?"}]


def test_native_documents_path_unchanged():
    """Non-ChatML (4.1): documents passed natively; no tool message injected."""
    tok = _FakeTokenizer(chatml=False)
    out = render_chat(tok, MSGS, documents=DOCS, add_generation_prompt=True)
    assert "NATIVE_DOC:" in out
    assert "SENTINEL_DOC_ONE" in out and "SENTINEL_DOC_TWO" in out
    assert "[tool]" not in out
    assert "<tool_response>" not in out


def test_documents_injected_as_tool_for_chatml():
    """ChatML (4.2): documents injected as a tool (search-response) message."""
    tok = _FakeTokenizer(chatml=True)
    out = render_chat(tok, MSGS, documents=DOCS, add_generation_prompt=True)
    assert "[tool]" in out and "<tool_response>" in out
    assert "SENTINEL_DOC_ONE" in out and "SENTINEL_DOC_TWO" in out
    # Documents serialized with the agentic record shape.
    assert '"source": "knowledge_base"' in out
    assert '"content": "SENTINEL_DOC_ONE"' in out
    # Tool response comes AFTER the user turn (user asks → tool returns → assistant).
    assert out.index("[user]") < out.index("[tool]")


def test_tool_message_after_last_user_turn():
    """The tool response is inserted after the last user turn, before assistant."""
    tok = _FakeTokenizer(chatml=True)
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = render_chat(tok, msgs, documents=DOCS)
    # tool block sits between the LAST user turn (q2) and the final assistant (a2)
    assert out.index("q2") < out.index("[tool]") < out.index("a2")


def test_document_content_key_variants():
    """Accepts str docs and {'content'/'text'/doc_id} dicts; normalizes shape."""
    tok = _FakeTokenizer(chatml=True)
    docs = ["PLAIN_STRING_DOC", {"content": "CONTENT_KEY_DOC", "document_id": "994"}]
    out = render_chat(tok, MSGS, documents=docs)
    assert "PLAIN_STRING_DOC" in out and "CONTENT_KEY_DOC" in out
    assert '"document_id": "994"' in out


@pytest.mark.parametrize("chatml", [True, False])
def test_no_documents_noop(chatml):
    """No documents → no tool message, no native doc rendering, either family."""
    tok = _FakeTokenizer(chatml=chatml)
    out = render_chat(tok, MSGS, documents=None, add_generation_prompt=False)
    assert "[tool]" not in out
    assert "<tool_response>" not in out
    assert "NATIVE_DOC:" not in out
    assert "[user] Is it answerable?" in out


# ---------- instruction_as_user_message placement ----------

# Judge/guardian shape: question → answer being judged → judge instruction → label.
JUDGE_MSGS = [
    {"role": "user", "content": "q1_question"},
    {"role": "assistant", "content": "a1_answer_being_judged"},
    {"role": "user", "content": "q2_judge_instruction"},
    {"role": "assistant", "content": "a2_label"},
]


def test_instruction_as_user_message_inserts_before_first_assistant():
    """flag=True: docs land before the FIRST assistant turn (the answer being
    judged), not after the last user turn."""
    tok = _FakeTokenizer(chatml=True)
    out = render_chat(tok, JUDGE_MSGS, documents=DOCS,
                      instruction_as_user_message=True)
    # tool block sits between the first user turn and the first assistant answer.
    assert out.index("q1_question") < out.index("[tool]") < out.index("a1_answer_being_judged")
    # and BEFORE the judge instruction (definitely not at the end).
    assert out.index("[tool]") < out.index("q2_judge_instruction")


def test_instruction_as_user_message_default_unchanged():
    """flag omitted: docs land after the LAST user turn (the judge instruction),
    i.e. just before the final label — the pre-existing behavior."""
    tok = _FakeTokenizer(chatml=True)
    out = render_chat(tok, JUDGE_MSGS, documents=DOCS)
    # tool block sits after the judge instruction, before the final label.
    assert out.index("q2_judge_instruction") < out.index("[tool]") < out.index("a2_label")


def test_instruction_as_user_message_no_assistant_falls_back():
    """flag=True but no assistant turn (e.g. eval prompt): fall back to
    after-last-user placement without crashing."""
    tok = _FakeTokenizer(chatml=True)
    msgs = [{"role": "user", "content": "only_user_turn"}]
    out = render_chat(tok, msgs, documents=DOCS, add_generation_prompt=True,
                      instruction_as_user_message=True)
    assert "[tool]" in out and "<tool_response>" in out
    assert out.index("only_user_turn") < out.index("[tool]")
