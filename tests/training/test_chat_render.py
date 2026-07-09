# SPDX-License-Identifier: Apache-2.0
"""Tests for render_chat: RAG documents must reach the prompt on both Granite
4.1 (native documents=) and Granite 4.2 (ChatML, documents= ignored → folded)."""

from __future__ import annotations

import pytest

from shadow_residual.training.chat_render import render_chat


class _FakeTokenizer:
    """Minimal tokenizer stub with a controllable vocab + a template that mimics
    each family's document handling.

    - chatml=True: <|im_start|> is in the vocab AND apply_chat_template IGNORES
      the documents= kwarg (like Granite 4.2). It just concatenates message
      contents — so any document text must already be inside a message.
    - chatml=False: no <|im_start|>; apply_chat_template RENDERS documents= into
      the output (like Granite 4.1's native path).
    """

    def __init__(self, chatml: bool):
        self.chatml = chatml
        self._vocab = {"<|im_start|>": 100256, "<|im_end|>": 100257} if chatml else {}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, 0)

    def apply_chat_template(self, messages, *, documents=None, tools=None,
                            tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        parts = []
        # Non-ChatML (4.1-style): render documents= natively.
        if not self.chatml and documents:
            for d in documents:
                parts.append("NATIVE_DOC:" + str(d))
        for m in messages:
            parts.append(f"[{m['role']}] {m.get('content','')}")
        if add_generation_prompt:
            parts.append("[assistant]")
        return "\n".join(parts)


DOCS = [{"text": "SENTINEL_DOC_ONE"}, {"text": "SENTINEL_DOC_TWO"}]
MSGS = [{"role": "user", "content": "Is it answerable?"}]


def test_native_documents_path_unchanged():
    """Non-ChatML (4.1): documents passed natively, not folded into a system msg."""
    tok = _FakeTokenizer(chatml=False)
    out = render_chat(tok, MSGS, documents=DOCS, add_generation_prompt=True)
    # Rendered via the native documents= path…
    assert "NATIVE_DOC:" in out
    assert "SENTINEL_DOC_ONE" in out and "SENTINEL_DOC_TWO" in out
    # …and NOT via a folded <documents> system block.
    assert "<documents>" not in out
    assert "[system]" not in out


def test_fold_documents_for_chatml():
    """ChatML (4.2): documents folded into a system message (template ignores documents=)."""
    tok = _FakeTokenizer(chatml=True)
    out = render_chat(tok, MSGS, documents=DOCS, add_generation_prompt=True)
    # The ChatML template drops documents=, so folding must put the doc text +
    # the 4.1-style <documents> block into a system message.
    assert "[system]" in out
    assert "<documents>" in out and "</documents>" in out
    assert "SENTINEL_DOC_ONE" in out and "SENTINEL_DOC_TWO" in out
    # docs serialized as JSON (matching 4.1's document|tojson)
    assert '"text": "SENTINEL_DOC_ONE"' in out or '"text":"SENTINEL_DOC_ONE"' in out


def test_fold_prepends_to_existing_system_message():
    """If a system message exists, the docs block is prepended to its content."""
    tok = _FakeTokenizer(chatml=True)
    msgs = [{"role": "system", "content": "ORIGINAL_SYS"}, {"role": "user", "content": "q"}]
    out = render_chat(tok, msgs, documents=DOCS)
    assert "<documents>" in out and "ORIGINAL_SYS" in out
    # exactly one system message (block merged into the existing one)
    assert out.count("[system]") == 1


@pytest.mark.parametrize("chatml", [True, False])
def test_no_documents_noop(chatml):
    """No documents → no folding, no system block injected, on either family."""
    tok = _FakeTokenizer(chatml=chatml)
    out = render_chat(tok, MSGS, documents=None, add_generation_prompt=False)
    assert "<documents>" not in out
    assert "[system]" not in out
    assert "[user] Is it answerable?" in out
