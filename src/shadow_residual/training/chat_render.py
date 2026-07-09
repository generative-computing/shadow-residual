# SPDX-License-Identifier: Apache-2.0
"""Shared chat-template rendering that handles RAG ``documents`` across model
families — used by both training (``data.py``) and eval (``answerability_eval.py``)
so they render prompts identically.

Granite 4.1's chat template renders a ``documents=`` kwarg into a system message.
The Granite 4.2 pre-release moved to ChatML and its template ignores
``documents=`` entirely (it stringifies structured content and has no document
block), so RAG context silently vanishes from the prompt — fatal for an
answerability task that must ground its decision in the documents.

For ChatML-style tokenizers (detected by the presence of ``<|im_start|>`` in the
vocab) we therefore **fold the documents into a system message** using the exact
text block Granite 4.1 produces natively, then call ``apply_chat_template``
without ``documents=``. Non-ChatML tokenizers (Granite 4.1) keep the native
``documents=`` path unchanged.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Copied verbatim from Granite 4.1's chat_template.jinja (the
# documents_system_message prefix/suffix) so a folded prompt is byte-equivalent
# to what 4.1 renders natively.
_DOCS_PREFIX = (
    "You are a helpful assistant with access to the following documents. You may "
    "use one or more documents to assist with the user query.\n\nYou are given a "
    "list of documents within <documents></documents> XML tags:\n<documents>"
)
_DOCS_SUFFIX = (
    "\n</documents>\n\nWrite the response to the user's input by strictly aligning "
    "with the facts in the provided documents. If the information needed to answer "
    "the question is not available in the documents, inform the user that the "
    "question cannot be answered based on the available data."
)


def _is_chatml_tokenizer(tokenizer: "PreTrainedTokenizerBase") -> bool:
    """True for ChatML-style tokenizers (Granite 4.2), whose template ignores
    ``documents=``. Detected by ``<|im_start|>`` being a known token — the exact
    property that distinguishes the ChatML format from Granite 4.1's role-tag
    format. No version-string matching, no template probing.
    """
    tok = "<|im_start|>"
    # Primary: membership in the vocab (authoritative). Do NOT rely on
    # convert_tokens_to_ids as a fallback — for an unknown token it returns the
    # unk id (often 0), which would false-positive on non-ChatML tokenizers.
    vocab_get = getattr(tokenizer, "get_vocab", None)
    if callable(vocab_get):
        try:
            if tok in vocab_get():
                return True
        except Exception:  # pragma: no cover - defensive
            pass
    added = getattr(tokenizer, "added_tokens_encoder", None)
    if isinstance(added, dict) and tok in added:
        return True
    return False


def _build_documents_system_text(documents: list[Any]) -> str:
    """Render ``documents`` into 4.1's system-message text block.

    Each document is JSON-serialized (matching 4.1's ``document | tojson``),
    one per line, wrapped in the documents prefix/suffix.
    """
    body = "".join("\n" + json.dumps(doc) for doc in documents)
    return _DOCS_PREFIX + body + _DOCS_SUFFIX


def _fold_documents_into_messages(messages: list[dict], documents: list[Any]) -> list[dict]:
    """Return a new messages list with the documents block folded into a system
    message. If the first message is already ``system``, prepend the block to its
    content (2 blank lines between, matching 4.1's join); otherwise insert a new
    leading system message.
    """
    docs_text = _build_documents_system_text(documents)
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        head = dict(messages[0])
        existing = head.get("content") or ""
        head["content"] = (docs_text + "\n\n" + existing) if existing else docs_text
        return [head] + messages[1:]
    return [{"role": "system", "content": docs_text}] + messages


def render_chat(
    tokenizer: "PreTrainedTokenizerBase",
    messages: list[dict],
    *,
    documents: Optional[list[Any]] = None,
    tools: Optional[list[Any]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Apply the chat template, ensuring RAG ``documents`` reach the prompt.

    - Non-ChatML tokenizer (Granite 4.1): pass ``documents=`` natively — the
      template renders them (behavior unchanged from before this helper existed).
    - ChatML tokenizer (Granite 4.2): its template ignores ``documents=``, so fold
      the documents into a system message (4.1's exact block) and call the
      template without ``documents=``.
    """
    if documents and _is_chatml_tokenizer(tokenizer):
        messages = _fold_documents_into_messages(messages, documents)
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        documents=documents,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


__all__ = ["render_chat"]
