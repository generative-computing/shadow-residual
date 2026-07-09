# SPDX-License-Identifier: Apache-2.0
"""Shared chat-template rendering that handles RAG ``documents`` across model
families — used by both training (``data.py``) and eval (``answerability_eval.py``)
so they render prompts identically.

Granite 4.1's chat template renders a ``documents=`` kwarg into a system message.
The Granite 4.2 pre-release moved to ChatML and its template ignores
``documents=`` entirely, so RAG context silently vanishes from the prompt — fatal
for an answerability task that must ground its decision in the documents.

For ChatML-style tokenizers (detected by ``<|im_start|>`` in the vocab) we
therefore inject the retrieved documents the **agentic** way: as a ``tool``
message (a search/knowledge-base tool response) placed right after the last user
turn. Granite 4.2's template renders a ``tool`` message as a
``<tool_response>[...]</tool_response>`` block, so the documents land in the
prompt as retrieved context — matching how 4.2 natively represents RAG. Non-ChatML
tokenizers (Granite 4.1) keep the native ``documents=`` path unchanged.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


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


def _normalize_document(doc: Any, idx: int) -> dict:
    """Normalize a document into a {source, document_id, content} record for the
    agentic tool response. Accepts a plain string, a ``{"text": ...}`` dict (the
    eval path's shape), or an arbitrary dict (kept, with sensible defaults filled).
    """
    if isinstance(doc, str):
        return {"source": "knowledge_base", "document_id": str(idx), "content": doc}
    if isinstance(doc, dict):
        content = doc.get("content", doc.get("text", ""))
        return {
            "source": doc.get("source", "knowledge_base"),
            "document_id": str(doc.get("document_id", doc.get("doc_id", idx))),
            "content": content,
        }
    return {"source": "knowledge_base", "document_id": str(idx), "content": str(doc)}


def _tool_message_for_documents(documents: list[Any]) -> dict:
    """Build a ``tool`` message whose content is a JSON array of the retrieved
    documents (agentic search-tool response)."""
    payload = [_normalize_document(d, i) for i, d in enumerate(documents)]
    return {"role": "tool", "content": json.dumps(payload, indent=2)}


def _insert_documents_as_tool(messages: list[dict], documents: list[Any]) -> list[dict]:
    """Return a new messages list with a ``tool`` (search-response) message
    carrying the documents, inserted after the LAST user turn (and before any
    trailing assistant turn). If there is no user turn, append after the last
    non-assistant message. This mirrors an agentic flow: user asks → tool returns
    retrieved docs → assistant answers.
    """
    messages = list(messages)
    tool_msg = _tool_message_for_documents(documents)
    # Find the last user message index.
    last_user = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = i
    if last_user is None:
        # No user turn — insert before the first assistant turn, else append.
        first_asst = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), len(messages))
        return messages[:first_asst] + [tool_msg] + messages[first_asst:]
    return messages[: last_user + 1] + [tool_msg] + messages[last_user + 1 :]


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
    - ChatML tokenizer (Granite 4.2): its template ignores ``documents=``, so
      inject them as a ``tool`` (search-response) message after the last user
      turn; the template renders it as a ``<tool_response>`` block.
    """
    if documents and _is_chatml_tokenizer(tokenizer):
        messages = _insert_documents_as_tool(messages, documents)
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
