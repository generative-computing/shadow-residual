# SPDX-License-Identifier: Apache-2.0
"""Data utilities: JSONL loader, response-only collator, boundary validator.

Two response-only masking paths are available, selected by train.py at runtime:

  1. SFTConfig(assistant_only_loss=True) — TRL handles masking via
     {% generation %} chat-template markers. Used when the model's chat
     template supports them.
  2. ResponseOnlyCollator (this module) — hand-rolled, scans for an
     assistant-marker token sequence and masks everything before it.
     Used when the chat template lacks the {% generation %} markers, or
     when the running TRL version lacks DataCollatorForCompletionOnlyLM
     (it was removed in TRL v1.0). TRL-version-independent.

Expected JSONL row shape:
    {
        "messages":  [{"role": "user", "content": "..."},
                      {"role": "assistant", "content": "..."}],
        "tools":     [...],   # OPTIONAL — passed verbatim to apply_chat_template
        "documents": [...],   # OPTIONAL — passed verbatim to apply_chat_template
    }

The granite chat template (and any other template that supports them)
renders ``tools`` and ``documents`` via dedicated Jinja blocks; we do
NOT reshape these fields locally. Whatever shape the template expects
is what the row must contain.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from shadow_residual.training.chat_render import render_chat

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _last_non_whitespace_token(
    ids: list[int], tokenizer: "PreTrainedTokenizerBase"
) -> int | None:
    """Return the last token id in ``ids`` that decodes to something other than
    pure whitespace, or None if the list is empty or all-whitespace.

    Used to decide whether a row already ends with the ``last_token`` marker:
    Granite's ChatML template appends a trailing newline after ``<|im_end|>``, so
    the marker is the last *meaningful* token even though the literal last token
    is the newline.
    """
    for tok_id in reversed(ids):
        if tokenizer.decode([tok_id]).strip():
            return tok_id
    return None


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the start index of `needle` in `haystack`, or -1 if absent."""
    if not needle or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if list(haystack[i : i + len(needle)]) == list(needle):
            return i
    return -1


def _find_last_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the start index of the LAST occurrence of `needle` in `haystack`, or -1.

    Multi-turn RAG conversations contain several assistant turns; only the
    final turn carries the supervised label (the answerability classification,
    or whatever short answer the adapter is being trained to produce).
    Matching the first marker would unmask earlier assistant turns and train
    the model to also generate the long factual content of those turns — a
    different and much harder task than the actual classification. Always
    scan from the end.
    """
    if not needle or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[i : i + len(needle)]) == list(needle):
            return i
    return -1


def load_jsonl_dataset(
    path: str | Path,
    tokenizer: "PreTrainedTokenizerBase",
    enable_thinking: bool = False,
    last_token: str | None = None,
) -> "Dataset":
    """Load a JSONL file and apply the model's chat template to each row.

    Each row must have a ``messages`` field. Optional ``tools`` and
    ``documents`` fields, when present, are forwarded verbatim as
    ``tools=`` / ``documents=`` kwargs to ``apply_chat_template`` — the
    template (granite's, in particular) is responsible for rendering
    them; we do not reshape.

    ``enable_thinking`` is forwarded to ``apply_chat_template`` to toggle the
    model's thinking/reasoning trace for templates that support it (Granite).
    Templates that don't understand the kwarg ignore it, so the default
    (False) is a no-op.

    ``last_token`` (when set) is an end-of-completion marker: any rendered row
    that doesn't already carry the marker at the end gets it appended (at the
    string level, so it survives whichever masking path the trainer picks).
    "At the end" ignores trailing whitespace tokens: Granite's ChatML template
    emits ``<|im_end|>\n`` (a trailing newline after the terminator), so the
    literal last token is the newline, not the marker — checking only the very
    last token id would wrongly re-append the marker and produce a doubled
    ``<|im_end|>\n<|im_end|>`` tail on every row. Rows already ending in the
    marker (with or without trailing whitespace) are left unchanged. The string
    must encode to exactly one token id (enforced by the caller).

    Returns a `datasets.Dataset` with a single ``text`` column containing the
    chat-templated string. SFTTrainer reads the ``text`` field directly.
    """
    from datasets import Dataset

    last_token_id: int | None = None
    if last_token is not None:
        ids = tokenizer.encode(last_token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"last_token {last_token!r} must encode to exactly one token id, "
                f"got {ids}."
            )
        last_token_id = ids[0]

    rows: list[dict[str, str]] = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"{path}: row is missing 'messages' field: {obj!r}")
            # render_chat folds documents into a system message for ChatML
            # tokenizers (Granite 4.2, whose template ignores documents=) and
            # passes them natively otherwise (Granite 4.1). See chat_render.
            text = render_chat(
                tokenizer,
                obj["messages"],
                documents=obj.get("documents"),
                tools=obj.get("tools"),
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            if last_token_id is not None:
                # Append the end-of-completion marker unless the row already
                # ends with it. Scan past trailing whitespace-only tokens before
                # comparing: Granite's ChatML template ends rows with
                # "<|im_end|>\n", so the literal last token is the newline and a
                # naive row_ids[-1] check would re-append the marker, doubling it.
                row_ids = tokenizer.encode(text, add_special_tokens=False)
                last_meaningful = _last_non_whitespace_token(row_ids, tokenizer)
                if last_meaningful != last_token_id:
                    text = text + last_token
            rows.append({"text": text})

    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return Dataset.from_list(rows)


class ResponseOnlyCollator:
    """Right-pads a batch and masks all label positions up to and including the
    FINAL assistant marker, so loss is computed on the final assistant turn's
    completion only (the marker itself is context, not supervised).

    Constructor takes the marker as a string (e.g. for Granite:
    ``<|start_of_role|>assistant<|end_of_role|>``); we tokenize it once and
    cache the IDs. At call time we pad each row's ``input_ids`` /
    ``attention_mask`` to the longest in the batch, copy ``input_ids`` into
    ``labels``, set positions up to and including the LAST marker occurrence to
    -100, and -100 the padding positions too.

    Two non-obvious design choices, both verified against the adapter team's
    parquet (scripts/inspect_label_boundary.py output —
    ``diff(labels - last_marker_end) values: [-3]`` constant across all
    sampled rows):

    - **Last marker, not first.** Multi-turn RAG conversations have multiple
      assistant turns; only the final turn is the supervised label. Masking
      from the first marker would train the model to also generate the long
      factual content of intermediate turns. This was the cause of LoRA
      underperforming aLoRA earlier on this branch — aLoRA's gating saved
      it; LoRA's wider loss horizon let it drift to free-form factual
      answers instead of the answerability label.
    - **Mask THROUGH the marker; supervise what follows.** The role-tag
      triplet ``<|start_of_role|>assistant<|end_of_role|>`` is MASKED context —
      supervision starts at the token *after* the marker, so ``<|end_of_role|>``
      is the last context token and everything after it is generation. This
      matches ``adapter.last_context_token = "<|end_of_role|>"`` and the
      ``validate_last_context_token_boundary`` check.
    """

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", response_template: str):
        if not response_template:
            raise ValueError("response_template must be a non-empty string")
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        if not self.response_template_ids:
            raise ValueError(
                f"response_template {response_template!r} encodes to empty token list"
            )
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features: list[dict]) -> dict:
        import torch

        # Each feature comes from SFTTrainer's tokenized dataset and has at
        # least input_ids; attention_mask defaults to 1s if absent.
        max_len = max(len(f["input_ids"]) for f in features)
        bsz = len(features)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

        for i, f in enumerate(features):
            ids = f["input_ids"]
            n = len(ids)
            input_ids[i, :n] = torch.as_tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = torch.as_tensor(
                f.get("attention_mask", [1] * n), dtype=torch.long
            )

            # Find the LAST response template; supervise the completion that
            # follows it. The marker triplet itself is MASKED (it is context,
            # not generation) — supervision starts at the token AFTER the
            # marker, so the marker's final token (<|end_of_role|>) is the last
            # context token. If absent, the whole row stays -100.
            cut = _find_last_subsequence(list(ids), self.response_template_ids)
            if cut >= 0:
                start = cut + len(self.response_template_ids)
                labels[i, start:n] = torch.as_tensor(ids[start:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DebugPrintingCollator:
    """Wraps any collator and prints a decoded sample of ~`probability` of batches.

    Logs the decoded text of one example per sampled batch alongside its
    input_ids, labels, and the count of trainable (non -100) positions — the
    same shape of information the reference training script prints. Works
    regardless of whether label masking came from TRL's assistant_only_loss
    path or our own ResponseOnlyCollator.
    """

    def __init__(self, base_collator, tokenizer: "PreTrainedTokenizerBase", probability: float = 0.001):
        self.base_collator = base_collator
        self.tokenizer = tokenizer
        self.probability = probability

    def __call__(self, features):
        import random

        batch = self.base_collator(features)
        if random.random() >= self.probability:
            return batch

        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        if input_ids is None or labels is None:
            return batch

        i = 0
        seq = input_ids[i].tolist()
        lbl = labels[i].tolist()
        trainable = sum(1 for x in lbl if x != -100)

        logger.info("=" * 70)
        logger.info("DEBUG COLLATOR OUTPUT")
        logger.info("=" * 70)
        logger.info("Example %d from batch:", i)
        logger.info("\nInput text:\n%s", self.tokenizer.decode(seq))
        logger.info("\nInput IDs: %s", seq)
        logger.info("\nLabels: %s", lbl)
        logger.info("\nTrainable tokens: %d/%d", trainable, len(lbl))
        logger.info("=" * 70)

        return batch


def validate_last_context_token_boundary(
    dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    collator,
    last_context_token: str,
    sample_n: int = 8,
) -> None:
    """Refuse to start training unless the supervised region starts right after
    ``last_context_token``.

    The adapter is always active; ``last_context_token`` does not gate the
    model. It is a data-shape contract: the final context/prompt token must be
    ``last_context_token``, and the supervised region (``labels != -100``) must
    begin at the very next position. This pins the prompt/completion boundary so
    a mismatched chat template / marker / tokenizer fails loudly instead of
    silently supervising the wrong span.

    Path-independent by construction: we run the *actual* trainer collator over
    the first ``sample_n`` rows and inspect the labels it produces. This covers
    both masking paths (TRL ``assistant_only_loss`` — where TRL owns masking, so
    we can only validate — and our own ``ResponseOnlyCollator``) because both
    ultimately emit a ``labels`` tensor.

    For each sampled row: find the first unmasked label position ``p``
    (``labels[p] != -100``) and assert ``input_ids[p-1]`` is
    ``last_context_token``. Any row where the boundary token is wrong raises.
    Rows with no unmasked labels (fully-masked prompt-only rows) are skipped
    with a warning.

    Args:
        dataset: the training dataset (``text`` column, from
            :func:`load_jsonl_dataset`).
        tokenizer: the training tokenizer.
        collator: the collator the trainer will actually use (already
            constructed, post-tokenization). Called on tokenized features.
        last_context_token: the expected final-context-token string; must
            encode to exactly one token id.
        sample_n: how many rows to check.
    """
    import torch

    ctx_ids = tokenizer.encode(last_context_token, add_special_tokens=False)
    if len(ctx_ids) != 1:
        raise ValueError(
            f"last_context_token {last_context_token!r} must encode to exactly "
            f"one token id, got {ctx_ids}."
        )
    ctx_id = ctx_ids[0]

    n_to_scan = min(sample_n, len(dataset))
    if n_to_scan == 0:
        raise ValueError("Cannot validate last_context_token: dataset is empty")

    n_checked = 0
    for i in range(n_to_scan):
        text = dataset[i].get("text")
        if text is None:
            raise ValueError(
                f"Row {i} has no 'text' column. "
                "Did you load the dataset with load_jsonl_dataset()?"
            )
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Run the real collator on a single-row batch and read back its labels.
        batch = collator([{"input_ids": ids, "attention_mask": [1] * len(ids)}])
        labels = batch.get("labels")
        input_ids = batch.get("input_ids")
        if labels is None or input_ids is None:
            raise ValueError(
                "Collator did not produce 'labels'/'input_ids'; cannot validate "
                "the last_context_token boundary."
            )
        row_labels = labels[0].tolist() if isinstance(labels, torch.Tensor) else list(labels[0])
        row_ids = input_ids[0].tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids[0])

        first_unmasked = next((j for j, v in enumerate(row_labels) if v != -100), None)
        if first_unmasked is None:
            logger.warning(
                "Row %d has no unmasked labels — skipping last_context_token "
                "boundary check for it.", i,
            )
            continue
        if first_unmasked == 0:
            raise ValueError(
                f"Row {i}: supervised region starts at position 0, so there is no "
                f"preceding context token to match against last_context_token "
                f"{last_context_token!r}."
            )
        preceding = row_ids[first_unmasked - 1]
        if preceding != ctx_id:
            raise ValueError(
                f"Row {i}: the token immediately before the supervised region is "
                f"{preceding} (decoded: {tokenizer.decode([preceding])!r}), but "
                f"last_context_token {last_context_token!r} encodes to {ctx_id}. "
                f"The prompt/completion boundary does not match last_context_token. "
                f"Common causes: the last_context_token string doesn't match the "
                f"chat template's assistant marker for this base model, the data "
                f"was produced with a different tokenizer, or the masking path "
                f"cuts at a different position."
            )
        n_checked += 1

    logger.info(
        "last_context_token boundary validated on %d/%d sampled rows.",
        n_checked, n_to_scan,
    )
