# SPDX-License-Identifier: Apache-2.0
"""Data utilities: JSONL loader, response-only collator, aLoRA sanity check.

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

    Returns a `datasets.Dataset` with a single ``text`` column containing the
    chat-templated string. SFTTrainer reads the ``text`` field directly.
    """
    from datasets import Dataset

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
            rows.append({"text": text})

    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return Dataset.from_list(rows)


class ResponseOnlyCollator:
    """Right-pads a batch and masks all label positions before the FINAL
    assistant marker, so loss is computed on the final assistant turn only.

    Constructor takes the marker as a string (e.g. for Granite:
    ``<|start_of_role|>assistant<|end_of_role|>``); we tokenize it once and
    cache the IDs. At call time we pad each row's ``input_ids`` /
    ``attention_mask`` to the longest in the batch, copy ``input_ids`` into
    ``labels``, set positions before the LAST marker occurrence to -100, and
    -100 the padding positions too.

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
    - **Mask BEFORE the marker, not through it.** The role-tag triplet
      ``<|start_of_role|>assistant<|end_of_role|>`` is part of the unmasked
      region — the model is trained to predict it too. Matches the adapter
      team's parquet ``labels`` (stored mask starts at the role-tag, not
      after it).
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

            # Find the LAST response template; mask everything before it.
            # If absent, the whole row stays -100 (no loss contribution).
            cut = _find_last_subsequence(list(ids), self.response_template_ids)
            if cut >= 0:
                labels[i, cut:n] = torch.as_tensor(ids[cut:], dtype=torch.long)

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


def validate_invocation_tokens_present(
    dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    invocation_ids: list[int],
    sample_n: int = 8,
) -> None:
    """Refuse to start training if the aLoRA invocation tokens are absent from the data.

    aLoRA's runtime scans every input sequence for the invocation token
    subsequence. If that scan never matches, the adapter never activates and
    training silently does nothing useful. Common causes:

    - invocation_tokens string doesn't match the chat template's assistant
      marker for this base model;
    - data was produced with a different tokenizer / chat template version;
    - wrong base model loaded.

    We sample the first `sample_n` rows of `dataset` (rendered as the `text`
    column by `load_jsonl_dataset`), tokenize each, and look for the invocation
    subsequence. Zero matches → ValueError. Partial matches → warning.
    """
    if not invocation_ids:
        raise ValueError("validate_invocation_tokens_present requires non-empty invocation_ids")

    n_to_scan = min(sample_n, len(dataset))
    if n_to_scan == 0:
        raise ValueError("Cannot validate invocation tokens: dataset is empty")

    rows_with_token = 0
    for i in range(n_to_scan):
        text = dataset[i].get("text")
        if text is None:
            raise ValueError(
                f"Row {i} has no 'text' column. "
                "Did you load the dataset with load_jsonl_dataset()?"
            )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if _find_subsequence(ids, invocation_ids) >= 0:
            rows_with_token += 1

    if rows_with_token == 0:
        raise ValueError(
            f"aLoRA invocation tokens {invocation_ids} not found in any of "
            f"{n_to_scan} sampled training rows. The adapter would never activate "
            f"and training would not learn anything. "
            f"Common causes: the invocation_tokens string doesn't match the chat "
            f"template's assistant marker for this base model, the data was "
            f"produced with a different tokenizer, or you're pointing at the wrong "
            f"base model."
        )

    if rows_with_token < n_to_scan:
        logger.warning(
            "aLoRA invocation tokens found in only %d of %d sampled rows — "
            "the adapter will be dormant on the remaining %d. Verify your data.",
            rows_with_token, n_to_scan, n_to_scan - rows_with_token,
        )
    else:
        logger.info(
            "aLoRA invocation tokens present in all %d sampled rows.", n_to_scan,
        )
