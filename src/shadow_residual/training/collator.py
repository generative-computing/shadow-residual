# SPDX-License-Identifier: Apache-2.0
"""Data collators for shadow-residual single-expert training.

Both collators require a ``labels`` column in the dataset.  The insertion
point for the control token is derived from the labels: the last
``-100 → non-(-100)`` transition marks where the completion begins, and
the control token is inserted immediately before that point.

Use :mod:`shadow_residual.training.add_labels` to stamp a
``labels`` column onto a raw tokenized dataset before training.

Two collator variants:

- :class:`DataCollatorForSingleExpert`: Pads to max length in batch.
- :class:`DataCollatorForSingleExpertPacked`: Concatenates sequences into
  packed batches with ``position_ids`` resets at document boundaries,
  eliminating padding waste for variable-length sequences.

Both support a ``classic_lora`` mode where the control token is inserted at
position 0 instead of before the completion.  This is appropriate for
standard LoRA adapters that activate at the beginning of the sequence and
work directly with SFT-formatted datasets.
"""

import logging
import warnings
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def _find_insertion_point(labels: list[int], example_idx: int = 0) -> int:
    """Find where to insert the control token from dataset labels.

    Scans for the last ``-100 → non-(-100)`` transition.  The control token
    is inserted immediately before that position (i.e., at the index of the
    first non-masked token in the final unmasked run).

    Warns if more than one such transition is found, as the collator only
    supports a single prompt/completion boundary per example.

    Args:
        labels: List of label token IDs (``-100`` for masked positions).
        example_idx: Example index, used in warning messages.

    Returns:
        Insertion index.  Returns ``0`` if no ``-100`` labels are found
        (all tokens are part of the completion).
    """
    last_transition = None
    transition_count = 0

    for i in range(len(labels) - 1):
        if labels[i] == -100 and labels[i + 1] != -100:
            last_transition = i + 1
            transition_count += 1

    if transition_count > 1:
        warnings.warn(
            f"Example {example_idx} has {transition_count} masked→unmasked "
            "transitions in labels. Only a single prompt/completion boundary "
            "is supported; inserting at the last transition."
        )

    return last_transition if last_transition is not None else 0


@dataclass
class DataCollatorForSingleExpert:
    """Collator for single-expert training with padded batching.

    For each tokenized example in the batch:
    1. Find the insertion point from ``labels``: last ``-100 → non-(-100)``
       transition (or position 0 in ``classic_lora`` mode).
    2. Insert ``adapter_token_id`` at that position in ``input_ids``.
    3. Insert ``1`` at the same position in ``attention_mask``.
    4. Insert ``-100`` at the same position in ``labels``.

    Every example must have ``input_ids``, ``attention_mask``, and ``labels``
    columns.

    Args:
        adapter_token_id: The control token ID to insert.
        classic_lora: Insert at position 0 instead of before the completion.
            Use for standard LoRA adapters on SFT datasets.
        pad_token_id: Token used for padding (default 0).
    """

    adapter_token_id: int
    classic_lora: bool = False
    pad_token_id: int = 0

    def __call__(self, features: list[dict]) -> dict:
        augmented = []
        for idx, feat in enumerate(features):
            ids = list(feat["input_ids"])
            mask = list(feat["attention_mask"])
            labels = list(feat["labels"])

            pos = 0 if self.classic_lora else _find_insertion_point(labels, idx)

            ids.insert(pos, self.adapter_token_id)
            mask.insert(pos, 1)
            labels.insert(pos, -100)

            augmented.append({"input_ids": ids, "attention_mask": mask, "labels": labels})

        max_len = max(len(a["input_ids"]) for a in augmented)

        batch_ids, batch_mask, batch_labels = [], [], []
        for a in augmented:
            pad_len = max_len - len(a["input_ids"])
            batch_ids.append(a["input_ids"] + [self.pad_token_id] * pad_len)
            batch_mask.append(a["attention_mask"] + [0] * pad_len)
            batch_labels.append(a["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


@dataclass
class DataCollatorForSingleExpertPacked:
    """Padding-free collator for single-expert training.

    Eliminates padding waste by concatenating examples into a single flat
    sequence with ``position_ids`` resets at document boundaries.  The absence
    of ``attention_mask`` in the output triggers transformers'
    ``create_causal_mask`` to use ``find_packed_sequence_indices`` for
    automatic document-boundary-aware causal masking.

    Insertion point is derived from ``labels`` (last ``-100 → non-(-100)``
    transition), or position 0 in ``classic_lora`` mode.

    Packing strategy:
    - Greedily packs examples (after control token insertion) until the next
      example would exceed ``max_seq_length``.
    - Examples that don't fit are skipped (resampled in future batches).
    - A single example exceeding ``max_seq_length`` is truncated with a warning.

    Output (batch size is always 1):
    - ``input_ids``: ``[1, total_tokens]`` — concatenated examples
    - ``labels``: ``[1, total_tokens]`` — per-document masking preserved,
      control token position always ``-100``, first token of each document
      forced to ``-100`` (no cross-document loss leakage)
    - ``position_ids``: ``[1, total_tokens]`` — resets to 0 per document
    - No ``attention_mask`` key (triggers packed-sequence detection)

    Args:
        adapter_token_id: The control token ID to insert.
        max_seq_length: Maximum packed sequence length.
        classic_lora: Insert at position 0 instead of before the completion.
    """

    adapter_token_id: int
    max_seq_length: int
    classic_lora: bool = False

    def _prepare_example(self, feat: dict, idx: int) -> dict:
        ids = list(feat["input_ids"])
        labels = list(feat["labels"])

        pos = 0 if self.classic_lora else _find_insertion_point(labels, idx)

        ids.insert(pos, self.adapter_token_id)
        labels.insert(pos, -100)

        return {"input_ids": ids, "labels": labels}

    def __call__(self, features: list[dict]) -> dict:
        prepared = []
        for idx, feat in enumerate(features):
            ex = self._prepare_example(feat, idx)
            ex_len = len(ex["input_ids"])

            if ex_len > self.max_seq_length:
                warnings.warn(
                    f"Example {idx} has {ex_len} tokens after control token "
                    f"insertion, exceeding max_seq_length={self.max_seq_length}. "
                    "Truncating."
                )
                ex["input_ids"] = ex["input_ids"][: self.max_seq_length]
                ex["labels"] = ex["labels"][: self.max_seq_length]

            prepared.append(ex)

        packed_ids = []
        packed_labels = []
        packed_position_ids = []

        for ex in prepared:
            ex_len = len(ex["input_ids"])
            if len(packed_ids) + ex_len > self.max_seq_length:
                continue  # doesn't fit; will be resampled

            packed_position_ids.extend(range(ex_len))

            doc_labels = list(ex["labels"])
            doc_labels[0] = -100  # no cross-document loss leakage

            packed_ids.extend(ex["input_ids"])
            packed_labels.extend(doc_labels)

        if not packed_ids:
            raise ValueError(
                f"No examples fit within max_seq_length={self.max_seq_length} "
                "after control token insertion."
            )

        return {
            "input_ids": torch.tensor([packed_ids], dtype=torch.long),
            "labels": torch.tensor([packed_labels], dtype=torch.long),
            "position_ids": torch.tensor([packed_position_ids], dtype=torch.long),
        }


__all__ = [
    "DataCollatorForSingleExpert",
    "DataCollatorForSingleExpertPacked",
]
