# SPDX-License-Identifier: Apache-2.0
"""Add a ``labels`` column to a raw tokenized HuggingFace dataset.

Stamps each example with a ``labels`` list: ``-100`` (masked) for all tokens
up to and including the last occurrence of the activation sequence, and real
token IDs for everything after (the completion).

This is a one-time preprocessing step required before training with
:mod:`shadow_residual.training.train`, whose collators expect
``labels`` to be present in the dataset.

Usage::

    python -m shadow_residual.training.add_labels \\
        --dataset_path /path/to/tokenized_dataset \\
        --activation_sequence 100264 78191 100265 \\
        --output_path /path/to/labeled_dataset

The dataset at ``--dataset_path`` must be a HuggingFace ``Dataset`` saved
via ``dataset.save_to_disk()``, with at least an ``input_ids`` column.
"""

import logging
from dataclasses import dataclass, field

from datasets import load_from_disk
from transformers import HfArgumentParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AddLabelsArguments:
    dataset_path: str = field(
        metadata={"help": "Path to a HuggingFace Dataset (saved with save_to_disk)."},
    )
    activation_sequence: list[int] = field(
        metadata={"help": "Token IDs marking the start of the completion (e.g., assistant role header)."},
    )
    output_path: str = field(
        metadata={"help": "Path to save the labeled dataset."},
    )


def _find_rightmost(ids: list[int], activation_sequence: list[int]) -> int:
    """Return index of the rightmost occurrence of activation_sequence in ids, or -1."""
    seq_len = len(activation_sequence)
    for i in range(len(ids) - seq_len, -1, -1):
        if ids[i : i + seq_len] == activation_sequence:
            return i
    return -1


def add_labels(dataset_path: str, activation_sequence: list[int], output_path: str) -> None:
    """Load a dataset, add a labels column, and save.

    Args:
        dataset_path: Path to raw tokenized HuggingFace Dataset.
        activation_sequence: Token IDs marking the completion boundary.
        output_path: Where to save the labeled dataset.
    """
    dataset = load_from_disk(dataset_path)
    logger.info("Loaded dataset: %d examples, columns: %s", len(dataset), dataset.column_names)

    missing = 0

    def label_example(example, idx):
        nonlocal missing
        ids = example["input_ids"]
        pos = _find_rightmost(ids, activation_sequence)
        if pos < 0:
            missing += 1
            labels = [-100] * len(ids)
        else:
            completion_start = pos + len(activation_sequence)
            labels = [-100] * completion_start + ids[completion_start:]
        return {"labels": labels}

    dataset = dataset.map(label_example, with_indices=True, desc="Adding labels")

    if missing > 0:
        logger.warning(
            "%d / %d examples did not contain the activation sequence — "
            "their labels are fully masked (no loss contribution).",
            missing, len(dataset),
        )

    dataset.save_to_disk(output_path)
    logger.info("Saved labeled dataset to %s", output_path)


def main():
    parser = HfArgumentParser(AddLabelsArguments)
    (args,) = parser.parse_args_into_dataclasses()
    add_labels(args.dataset_path, args.activation_sequence, args.output_path)


if __name__ == "__main__":
    main()
