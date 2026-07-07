# SPDX-License-Identifier: Apache-2.0
"""Training utilities for GraniteSwitch experts (experimental).

Public API:

- :class:`DataCollatorForSingleExpert`,
  :class:`DataCollatorForSingleExpertPacked` — control-token-injecting
  collators (padded and padding-free).
- :func:`recommend_batch_size`, :func:`analyze_dataset` — packing-stats
  helpers for tuning batch size in padding-free mode.
- :func:`add_labels` — preprocessing step that stamps a ``labels`` column
  derived from an activation token sequence.
"""

from .add_labels import add_labels
from .collator import (
    DataCollatorForSingleExpert,
    DataCollatorForSingleExpertPacked,
)
from .pack_stats import analyze_dataset, recommend_batch_size

__all__ = [
    "DataCollatorForSingleExpert",
    "DataCollatorForSingleExpertPacked",
    "add_labels",
    "analyze_dataset",
    "recommend_batch_size",
]
