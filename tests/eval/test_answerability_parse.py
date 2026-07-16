# SPDX-License-Identifier: Apache-2.0
"""Tests for answerability prediction parsing across Granite 4.1 (clean label)
and Granite 4.2 (ChatML <think>...</think> reasoning-wrapped) output formats."""

from __future__ import annotations

import pytest

# The eval module imports torch/numpy/pandas at top level.
pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("pandas")

from shadow_residual.eval.answerability_eval import process_predictions


def test_granite_41_clean_labels():
    """4.1 emits the bare label (no think block)."""
    assert process_predictions(["answerable", "unanswerable"]) == ["answerable", "unanswerable"]


def test_granite_42_think_wrapped():
    """4.2 emits <think>reasoning</think>label — parse after the closing tag."""
    preds = [
        "<think>Okay, the user is asking whether the docs cover this.</think>unanswerable",
        "<think>The documents mention it directly.</think>answerable",
    ]
    assert process_predictions(preds) == ["unanswerable", "answerable"]


def test_granite_42_reasoning_prefix_without_would_have_failed_before():
    """The exact 4.2-8b failure case: reasoning text then </think> then label.

    Before the fix, the first token ('Okay,') → 'others' (0.0 across the board).
    After the fix, the label after </think> is read correctly.
    """
    preds = ["Okay, the user wrote a question. </think>unanswerable"]
    assert process_predictions(preds) == ["unanswerable"]


def test_empty_think_block():
    """<think></think>label (empty reasoning) still parses the trailing label."""
    assert process_predictions(["<think></think>answerable"]) == ["answerable"]


def test_quoted_label_after_think():
    """Quoted label following the think block."""
    assert process_predictions(['<think>reasoning</think>"unanswerable"']) == ["unanswerable"]


def test_unparseable_is_others():
    """Genuinely unparseable output → 'others' (no label present)."""
    assert process_predictions(["<think>hmm</think>I am not sure about this"]) == ["others"]
    assert process_predictions(["complete gibberish"]) == ["others"]


def test_unanswerable_not_shadowed_by_answerable_substring():
    """'unanswerable' must not be misread as 'answerable' (substring trap)."""
    assert process_predictions(["<think>x</think>unanswerable"]) == ["unanswerable"]
