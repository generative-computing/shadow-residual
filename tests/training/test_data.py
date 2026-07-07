# SPDX-License-Identifier: Apache-2.0
"""Tests for the JSONL loader, subsequence helper, and aLoRA invocation validator."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

# Skip the whole module unless transformers + datasets + torch are installed.
# torch is required because importing data.py pulls in collator.py which
# uses torch directly.
pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("torch")

from shadow_residual.training.data import (  # noqa: E402
    ResponseOnlyCollator,
    _find_last_subsequence,
    _find_subsequence,
    load_jsonl_dataset,
    validate_invocation_tokens_present,
)


# ---------- Fixtures ----------


@pytest.fixture(scope="module")
def tokenizer():
    """Tiny chat-template-aware tokenizer.

    `hf-internal-testing/tiny-random-LlamaForCausalLM` ships a chat template
    and is a few KB to download — fine for CI.

    The Granite chat markers are registered as additional special tokens so
    they tokenize **atomically and context-independently** — exactly as the
    real Granite tokenizers treat them. Without this, the stock Llama BPE
    encodes a marker differently in isolation vs. embedded in surrounding text
    (the leading ``<`` merges with preceding bytes), so the standalone-encoded
    ``invocation_ids`` would not appear as a contiguous subsequence inside a
    full row's tokenization — a fixture artifact, not behavior these tests
    intend to exercise.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens(
        {"additional_special_tokens": [
            "<|start_of_role|>", "<|end_of_role|>", "<|end_of_turn|>",
        ]}
    )
    return tok


@pytest.fixture
def jsonl_path(tmp_path: Path) -> Path:
    rows = [
        {"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]},
        {"messages": [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "Two."},
        ]},
        {"messages": [
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]},
        {"messages": [
            {"role": "user", "content": "Color of the sky?"},
            {"role": "assistant", "content": "Blue."},
        ]},
    ]
    p = tmp_path / "tiny.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# ---------- load_jsonl_dataset ----------


def test_load_jsonl_dataset_roundtrip(jsonl_path, tokenizer):
    ds = load_jsonl_dataset(jsonl_path, tokenizer)
    assert len(ds) == 4
    assert set(ds.column_names) == {"text"}
    for row in ds:
        assert isinstance(row["text"], str)
        assert row["text"]


def test_load_jsonl_rejects_missing_messages(tmp_path, tokenizer):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"text": "no messages key"}) + "\n")
    with pytest.raises(ValueError, match="missing 'messages'"):
        load_jsonl_dataset(bad, tokenizer)


def test_load_jsonl_rejects_empty_file(tmp_path, tokenizer):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty dataset"):
        load_jsonl_dataset(empty, tokenizer)


def test_load_jsonl_skips_blank_lines(tmp_path, tokenizer):
    p = tmp_path / "blanks.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]},
    ]
    with p.open("w") as f:
        f.write("\n")
        f.write(json.dumps(rows[0]) + "\n")
        f.write("\n\n")
        f.write(json.dumps(rows[1]) + "\n")
        f.write("\n")
    ds = load_jsonl_dataset(p, tokenizer)
    assert len(ds) == 2


# ---------- tools / documents pass-through ----------


def test_load_jsonl_accepts_tools_and_documents(tmp_path, tokenizer):
    """load_jsonl_dataset accepts rows with optional tools/documents fields
    without crashing; the kwargs are forwarded to apply_chat_template (the
    template is responsible for rendering them — the granite template does,
    the tiny-llama template ignores them, both are valid).
    """
    p = tmp_path / "with_tools_docs.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ],
            "tools": [{"name": "search", "description": "search the web"}],
            "documents": [{"doc_id": "0", "text": "hello world"}],
        },
        {  # tools only
            "messages": [
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
            "tools": [{"name": "calc"}],
        },
        {  # documents only
            "messages": [
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
            ],
            "documents": [{"doc_id": "0", "text": "doc"}],
        },
        {  # neither (already covered by other tests, here for completeness)
            "messages": [
                {"role": "user", "content": "Q4"},
                {"role": "assistant", "content": "A4"},
            ],
        },
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ds = load_jsonl_dataset(p, tokenizer)
    assert len(ds) == 4
    for row in ds:
        assert isinstance(row["text"], str)
        assert row["text"]


# ---------- _find_subsequence ----------


def test_find_subsequence_match():
    assert _find_subsequence([1, 2, 3, 4, 5], [3, 4]) == 2
    assert _find_subsequence([1, 2, 3, 4, 5], [1]) == 0
    assert _find_subsequence([1, 2, 3, 4, 5], [5]) == 4


def test_find_subsequence_no_match():
    assert _find_subsequence([1, 2, 3], [9]) == -1
    assert _find_subsequence([1, 2, 3], [2, 4]) == -1


def test_find_subsequence_edge_cases():
    assert _find_subsequence([], [1]) == -1
    assert _find_subsequence([1, 2], []) == -1
    assert _find_subsequence([1, 2], [1, 2, 3]) == -1  # needle longer than haystack


# ---------- _find_last_subsequence ----------


def test_find_last_subsequence_returns_last_match():
    # Three occurrences of [9, 9] starting at indices 0, 3, 6.
    assert _find_last_subsequence([9, 9, 1, 9, 9, 1, 9, 9], [9, 9]) == 6
    # Single match → same as first.
    assert _find_last_subsequence([1, 2, 3, 4, 5], [3, 4]) == 2


def test_find_last_subsequence_no_match():
    assert _find_last_subsequence([1, 2, 3], [9]) == -1
    assert _find_last_subsequence([1, 2, 3], [2, 4]) == -1


def test_find_last_subsequence_edge_cases():
    assert _find_last_subsequence([], [1]) == -1
    assert _find_last_subsequence([1, 2], []) == -1
    assert _find_last_subsequence([1, 2], [1, 2, 3]) == -1


def test_find_last_subsequence_overlapping():
    # Greedy from the end — want index 4, not 0 (despite overlap at 0,2,4).
    assert _find_last_subsequence([1, 1, 1, 1, 1, 1], [1, 1]) == 4


# ---------- validate_invocation_tokens_present ----------


def _ds_with_text(rows: list[str]):
    """Build a minimal datasets.Dataset with a 'text' column."""
    from datasets import Dataset
    return Dataset.from_list([{"text": t} for t in rows])


def test_validate_invocation_tokens_all_present(tokenizer, caplog):
    marker = "<|start_of_role|>assistant<|end_of_role|>"
    invocation_ids = tokenizer.encode(marker, add_special_tokens=False)
    rows = [
        f"<|start_of_role|>user<|end_of_role|>q{i}<|end_of_turn|>"
        f"{marker}a{i}<|end_of_turn|>"
        for i in range(4)
    ]
    ds = _ds_with_text(rows)
    with caplog.at_level(logging.INFO):
        validate_invocation_tokens_present(ds, tokenizer, invocation_ids)
    assert any("present in all" in r.message for r in caplog.records)


def test_validate_invocation_tokens_missing_raises(tokenizer):
    marker = "<|start_of_role|>assistant<|end_of_role|>"
    invocation_ids = tokenizer.encode(marker, add_special_tokens=False)
    ds = _ds_with_text(["just some text", "another row", "no marker here"])
    with pytest.raises(ValueError, match="not found in any of"):
        validate_invocation_tokens_present(ds, tokenizer, invocation_ids)


def test_validate_invocation_tokens_partial_warns(tokenizer, caplog):
    marker = "<|start_of_role|>assistant<|end_of_role|>"
    invocation_ids = tokenizer.encode(marker, add_special_tokens=False)
    ds = _ds_with_text([
        f"<|start_of_role|>user<|end_of_role|>q<|end_of_turn|>{marker}a<|end_of_turn|>",
        "no marker here",
    ])
    with caplog.at_level(logging.WARNING):
        validate_invocation_tokens_present(ds, tokenizer, invocation_ids)
    assert any("dormant on the remaining" in r.message for r in caplog.records)


def test_validate_invocation_tokens_empty_ids_raises(tokenizer):
    ds = _ds_with_text(["anything"])
    with pytest.raises(ValueError, match="non-empty invocation_ids"):
        validate_invocation_tokens_present(ds, tokenizer, [])


def test_validate_invocation_tokens_empty_dataset_raises(tokenizer):
    ds = _ds_with_text([])
    with pytest.raises(ValueError, match="dataset is empty"):
        validate_invocation_tokens_present(ds, tokenizer, [1, 2, 3])


# ---------- ResponseOnlyCollator ----------


def test_response_only_collator_masks_before_marker(tokenizer):
    marker = "<|start_of_role|>assistant<|end_of_role|>"
    coll = ResponseOnlyCollator(tokenizer, marker)
    pre = tokenizer.encode("hello", add_special_tokens=False)
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    resp = tokenizer.encode("the answer", add_special_tokens=False)
    ids = pre + marker_ids + resp
    out = coll([{"input_ids": ids, "attention_mask": [1] * len(ids)}])
    labels = out["labels"][0].tolist()
    # Mask BEFORE marker → -100; from the marker onward → real ids
    # (matches the adapter team's parquet labels mask).
    marker_start = len(pre)
    assert all(x == -100 for x in labels[:marker_start])
    assert labels[marker_start:] == ids[marker_start:]


def test_response_only_collator_uses_last_marker_on_multi_turn(tokenizer):
    """Multi-turn rows have multiple assistant markers; only the FINAL one
    should bound the loss. The whole point of the last-marker fix."""
    marker = "<|start_of_role|>assistant<|end_of_role|>"
    coll = ResponseOnlyCollator(tokenizer, marker)
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    pre = tokenizer.encode("system stuff", add_special_tokens=False)
    turn1_body = tokenizer.encode("first answer text", add_special_tokens=False)
    user_turn = tokenizer.encode("follow-up question", add_special_tokens=False)
    final_body = tokenizer.encode("final label", add_special_tokens=False)
    # Two assistant markers; the LAST one is the only one that should
    # bound the unmasked region.
    ids = pre + marker_ids + turn1_body + user_turn + marker_ids + final_body
    out = coll([{"input_ids": ids, "attention_mask": [1] * len(ids)}])
    labels = out["labels"][0].tolist()
    last_marker_start = len(pre) + len(marker_ids) + len(turn1_body) + len(user_turn)
    assert all(x == -100 for x in labels[:last_marker_start])
    assert labels[last_marker_start:] == ids[last_marker_start:]


def test_response_only_collator_full_mask_when_marker_absent(tokenizer):
    coll = ResponseOnlyCollator(tokenizer, "<|nope|>")
    ids = tokenizer.encode("just some text without the marker", add_special_tokens=False)
    out = coll([{"input_ids": ids, "attention_mask": [1] * len(ids)}])
    labels = out["labels"][0].tolist()
    # Marker absent → no positions become real labels → row contributes 0 loss.
    assert all(x == -100 for x in labels)


def test_response_only_collator_pads_batch(tokenizer):
    marker = "<|start_of_role|>assistant<|end_of_role|>"
    coll = ResponseOnlyCollator(tokenizer, marker)
    short = tokenizer.encode("a" + marker + "x", add_special_tokens=False)
    long = tokenizer.encode("aaaaa" + marker + "yyy", add_special_tokens=False)
    out = coll([
        {"input_ids": short, "attention_mask": [1] * len(short)},
        {"input_ids": long, "attention_mask": [1] * len(long)},
    ])
    # All rows padded to the longest row's length.
    assert out["input_ids"].shape == out["labels"].shape == out["attention_mask"].shape
    assert out["input_ids"].shape[0] == 2
    assert out["input_ids"].shape[1] == len(long)
    # Pad positions on the short row have attention_mask=0 and labels=-100.
    pad_count = len(long) - len(short)
    assert out["attention_mask"][0, -pad_count:].sum().item() == 0
    assert (out["labels"][0, -pad_count:] == -100).all().item()


def test_response_only_collator_empty_template_rejected(tokenizer):
    with pytest.raises(ValueError, match="non-empty"):
        ResponseOnlyCollator(tokenizer, "")


def test_validate_invocation_tokens_sample_n_caps_scan(tokenizer):
    """If sample_n < len(dataset), only the first sample_n rows are scanned."""
    marker = "<|start_of_role|>assistant<|end_of_role|>"
    invocation_ids = tokenizer.encode(marker, add_special_tokens=False)
    rows_with = [
        f"<|start_of_role|>user<|end_of_role|>q{i}<|end_of_turn|>"
        f"{marker}a{i}<|end_of_turn|>"
        for i in range(2)
    ]
    rows_without = ["no marker"] * 10
    ds = _ds_with_text(rows_with + rows_without)
    # Only scans the first 2 → no exception, no warning about dormant.
    validate_invocation_tokens_present(ds, tokenizer, invocation_ids, sample_n=2)


# ---------- enable_thinking ----------


class _RecordingTokenizer:
    """Minimal stub that records every apply_chat_template kwargs call."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "RENDERED"


def _one_row_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "one.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    return p


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_enable_thinking_forwarded_to_chat_template(tmp_path, enable_thinking):
    """load_jsonl_dataset must pass enable_thinking through to apply_chat_template."""
    tok = _RecordingTokenizer()
    ds = load_jsonl_dataset(_one_row_jsonl(tmp_path), tok, enable_thinking=enable_thinking)
    assert len(tok.calls) == 1
    assert tok.calls[0]["enable_thinking"] is enable_thinking
    assert ds[0]["text"] == "RENDERED"


def test_enable_thinking_defaults_false(tmp_path):
    """Omitting enable_thinking defaults to False (prior behavior preserved)."""
    tok = _RecordingTokenizer()
    load_jsonl_dataset(_one_row_jsonl(tmp_path), tok)
    assert tok.calls[0]["enable_thinking"] is False
