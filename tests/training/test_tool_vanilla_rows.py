# SPDX-License-Identifier: Apache-2.0
"""Tool training renders a seeded fraction (data.vanilla_frac, default 0.5) of the
source rows as plain vanilla-SR chat rows and the rest as tool rows, in one dataset.

Tokenizer-dependent cases use the open-source Granite 4.2 tokenizer and are
skipped if it can't be loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from shadow_residual.training.tool_data import (
    AUTOREGRESSIVE,
    EXTERNAL,
    TOOL_FORMATTING_MARKERS,
    VANILLA_MARKERS,
)
from shadow_residual.training.tool_data.convert import convert, to_vanilla

G42 = "ibm-granite/granite-4.2-3b"
IM_END = "<|im_end|>"
MODES = [EXTERNAL, AUTOREGRESSIVE]

GUARD_INSTR = (
    "<guardian>As a judge agent, your role is to help assess whether the provided "
    "text meets the given judging criteria.\n\n### Criteria: Violence refers to content "
    "that promotes physical harm.\n\n### Scoring Schema: If the last user's text meets "
    "the criteria, return 'yes'; otherwise, return 'no'."
)


def _ans(i: int, label: str) -> dict:
    """An adapter-bench g42-alora-4.1-split row: documents as inline tool turns."""
    return {"messages": [
        {"role": "user", "content": f"Earlier question {i}?"},
        {"role": "assistant", "content": f"Earlier answer {i}."},
        {"role": "tool", "content": f"Document one for row {i}."},
        {"role": "tool", "content": f"Document two for row {i}."},
        {"role": "user", "content": f"Question {i}?"},
        {"role": "assistant", "content": f'"{label}"'},
    ]}


ANS_ROWS = [_ans(i, "answerable" if i % 2 else "unanswerable") for i in range(10)]
GUARD_ROWS = [{"messages": [
    {"role": "user", "content": f"Text {i}."},
    {"role": "user", "content": GUARD_INSTR},
    {"role": "assistant", "content": '{"score": "%s"}' % ("yes" if i % 2 else "no")},
]} for i in range(6)]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --- converters (pure Python) --------------------------------------------------


def test_vanilla_answerability_lifts_documents_and_keeps_conversation():
    v = to_vanilla("answerability", ANS_ROWS[0])
    assert [m["role"] for m in v["messages"]] == ["user", "assistant", "user", "assistant"]
    assert v["messages"][-1]["content"] == '"unanswerable"'
    assert v["documents"] == [
        {"document_id": 0, "content": "Document one for row 0."},
        {"document_id": 1, "content": "Document two for row 0."},
    ]


def test_vanilla_intrinsic_is_the_source_row():
    v = to_vanilla("guardian_core", GUARD_ROWS[1])
    assert v == {"messages": GUARD_ROWS[1]["messages"]}
    qc = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": '{"clarification": "CLEAR"}'}],
          "documents": [{"doc_id": "", "text": "t"}]}
    assert to_vanilla("query_clarification", qc) == qc


@pytest.mark.parametrize("task, row", [
    ("answerability", {"messages": [{"role": "user", "content": "q?"}]}),
    ("guardian_core", {"messages": [{"role": "user", "content": GUARD_INSTR}]}),
    ("query_clarification", {"messages": [{"role": "user", "content": "q"}], "ground_truth": "Which one?"}),
])
def test_vanilla_requires_gold(task, row):
    with pytest.raises(ValueError, match="must end with the gold assistant turn"):
        to_vanilla(task, row)


@pytest.mark.parametrize("task, gt, gold", [
    ("answerability", "Unanswerable", '"unanswerable"'),
    ("guardian_core", "Yes", '{"score": "yes"}'),
])
def test_vanilla_gold_from_ground_truth(task, gt, gold):
    instr = "q?" if task == "answerability" else GUARD_INSTR
    v = to_vanilla(task, {"messages": [{"role": "user", "content": instr}], "ground_truth": gt})
    assert v["messages"][-1] == {"role": "assistant", "content": gold}


def test_document_ids_continue_after_documents_key():
    row = {**ANS_ROWS[0], "documents": [{"doc_id": "a", "text": "x"}]}
    docs = to_vanilla("answerability", row)["documents"]
    assert [d.get("document_id", d.get("doc_id")) for d in docs] == ["a", 1, 2]


# --- tokenizer cases -------------------------------------------------------------


@pytest.fixture(scope="module")
def tok():
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    pytest.importorskip("torch")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(G42)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{G42} tokenizer unavailable: {e}")


def _load(tok, path, mode, task, frac=0.5, seed=2):
    from shadow_residual.training.data import load_jsonl_dataset
    return load_jsonl_dataset(path, tok, last_token=IM_END, tool_formatting=mode,
                              tool_task=task, vanilla_frac=frac, seed=seed)


@pytest.mark.parametrize("mode", MODES)
def test_split_is_exact_and_seeded(tok, tmp_path, mode):
    path = _write(tmp_path / "src.jsonl", ANS_ROWS)
    ds = _load(tok, path, mode, "answerability")
    assert ds["kind"].count("vanilla") == 5 and ds["kind"].count("tool") == 5
    assert _load(tok, path, mode, "answerability")["kind"] == ds["kind"]
    assert _load(tok, path, mode, "answerability", seed=7)["kind"] != ds["kind"]
    assert _load(tok, path, mode, "answerability", frac=0.3)["kind"].count("vanilla") == 3


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("task, rows", [("answerability", ANS_ROWS), ("guardian_core", GUARD_ROWS)])
def test_tool_and_vanilla_rows_match_the_single_formats(tok, tmp_path, mode, task, rows):
    """Tool rows are byte-identical to a tool-only load (--vanilla-frac 0); vanilla
    rows to a plain (non-tool) load of the vanilla source row — the vanilla-SR training format."""
    from shadow_residual.training.data import load_jsonl_dataset
    path = _write(tmp_path / "src.jsonl", rows)
    mixed = _load(tok, path, mode, task)
    tool_only = load_jsonl_dataset(path, tok, last_token=IM_END, tool_formatting=mode, tool_task=task)
    plain = load_jsonl_dataset(_write(tmp_path / "v.jsonl", [to_vanilla(task, r) for r in rows]),
                               tok, last_token=IM_END)
    for i, (text, kind) in enumerate(zip(mixed["text"], mixed["kind"])):
        assert text == (tool_only if kind == "tool" else plain)[i]["text"]


def test_vanilla_answerability_matches_bench_document_placement(tok, tmp_path):
    """The vanilla row equals the benchmark renderer's output (data_v1): one JSON
    tool message after the last user turn, documents indexed 0..n-1."""
    from shadow_residual.training.chat_render import render_chat
    path = _write(tmp_path / "src.jsonl", ANS_ROWS[:2])
    ds = _load(tok, path, EXTERNAL, "answerability", frac=0.5, seed=0)
    i = ds["kind"].index("vanilla")
    row = ANS_ROWS[i]
    docs = [{"source": "knowledge_base", "document_id": str(k), "content": m["content"]}
            for k, m in enumerate(m for m in row["messages"] if m["role"] == "tool")]
    msgs = [m for m in row["messages"] if m["role"] != "tool"]
    bench = [*msgs[:3], {"role": "tool", "content": json.dumps(docs, indent=2)}, msgs[3]]
    assert ds[i]["text"] == render_chat(tok, bench, add_generation_prompt=False, enable_thinking=False)


@pytest.mark.parametrize("mode", MODES)
def test_collator_supervises_each_kind_exactly(tok, tmp_path, mode):
    from shadow_residual.training.data import ResponseOnlyCollator

    ds = _load(tok, _write(tmp_path / "src.jsonl", ANS_ROWS), mode, "answerability")
    col = ResponseOnlyCollator(tok, [TOOL_FORMATTING_MARKERS[mode].assistant_marker,
                                     VANILLA_MARKERS.assistant_marker])
    for text, kind, src in zip(ds["text"], ds["kind"], ANS_ROWS):
        ids = tok.encode(text, add_special_tokens=False)
        labels = col([{"input_ids": ids}])["labels"][0].tolist()
        first = next(j for j, v in enumerate(labels) if v != -100)
        sup = tok.decode([v for v in labels if v != -100])
        if kind == "vanilla":
            assert tok.decode(ids[first - 1]) == "</think>"
            assert sup == src["messages"][-1]["content"] + IM_END + "\n"
        else:
            payload = json.dumps({"answerable": src["messages"][-1]["content"] == '"answerable"'})
            assert tok.decode(ids[first - 1]) == TOOL_FORMATTING_MARKERS[mode].last_context_token
            if mode == EXTERNAL:
                assert sup == payload + IM_END
            else:
                assert sup == "\n<|im_start|>user\n<tool_response>\n" + payload + "\n</tool_response>\n" + IM_END


def test_single_template_collator_unchanged(tok):
    """One marker still cuts after its last occurrence (the single-marker behavior)."""
    from shadow_residual.training.data import ResponseOnlyCollator
    text = "<|im_start|>assistant\n<think></think>a<|im_end|>\n<|im_start|>assistant\n<think></think>b<|im_end|>"
    col = ResponseOnlyCollator(tok, VANILLA_MARKERS.assistant_marker)
    ids = tok.encode(text, add_special_tokens=False)
    labels = col([{"input_ids": ids}])["labels"][0].tolist()
    assert tok.decode([v for v in labels if v != -100]) == "b<|im_end|>"


def test_loader_rejects_row_cut_at_the_wrong_marker(tok, tmp_path):
    """A vanilla gold holding the tool marker would be cut inside the answer."""
    row = {"messages": [
        {"role": "user", "content": "Tell me."},
        {"role": "assistant", "content": '{"clarification": "Which </tool_call> do you mean?"}'},
    ]}
    path = _write(tmp_path / "src.jsonl", [row, row])
    with pytest.raises(ValueError, match=r"src\.jsonl:\d: vanilla row would be cut at '</tool_call>'"):
        _load(tok, path, AUTOREGRESSIVE, "query_clarification", frac=0.5)


@pytest.mark.parametrize("with_vanilla", [True, False])
def test_loader_drops_rows_at_or_over_max_length(tok, tmp_path, caplog, with_vanilla):
    """Truncation would cut the gold span and leave an earlier marker as the cut."""
    import logging
    from shadow_residual.training.data import load_jsonl_dataset
    long = _ans(99, "answerable")
    long["messages"][2]["content"] = "word " * 3000
    path = _write(tmp_path / "src.jsonl", [*ANS_ROWS[:4], long])
    with caplog.at_level(logging.WARNING, logger="shadow_residual.training.data"):
        ds = load_jsonl_dataset(path, tok, last_token=IM_END, tool_formatting=EXTERNAL, tool_task="answerability",
                                vanilla_frac=0.5 if with_vanilla else None, max_length=2048)
    assert len(ds) == 4
    assert "dropped 1 rows of >= max_length=2048 tokens" in caplog.text
    full = load_jsonl_dataset(path, tok, last_token=IM_END, tool_formatting=EXTERNAL, tool_task="answerability",
                              vanilla_frac=0.5 if with_vanilla else None)
    assert len(full) == 5


def test_validator_rejects_all_masked_sample(tok, tmp_path):
    from shadow_residual.training.data import ResponseOnlyCollator, validate_last_context_token_boundary
    ds = _load(tok, _write(tmp_path / "src.jsonl", ANS_ROWS), AUTOREGRESSIVE, "answerability")
    vanilla = ds.select([i for i, k in enumerate(ds["kind"]) if k == "vanilla"])
    with pytest.raises(ValueError, match="None of the 5 sampled rows has supervised labels"):
        validate_last_context_token_boundary(vanilla, tok, ResponseOnlyCollator(tok, "</tool_call>"), "</think>")


def test_loader_vanilla_rows_require_tool_task(tok, tmp_path):
    from shadow_residual.training.data import load_jsonl_dataset
    path = _write(tmp_path / "src.jsonl", ANS_ROWS)
    with pytest.raises(ValueError, match="vanilla_frac requires tool_task"):
        load_jsonl_dataset(path, tok, tool_formatting=EXTERNAL, vanilla_frac=0.5)


@pytest.mark.parametrize("mode", MODES)
def test_validate_boundaries_checks_each_kind(tok, tmp_path, mode):
    from shadow_residual.config import load_training_config
    from shadow_residual.training.data import ResponseOnlyCollator
    from shadow_residual.training.train import apply_tool_formatting, response_templates, validate_boundaries

    cfg = load_training_config(_cfg(tmp_path, mode))
    apply_tool_formatting(cfg)
    ds = _load(tok, _write(tmp_path / "src.jsonl", ANS_ROWS), mode, "answerability")
    col = ResponseOnlyCollator(tok, response_templates(cfg))
    validate_boundaries(cfg, ds, tok, col)

    # A tool-only collator cuts vanilla rows at their documents block (external)
    # or not at all (autoregressive); both fail on the vanilla rows.
    tool_only = ResponseOnlyCollator(tok, cfg.data.assistant_marker)
    with pytest.raises(ValueError, match="last_context_token '</think>'"):
        validate_boundaries(cfg, ds, tok, tool_only)
    with pytest.raises(ValueError, match="left no vanilla rows"):
        validate_boundaries(cfg, ds.select([i for i, k in enumerate(ds["kind"]) if k == "tool"]), tok, col)


# --- config / CLI ----------------------------------------------------------------


def _cfg(tmp_path: Path, mode: str = EXTERNAL, **data) -> Path:
    p = tmp_path / "tool.yaml"
    p.write_text(yaml.safe_dump({
        "model": {"base": G42},
        "data": {"train_path": "t.jsonl", "val_path": "v.jsonl", "tool_formatting": mode,
                 "tool_task": "answerability", **data},
        "adapter": {"target_modules": {"q_proj": 8, "cross_stream": 8}},
        "save": {"output_dir": str(tmp_path / "out")},
    }))
    return p


def _cli(tmp_path, *argv, **data):
    from shadow_residual.config import load_training_config
    from shadow_residual.training.train import _build_parser, apply_cli_overrides, apply_tool_formatting
    path = _cfg(tmp_path, **data)
    args = _build_parser().parse_args(["--config", str(path), *argv])
    cfg = load_training_config(path)
    apply_cli_overrides(cfg, args)
    apply_tool_formatting(cfg)
    return cfg


def test_tool_training_defaults_to_half_vanilla_rows(tmp_path):
    assert _cli(tmp_path).data.vanilla_frac == 0.5


def test_cli_vanilla_frac_overrides_and_zero_disables(tmp_path):
    assert _cli(tmp_path, "--vanilla-frac", "0.25").data.vanilla_frac == 0.25
    assert _cli(tmp_path, "--vanilla-frac", "0").data.vanilla_frac == 0.0


def test_pre_converted_tool_rows_default_to_tool_rows_only(tmp_path):
    assert _cli(tmp_path, tool_task=None).data.vanilla_frac == 0.0


def test_non_tool_training_has_no_vanilla_rows(tmp_path):
    cfg = _cli(tmp_path, tool_formatting=None, tool_task=None)
    assert cfg.data.vanilla_frac == 0.0 and cfg.adapter.last_context_token is None


@pytest.mark.parametrize("frac", ["1", "-0.1"])
def test_cli_vanilla_frac_out_of_range(tmp_path, frac):
    with pytest.raises(ValueError, match=r"--vanilla-frac must be in \[0, 1\)"):
        _cli(tmp_path, "--vanilla-frac", frac)


def test_config_vanilla_frac_out_of_range(tmp_path):
    from shadow_residual.config import load_training_config
    with pytest.raises(Exception, match="vanilla_frac"):
        load_training_config(_cfg(tmp_path, vanilla_frac=1.0))


def test_config_vanilla_frac_zero_is_off(tmp_path):
    assert _cli(tmp_path, vanilla_frac=0.0).data.vanilla_frac == 0.0


def test_explicit_vanilla_frac_requires_tool_task(tmp_path):
    with pytest.raises(ValueError, match=r"vanilla_frac=0\.5 requires data\.tool_task"):
        _cli(tmp_path, tool_task=None, vanilla_frac=0.5)


def test_vanilla_rows_reject_thinking(tmp_path):
    with pytest.raises(ValueError, match="thinking-off.*--vanilla-frac 0"):
        _cli(tmp_path, "--thinking")
    assert _cli(tmp_path, "--thinking", "--vanilla-frac", "0").data.enable_thinking


def test_response_templates(tmp_path):
    from shadow_residual.training.train import response_templates
    cfg = _cli(tmp_path)
    assert response_templates(cfg) == ["<tool_response>\n", VANILLA_MARKERS.assistant_marker]
    assert response_templates(_cli(tmp_path, "--vanilla-frac", "0")) == "<tool_response>\n"


def test_reference_tool_config_uses_the_default(monkeypatch):
    from shadow_residual.config import load_training_config
    from shadow_residual.training.train import apply_tool_formatting
    for k, v in {"BASE_MODEL": G42, "DATA_ROOT": "/d", "RUN_DIR": "/r", "EVAL_DATA": "/e"}.items():
        monkeypatch.setenv(k, v)
    cfg = load_training_config(Path(__file__).resolve().parents[2] / "src/shadow_residual/config/sr-tool-g42.yaml")
    assert cfg.data.vanilla_frac is None and cfg.data.tool_task == "answerability"
    apply_tool_formatting(cfg)
    assert cfg.data.vanilla_frac == 0.5


def test_vanilla_markers():
    assert VANILLA_MARKERS == ("<|im_start|>assistant\n<think></think>", "</think>", IM_END)


def test_convert_unchanged_for_documents_key_rows():
    """Rows that already carry a documents key (data_v4 shape) convert as before."""
    row = {"messages": [{"role": "user", "content": "q?"}, {"role": "assistant", "content": '"answerable"'}],
           "documents": [{"doc_id": "", "text": "t"}]}
    assert convert("answerability", row, include_response=True)["documents"] == row["documents"]
