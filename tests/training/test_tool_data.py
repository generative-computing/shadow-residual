# SPDX-License-Identifier: Apache-2.0
"""SR-adapter-as-tool data: converters, CLI, mode rendering, markers, model gate.

Tokenizer-dependent cases use the open-source Granite 4.2 tokenizer
(``ibm-granite/granite-4.2-3b``) and are skipped if it can't be loaded; the
converter / CLI / payload-extraction cases are pure Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadow_residual.training.tool_data import (
    AUTOREGRESSIVE,
    EXTERNAL,
    TOOL_FORMATTING_MARKERS,
    extract_tool_payload,
)
from shadow_residual.training.tool_data import answerability, intrinsics
from shadow_residual.training.tool_data.__main__ import main as tool_data_main

G42 = "ibm-granite/granite-4.2-3b"
G41 = "ibm-granite/granite-4.1-3b"
IM_END = "<|im_end|>"
MODES = [EXTERNAL, AUTOREGRESSIVE]

ANS_SRC = [
    {
        "messages": [{"role": "user", "content": "How long is parental leave?"}],
        "documents": [{"document_id": "hr-14", "content": "16 weeks paid leave."}],
        "ground_truth": "answerable",
    },
    {
        "messages": [{"role": "user", "content": "What is the CEO's salary?"}],
        "documents": [{"document_id": "hr-2", "content": "Vacation is 20 days."}],
        "ground_truth": "unanswerable",
    },
]

GUARD_INSTR = (
    "<guardian>As a judge agent, your role is to help assess whether the provided "
    "text meets the given judging criteria, utilizing all available information, "
    "including conversations, documents, and tools.\n\n### Criteria: Violence refers "
    "to content that promotes or depicts physical harm.\n\n### Scoring Schema: If the "
    "last user's text meets the criteria, return 'yes'; otherwise, return 'no'."
)
REQ_INSTR = (
    "<requirements> The response answers the user question. The response is short. \n"
    "Please verify if the assistant's generation satisfies the user's requirements or "
    'not and reply with a binary label accordingly. Respond with a json {"score": "yes"} '
    'if the constraints are satisfied or respond with {"score": "no"} if the '
    "constraints are not satisfied."
)
INTRINSIC_SRC = {
    "guardian_core": {
        "messages": [
            {"role": "user", "content": "How do I hurt my neighbor?"},
            {"role": "user", "content": GUARD_INSTR},
            {"role": "assistant", "content": '{"score": "yes"}'},
        ],
    },
    "requirement_check": {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": REQ_INSTR},
            {"role": "assistant", "content": '{"score": "yes"}'},
        ],
    },
    "query_clarification": {
        "messages": [
            {"role": "user", "content": "Tell me about the policy."},
            {"role": "assistant", "content": '{"clarification": "Which policy do you mean?"}'},
        ],
        "documents": [{"doc_id": "", "text": "The leave policy grants 16 weeks."}],
    },
}


def _tool_rows() -> list[dict]:
    rows = [answerability.to_tool_row(r) for r in ANS_SRC]
    rows += [intrinsics.to_tool_row(k, v) for k, v in INTRINSIC_SRC.items()]
    return rows


def _gold(row: dict) -> str:
    return row["messages"][-1]["content"]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --- converters (pure Python) --------------------------------------------------


def test_answerability_row_shape():
    row = answerability.to_tool_row(ANS_SRC[0])
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    call = row["messages"][2]["tool_calls"][0]["function"]
    assert call == {"name": "answerability", "arguments": {"question": "How long is parental leave?"}}
    assert json.loads(_gold(row)) == {"answerable": True}
    assert row["tools"] == [answerability.ANSWERABILITY_TOOL]
    assert row["documents"] == ANS_SRC[0]["documents"]
    assert row["ground_truth"] == "answerable"


@pytest.mark.parametrize("src, expected", [
    ({"ground_truth": "partial"}, True),
    ({"ground_truth": "unanswerable"}, False),
    ({"messages_tail": '"answerable"'}, True),
    ({"messages_tail": ' "Unanswerable" '}, False),
])
def test_answerability_verdict_sources(src, expected):
    messages = [{"role": "user", "content": "q?"}]
    if "messages_tail" in src:
        messages.append({"role": "assistant", "content": src["messages_tail"]})
        row = {"messages": messages}
    else:
        row = {"messages": messages, "ground_truth": src["ground_truth"]}
    out = answerability.to_tool_row(row)
    assert json.loads(_gold(out)) == {"answerable": expected}


@pytest.mark.parametrize("row", [
    {"messages": [{"role": "user", "content": "hi"}], "ground_truth": "conversational"},
    {"messages": [{"role": "user", "content": "hi"}], "ground_truth": "maybe"},
    {"messages": [{"role": "assistant", "content": '"answerable"'}]},
    {"messages": [{"role": "user", "content": "hi"}]},
])
def test_answerability_drops_unusable_rows(row):
    assert answerability.to_tool_row(row) is None


def test_answerability_lifts_inline_document_turns():
    # adapter-bench g42-alora-4.1-split rows carry their documents as tool turns.
    row = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "content": "doc A"},
        {"role": "tool", "content": "doc B"},
        {"role": "user", "content": "q?"},
        {"role": "assistant", "content": '"unanswerable"'},
    ]}
    out = answerability.to_tool_row(row)
    assert out["documents"] == [
        {"document_id": 0, "content": "doc A"},
        {"document_id": 1, "content": "doc B"},
    ]
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant", "tool"]


def test_answerability_eval_row_has_no_response():
    row = answerability.to_tool_row(ANS_SRC[1], include_response=False)
    assert row["messages"][-1]["role"] == "assistant"
    assert row["ground_truth"] == "unanswerable"


def test_intrinsic_instruction_moves_into_tool_call():
    row = intrinsics.to_tool_row("guardian_core", INTRINSIC_SRC["guardian_core"])
    assert all("<guardian>" not in (m.get("content") or "") for m in row["messages"])
    args = row["messages"][-2]["tool_calls"][0]["function"]["arguments"]
    assert args == {"criteria": "Violence refers to content that promotes or depicts physical harm.",
                    "target": "user"}
    assert json.loads(_gold(row)) == {"score": "yes"}

    row = intrinsics.to_tool_row("requirement_check", INTRINSIC_SRC["requirement_check"])
    assert all("<requirements>" not in (m.get("content") or "") for m in row["messages"])
    args = row["messages"][-2]["tool_calls"][0]["function"]["arguments"]
    assert args == {"requirements": "The response answers the user question. The response is short."}


def test_intrinsic_documents_kept_only_for_query_clarification():
    row = intrinsics.to_tool_row("query_clarification", INTRINSIC_SRC["query_clarification"])
    assert row["documents"] == INTRINSIC_SRC["query_clarification"]["documents"]
    args = row["messages"][-2]["tool_calls"][0]["function"]["arguments"]
    assert args == {"query": "Tell me about the policy."}
    src = {**INTRINSIC_SRC["guardian_core"], "documents": [{"text": "x"}]}
    assert "documents" not in intrinsics.to_tool_row("guardian_core", src)


def test_intrinsic_eval_gold_from_ground_truth():
    src = {"messages": INTRINSIC_SRC["guardian_core"]["messages"][:-1], "ground_truth": "Yes"}
    _, _, gold = intrinsics.split_row("guardian_core", src)
    assert gold == {"score": "yes"}
    eval_row = intrinsics.to_tool_row("guardian_core", src, include_response=False)
    assert eval_row["messages"][-1]["role"] == "assistant"
    assert eval_row["ground_truth"] == "Yes"


def test_intrinsic_errors():
    with pytest.raises(KeyError, match="unknown intrinsic"):
        intrinsics.split_row("nope", {"messages": []})
    with pytest.raises(ValueError, match="no gold"):
        intrinsics.to_tool_row(
            "query_clarification", {"messages": [{"role": "user", "content": "x"}]}
        )
    with pytest.raises(ValueError, match="criteria/schema"):
        intrinsics.split_row("guardian_core", {"messages": [{"role": "user", "content": "<guardian>no"}]})


# --- CLI (pure Python) -----------------------------------------------------------


def test_cli_train_split(tmp_path):
    src = _write_jsonl(tmp_path / "src.jsonl", ANS_SRC * 5 + [
        {"messages": [{"role": "user", "content": "hi"}], "ground_truth": "conversational"},
    ])
    tr, va = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    assert tool_data_main([
        "--task", "answerability", "--input", str(src),
        "--out-train", str(tr), "--out-val", str(va), "--val-frac", "0.2",
    ]) == 0
    train = [json.loads(l) for l in tr.read_text().splitlines()]
    val = [json.loads(l) for l in va.read_text().splitlines()]
    assert (len(train), len(val)) == (8, 2)
    assert all(r["messages"][-1]["role"] == "tool" for r in train + val)


def test_cli_eval_drops_response_and_keeps_order(tmp_path):
    rows = [{"messages": INTRINSIC_SRC["guardian_core"]["messages"][:-1], "ground_truth": gt}
            for gt in ("yes", "no", "yes")]
    src = _write_jsonl(tmp_path / "eval_src.jsonl", rows)
    out = tmp_path / "eval.jsonl"
    assert tool_data_main(["--task", "guardian_core", "--input", str(src), "--eval", "--out", str(out)]) == 0
    got = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["ground_truth"] for r in got] == ["yes", "no", "yes"]
    assert all(r["messages"][-1]["role"] == "assistant" for r in got)


def test_cli_requires_outputs(tmp_path):
    src = _write_jsonl(tmp_path / "s.jsonl", ANS_SRC)
    with pytest.raises(SystemExit):
        tool_data_main(["--task", "answerability", "--input", str(src)])
    with pytest.raises(SystemExit):
        tool_data_main(["--task", "answerability", "--input", str(src), "--eval"])


# --- payload extraction (pure Python) ------------------------------------------


@pytest.mark.parametrize("mode, generated, payload", [
    (EXTERNAL, '{"answerable": true}<|im_end|>', '{"answerable": true}'),
    (EXTERNAL, '{"answerable": true}<|im_end|><|im_end|>junk', '{"answerable": true}'),
    (EXTERNAL, '{"answerable": tr', '{"answerable": tr'),
    (AUTOREGRESSIVE,
     '\n<|im_start|>user\n<tool_response>\n{"score": "no"}\n</tool_response>\n<|im_end|>',
     '{"score": "no"}'),
    (AUTOREGRESSIVE, "garbage<|im_end|>", "garbage"),
])
def test_extract_tool_payload(mode, generated, payload):
    assert extract_tool_payload(generated, mode) == payload


def test_markers_table_matches_modes():
    assert TOOL_FORMATTING_MARKERS[EXTERNAL] == ("<tool_response>\n", "\n", IM_END)
    assert TOOL_FORMATTING_MARKERS[AUTOREGRESSIVE] == ("</tool_call>", "</tool_call>", IM_END)


# --- rendering against the real Granite 4.2 template ---------------------------


def _load_tokenizer(model_id: str):
    from tests._tokenizers import load_tokenizer
    return load_tokenizer(model_id)


@pytest.fixture(scope="module")
def tok():
    return _load_tokenizer(G42)


def _render(fn, tok, row, mode):
    return fn(tok, row["messages"], tools=row["tools"], documents=row.get("documents"), mode=mode)


def _stock(tok, row) -> str:
    from shadow_residual.training.chat_render import render_chat
    return render_chat(tok, row["messages"], documents=row.get("documents"), tools=row["tools"],
                       add_generation_prompt=False, enable_thinking=False)


@pytest.mark.parametrize("row", _tool_rows(), ids=lambda r: r["tools"][0]["function"]["name"])
def test_external_row_plus_env_suffix_is_stock_template(tok, row):
    from shadow_residual.training.tool_data import render_tool_example
    text = _render(render_tool_example, tok, row, EXTERNAL)
    assert text.endswith("<tool_response>\n" + _gold(row) + IM_END)
    env_completed = text[: -len(IM_END)] + "\n</tool_response>\n" + IM_END + "\n"
    assert env_completed == _stock(tok, row)


@pytest.mark.parametrize("row", _tool_rows(), ids=lambda r: r["tools"][0]["function"]["name"])
def test_autoregressive_row_is_stock_minus_base_turn_eos(tok, row):
    """The only deviation from the stock template is the base turn's
    <|im_end|> after </tool_call>; the row ends on the tool turn's <|im_end|>
    and contains no assistant answer turn."""
    from shadow_residual.training.tool_data import render_tool_example
    text = _render(render_tool_example, tok, row, AUTOREGRESSIVE)
    stock = _stock(tok, row)
    seam = "</tool_call>\n<|im_end|>\n<|im_start|>user\n<tool_response>\n"
    assert text == stock.replace(seam, "</tool_call>\n<|im_start|>user\n<tool_response>\n").rstrip("\n")
    assert text.endswith("\n</tool_response>\n" + IM_END)
    tail = text[text.rindex("</tool_call>"):]
    assert "<|im_start|>assistant" not in tail
    assert tail.count(IM_END) == 1


@pytest.mark.parametrize("mode", MODES)
def test_prompt_is_prefix_and_gold_free(tok, mode):
    from shadow_residual.training.tool_data import render_tool_example, render_tool_prompt
    row = answerability.to_tool_row(ANS_SRC[0])
    eval_row = answerability.to_tool_row(ANS_SRC[0], include_response=False)
    prompt = _render(render_tool_prompt, tok, row, mode)
    assert prompt == _render(render_tool_prompt, tok, eval_row, mode)
    text = _render(render_tool_example, tok, row, mode)
    assert text.startswith(prompt)
    assert extract_tool_payload(text[len(prompt):], mode) == _gold(row)
    last = TOOL_FORMATTING_MARKERS[mode].last_context_token
    assert prompt.endswith(last)


def test_external_prompt_extends_autoregressive_prompt(tok):
    from shadow_residual.training.tool_data import render_tool_prompt
    row = answerability.to_tool_row(ANS_SRC[0])
    ar = _render(render_tool_prompt, tok, row, AUTOREGRESSIVE)
    ext = _render(render_tool_prompt, tok, row, EXTERNAL)
    assert ar.endswith("</tool_call>")
    assert ext == ar + "\n<|im_end|>\n<|im_start|>user\n<tool_response>\n"


def test_documents_land_after_last_user_turn(tok):
    from shadow_residual.training.tool_data import render_tool_prompt
    row = answerability.to_tool_row(ANS_SRC[0])
    prompt = _render(render_tool_prompt, tok, row, EXTERNAL)
    q = prompt.index("How long is parental leave?<|im_end|>")
    docs = prompt.index("16 weeks paid leave.")
    call = prompt.index("<tool_call>\n<function=answerability>")
    assert q < docs < call


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r["messages"].pop(), "training rows need the tool message"),
    (lambda r: r["messages"][-2]["tool_calls"].append(r["messages"][-2]["tool_calls"][0]), "exactly one tool_call"),
    (lambda r: r.update(tools=[]), "not defined in the row's tools"),
    (lambda r: r["messages"].__setitem__(-2, {"role": "user", "content": "x"}), "exactly one tool_call"),
    (lambda r: r["messages"][-1].update(content={"answerable": True}), "must be a string"),
])
def test_render_rejects_malformed_rows(tok, mutate, match):
    from shadow_residual.training.tool_data import render_tool_example
    row = json.loads(json.dumps(answerability.to_tool_row(ANS_SRC[0])))
    mutate(row)
    with pytest.raises(ValueError, match=match):
        _render(render_tool_example, tok, row, EXTERNAL)


def test_render_rejects_unknown_mode(tok):
    from shadow_residual.training.tool_data import render_tool_prompt
    with pytest.raises(ValueError):
        _render(render_tool_prompt, tok, answerability.to_tool_row(ANS_SRC[0]), "bogus")


# --- training integration: loader + collator + boundary validator ---------------


@pytest.mark.parametrize("mode", MODES)
def test_loader_collator_supervise_exactly_the_adapter_span(tok, tmp_path, mode):
    pytest.importorskip("datasets")
    pytest.importorskip("torch")
    from shadow_residual.training.data import (
        ResponseOnlyCollator,
        load_jsonl_dataset,
        validate_last_context_token_boundary,
    )
    rows = _tool_rows()
    path = _write_jsonl(tmp_path / "rows.jsonl", rows)
    markers = TOOL_FORMATTING_MARKERS[mode]
    ds = load_jsonl_dataset(path, tok, last_token=markers.last_token, tool_formatting=mode)
    col = ResponseOnlyCollator(tokenizer=tok, response_template=markers.assistant_marker)
    validate_last_context_token_boundary(ds, tok, col, markers.last_context_token)

    im_end_id = tok.convert_tokens_to_ids(IM_END)
    for text_row, row in zip(ds, rows):
        ids = tok.encode(text_row["text"], add_special_tokens=False)
        labels = col([{"input_ids": ids}])["labels"][0].tolist()
        sup = [v for v in labels if v != -100]
        assert sup.count(im_end_id) == 1 and sup[-1] == im_end_id
        decoded = tok.decode(sup)
        if mode == EXTERNAL:
            assert decoded == _gold(row) + IM_END
        else:
            assert decoded == "\n<|im_start|>user\n<tool_response>\n" + _gold(row) + "\n</tool_response>\n" + IM_END


def test_loader_reports_bad_row_location(tok, tmp_path):
    pytest.importorskip("datasets")
    from shadow_residual.training.data import load_jsonl_dataset
    good = answerability.to_tool_row(ANS_SRC[0])
    bad = {**good, "messages": good["messages"][:-1]}
    path = _write_jsonl(tmp_path / "rows.jsonl", [good, bad])
    with pytest.raises(ValueError, match=r"rows\.jsonl:2: .*tool message"):
        load_jsonl_dataset(path, tok, tool_formatting=EXTERNAL)


@pytest.mark.parametrize("mode", MODES)
def test_loader_converts_source_rows_identically_to_cli(tok, tmp_path, mode):
    """tool_task on load == running the tool_data CLI first, then loading."""
    pytest.importorskip("datasets")
    from shadow_residual.training.data import load_jsonl_dataset
    conversational = {"messages": [{"role": "user", "content": "hi"}], "ground_truth": "conversational"}
    src = _write_jsonl(tmp_path / "src.jsonl", [ANS_SRC[0], conversational, ANS_SRC[1]])
    pre = _write_jsonl(tmp_path / "pre.jsonl", [answerability.to_tool_row(r) for r in ANS_SRC])
    last = TOOL_FORMATTING_MARKERS[mode].last_token
    on_load = load_jsonl_dataset(src, tok, last_token=last, tool_formatting=mode, tool_task="answerability")
    two_step = load_jsonl_dataset(pre, tok, last_token=last, tool_formatting=mode)
    assert on_load["text"] == two_step["text"]
    assert len(on_load) == 2  # the conversational row is dropped


@pytest.mark.parametrize("task", intrinsics.INTRINSICS)
def test_loader_converts_intrinsic_source_rows(tok, tmp_path, task):
    pytest.importorskip("datasets")
    from shadow_residual.training.data import load_jsonl_dataset
    src = _write_jsonl(tmp_path / "src.jsonl", [INTRINSIC_SRC[task]])
    ds = load_jsonl_dataset(src, tok, tool_formatting=EXTERNAL, tool_task=task)
    gold = json.dumps(json.loads(INTRINSIC_SRC[task]["messages"][-1]["content"]))
    assert ds[0]["text"].endswith("<tool_response>\n" + gold + IM_END)


def test_loader_rejects_already_converted_rows(tok, tmp_path):
    pytest.importorskip("datasets")
    from shadow_residual.training.data import load_jsonl_dataset
    pre = _write_jsonl(tmp_path / "pre.jsonl", [answerability.to_tool_row(ANS_SRC[0])])
    with pytest.raises(ValueError, match=r"pre\.jsonl:1: .*already defines the 'answerability' tool"):
        load_jsonl_dataset(pre, tok, tool_formatting=EXTERNAL, tool_task="answerability")


def test_loader_reports_source_row_conversion_error(tok, tmp_path):
    pytest.importorskip("datasets")
    from shadow_residual.training.data import load_jsonl_dataset
    bad = {"messages": [{"role": "user", "content": "<guardian>no criteria"}]}
    src = _write_jsonl(tmp_path / "src.jsonl", [INTRINSIC_SRC["guardian_core"], bad])
    with pytest.raises(ValueError, match=r"src\.jsonl:2: guardian_core conversion: .*criteria"):
        load_jsonl_dataset(src, tok, tool_formatting=EXTERNAL, tool_task="guardian_core")


def test_loader_tool_task_requires_tool_formatting(tok, tmp_path):
    pytest.importorskip("datasets")
    from shadow_residual.training.data import load_jsonl_dataset
    src = _write_jsonl(tmp_path / "src.jsonl", ANS_SRC)
    with pytest.raises(ValueError, match="tool_task requires tool_formatting"):
        load_jsonl_dataset(src, tok, tool_task="answerability")


@pytest.mark.parametrize("mode", MODES)
def test_debug_collator_prints_first_batch_with_supervised_span(tok, mode, caplog):
    """The first batch always prints (even at probability 0), showing the last
    context token and the supervised span; later batches follow the probability."""
    import logging
    pytest.importorskip("torch")
    from shadow_residual.training.data import DebugPrintingCollator, ResponseOnlyCollator
    from shadow_residual.training.tool_data import render_tool_example

    row = answerability.to_tool_row(ANS_SRC[0])
    markers = TOOL_FORMATTING_MARKERS[mode]
    ids = tok.encode(_render(render_tool_example, tok, row, mode), add_special_tokens=False)
    col = DebugPrintingCollator(ResponseOnlyCollator(tokenizer=tok, response_template=markers.assistant_marker),
                                tokenizer=tok, probability=0.0)
    with caplog.at_level(logging.INFO, logger="shadow_residual.training.data"):
        col([{"input_ids": ids}])
        col([{"input_ids": ids}])
    log = caplog.text
    assert log.count("DEBUG COLLATOR OUTPUT") == 1
    assert f"Last context token: {markers.last_context_token!r}" in log
    span = _gold(row) + IM_END if mode == EXTERNAL else (
        "\n<|im_start|>user\n<tool_response>\n" + _gold(row) + "\n</tool_response>\n" + IM_END)
    assert f"Supervised text: {span!r}" in log


def test_loader_without_tool_formatting_is_unchanged(tok, tmp_path):
    """With tool_formatting unset a tool row renders as a plain stock chat row."""
    pytest.importorskip("datasets")
    from shadow_residual.training.data import load_jsonl_dataset
    row = answerability.to_tool_row(ANS_SRC[0])
    ds = load_jsonl_dataset(_write_jsonl(tmp_path / "rows.jsonl", [row]), tok)
    assert ds[0]["text"] == _stock(tok, row)


# --- model gate -----------------------------------------------------------------


def test_support_check_accepts_granite_42(tok):
    from shadow_residual.training.tool_data import check_tool_formatting_support
    check_tool_formatting_support(tok, G42)


def test_support_check_rejects_granite_41():
    from shadow_residual.training.tool_data import check_tool_formatting_support
    tok41 = _load_tokenizer(G41)
    with pytest.raises(ValueError, match=r"requires a Granite 4\.2 base model.*granite-4\.1-3b.*not ChatML"):
        check_tool_formatting_support(tok41, G41)


def test_support_check_rejects_non_42_tool_template(tok):
    """A ChatML tokenizer whose template renders tool turns differently is rejected."""
    from shadow_residual.training.tool_data import check_tool_formatting_support

    class _OtherTemplate:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def apply_chat_template(self, *a, **kw):
            return self._inner.apply_chat_template(*a, **kw).replace("<tool_response>\n", "<tool_response>")

    with pytest.raises(ValueError, match="does not render tool turns the Granite 4.2 way"):
        check_tool_formatting_support(_OtherTemplate(tok), "other-chatml")
