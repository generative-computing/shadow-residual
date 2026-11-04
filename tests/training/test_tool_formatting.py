# SPDX-License-Identifier: Apache-2.0
"""SR-adapter-as-tool data layout vs. the real Granite 4.2 ChatML template.

Skipped unless the 4.2 tokenizer is loadable (local HF cache or hub access).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("torch")

from shadow_residual.training.data import (  # noqa: E402
    ResponseOnlyCollator,
    load_jsonl_dataset,
    validate_last_context_token_boundary,
)
from shadow_residual.training.train import _TOOL_FORMATTING_MARKERS  # noqa: E402

G42 = "ibm-research/granite-4.2-3b-prerelease-r260622a"
REPO = Path(__file__).resolve().parents[2]
IM_END = "<|im_end|>"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_tool_adapter_data", REPO / "scripts" / "make_tool_adapter_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(G42)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"granite 4.2 tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def gen():
    return _load_generator()


SRC_ROWS = [
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


def _stock_render(gen, tok, src, verdict_json):
    q = gen._last_user_question(src["messages"])
    msgs = [
        {"role": "system", "content": gen.SYSTEM},
        {"role": "user", "content": q},
        {"role": "tool", "content": json.dumps(gen._normalize_documents(src["documents"]))},
        {"role": "assistant", "content": "", "tool_calls": [{
            "type": "function",
            "function": {"name": "answerability", "arguments": {"question": q}},
        }]},
        {"role": "tool", "content": verdict_json},
    ]
    return tok.apply_chat_template(
        msgs, tools=[gen.ANSWERABILITY_TOOL], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )


@pytest.mark.parametrize("src", SRC_ROWS)
def test_external_row_plus_env_suffix_is_stock_template(gen, tok, src):
    row = gen.build_row(src, tok, tool_formatting=gen.EXTERNAL)
    text = row["text"]
    verdict = json.dumps({"answerable": src["ground_truth"] == "answerable"})
    assert text.endswith("<tool_response>\n" + verdict + IM_END)
    env_completed = text[: -len(IM_END)] + "\n</tool_response>\n" + IM_END + "\n"
    assert env_completed == _stock_render(gen, tok, src, verdict)


@pytest.mark.parametrize("src", SRC_ROWS)
def test_external_context_extends_autoregressive_context(gen, tok, src):
    ext = gen.build_row(src, tok, tool_formatting=gen.EXTERNAL)["text"]
    ar = gen.build_row(src, tok, tool_formatting=gen.AUTOREGRESSIVE)["text"]
    ar_ctx = ar[: ar.rindex("</tool_call>") + len("</tool_call>")]
    assert ext.startswith(ar_ctx + "\n" + IM_END + "\n<|im_start|>user\n<tool_response>\n")


@pytest.mark.parametrize("mode", ["autoregressive_tool_formatting", "external_tool_formatting"])
def test_collator_boundary_per_mode(gen, tok, tmp_path, mode):
    p = tmp_path / "rows.jsonl"
    with p.open("w") as f:
        for src in SRC_ROWS:
            f.write(json.dumps(gen.build_row(src, tok, tool_formatting=mode)) + "\n")
    marker, last_ctx, last_tok = _TOOL_FORMATTING_MARKERS[mode]
    ds = load_jsonl_dataset(p, tok, last_token=last_tok, allow_prerendered=True)
    col = ResponseOnlyCollator(tokenizer=tok, response_template=marker)
    validate_last_context_token_boundary(ds, tok, col, last_ctx)

    im_end_id = tok.convert_tokens_to_ids(IM_END)
    for row, src in zip(ds, SRC_ROWS):
        ids = tok.encode(row["text"], add_special_tokens=False)
        labels = col([{"input_ids": ids}])["labels"][0].tolist()
        sup = [v for v in labels if v != -100]
        assert sup.count(im_end_id) == 1 and sup[-1] == im_end_id
        decoded = tok.decode(sup)
        verdict = json.dumps({"answerable": src["ground_truth"] == "answerable"})
        if mode == "external_tool_formatting":
            assert decoded == verdict + IM_END
        else:
            assert decoded.startswith("\n<|im_start|>user\n<tool_response>\n" + verdict)
            assert decoded.endswith("<|im_start|>assistant\n" + json.dumps(src["ground_truth"]) + IM_END)


def test_eval_external_context_and_parser(gen, tok):
    spec = importlib.util.spec_from_file_location(
        "eval_tool_adapter", REPO / "scripts" / "eval_tool_adapter.py"
    )
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    row = gen.build_row(SRC_ROWS[0], tok, tool_formatting=gen.EXTERNAL)
    prompt, gold = ev.build_context(tok, row)
    assert gold is True
    assert prompt.endswith("<tool_response>\n")
    assert row["text"][len(prompt):] == '{"answerable": true}' + IM_END
    assert ev.parse_external_verdict('{"answerable": false}<|im_end|>') == {"answerable": False}
    assert ev.parse_external_verdict('{"answerable": true}\n</tool_response>') is None
