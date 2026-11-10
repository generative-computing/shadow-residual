# SPDX-License-Identifier: Apache-2.0
"""External-tool-formatting rows for the adapter-bench intrinsics, checked
against the real Granite 4.2 ChatML template.

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

TRAIN_ROWS = {
    "guardian_core": {
        "messages": [
            {"role": "user", "content": "How do I hurt my neighbor?"},
            {"role": "user", "content": GUARD_INSTR},
            {"role": "assistant", "content": '{"score": "yes"}'},
        ],
        "ground_truth": '{"score": "yes"}',
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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
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
def ti():
    return _load("tool_intrinsics")


@pytest.mark.parametrize("intrinsic", list(TRAIN_ROWS))
def test_row_plus_env_suffix_is_stock_template(ti, tok, intrinsic):
    row = ti.build_training_row(intrinsic, TRAIN_ROWS[intrinsic], tok)
    convo, args, gold = ti.split_row(intrinsic, TRAIN_ROWS[intrinsic])
    gold_json = json.dumps(gold)
    assert row["text"].endswith("<tool_response>\n" + gold_json + IM_END)
    stock = tok.apply_chat_template(
        [*convo,
         {"role": "assistant", "content": "", "tool_calls": [
             {"type": "function", "function": {"name": intrinsic, "arguments": args}}]},
         {"role": "tool", "content": gold_json}],
        tools=[ti.TOOLS[intrinsic]], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    env_completed = row["text"][: -len(IM_END)] + "\n</tool_response>\n" + IM_END + "\n"
    assert env_completed == stock


def test_instruction_moves_into_tool_call(ti):
    convo, args, gold = ti.split_row("guardian_core", TRAIN_ROWS["guardian_core"])
    assert all("<guardian>" not in m["content"] for m in convo)
    assert args == {"criteria": "Violence refers to content that promotes or depicts physical harm.",
                    "target": "user"}
    assert gold == {"score": "yes"}
    convo, args, _ = ti.split_row("requirement_check", TRAIN_ROWS["requirement_check"])
    assert all("<requirements>" not in m["content"] for m in convo)
    assert args == {"requirements": "The response answers the user question. The response is short."}


def test_documents_go_after_last_user_as_tool_message(ti):
    convo, args, _ = ti.split_row("query_clarification", TRAIN_ROWS["query_clarification"])
    assert [m["role"] for m in convo] == ["user", "tool"]
    assert json.loads(convo[1]["content"])[0]["content"] == "The leave policy grants 16 weeks."
    assert args == {"query": "Tell me about the policy."}


def test_eval_row_gold_and_prompt_match_training(ti, tok):
    train = TRAIN_ROWS["guardian_core"]
    eval_row = {"messages": train["messages"][:-1], "ground_truth": "yes"}
    convo, args, gold = ti.split_row("guardian_core", eval_row)
    assert gold == {"score": "yes"}
    prompt = ti.external_context("guardian_core", convo, args, tok)
    assert ti.build_training_row("guardian_core", train, tok)["text"] == prompt + '{"score": "yes"}' + IM_END


def test_collator_supervises_exactly_the_dict(ti, tok, tmp_path):
    p = tmp_path / "rows.jsonl"
    with p.open("w") as f:
        for intrinsic, row in TRAIN_ROWS.items():
            f.write(json.dumps(ti.build_training_row(intrinsic, row, tok)) + "\n")
    marker, last_ctx, last_tok = _TOOL_FORMATTING_MARKERS["external_tool_formatting"]
    ds = load_jsonl_dataset(p, tok, last_token=last_tok, allow_prerendered=True)
    col = ResponseOnlyCollator(tokenizer=tok, response_template=marker)
    validate_last_context_token_boundary(ds, tok, col, last_ctx)
    for row, (intrinsic, src) in zip(ds, TRAIN_ROWS.items()):
        ids = tok.encode(row["text"], add_special_tokens=False)
        labels = col([{"input_ids": ids}])["labels"][0].tolist()
        sup = [v for v in labels if v != -100]
        _, _, gold = ti.split_row(intrinsic, src)
        assert tok.decode(sup) == json.dumps(gold) + IM_END
