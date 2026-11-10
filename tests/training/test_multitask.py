# SPDX-License-Identifier: Apache-2.0
"""CPU tests for shared-A multi-task SR training (training/multitask.py)."""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.config_helpers import set_shadow_residual
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.training.multitask import (
    MultiTaskCollator,
    parse_task_specs,
    setup_shared_a,
)

TASKS = ["ans", "guard", "qc", "req"]
TARGETS = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "cross_stream"]


def _tiny_sr(seed=0):
    cfg = ShadowResidualConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_adapters=0,
        max_lora_rank=4, switch_head_dim=16,
    )
    set_shadow_residual(cfg, enabled=True)
    torch.manual_seed(seed)
    return ShadowResidualForCausalLM(cfg).eval()


def _lora_cfg():
    return LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")


def _shared_model():
    model = get_peft_model(_tiny_sr(), _lora_cfg(), adapter_name=TASKS[0])
    router = setup_shared_a(model, TASKS, _lora_cfg())
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.2)
    return model.eval(), router


def _lora_layers(model):
    from peft.tuners.lora.layer import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def test_a_is_tied_and_b_is_distinct():
    model, _ = _shared_model()
    layers = _lora_layers(model)
    assert len(layers) == 2 * len(TARGETS)
    for layer in layers:
        a0 = layer.lora_A[TASKS[0]].weight
        for t in TASKS[1:]:
            assert layer.lora_A[t].weight is a0
            assert layer.lora_B[t].weight is not layer.lora_B[TASKS[0]].weight
    n_params = {id(p) for n, p in model.named_parameters() if "lora_" in n and p.requires_grad}
    expected = len(layers) * (1 + len(TASKS))
    assert len(n_params) == expected


def test_routed_forward_matches_single_adapter_forward(tmp_path):
    """Each row of a mixed batch must equal running its task's adapter alone."""
    model, router = _shared_model()
    ids = torch.randint(0, 300, (4, 7))
    router.set(torch.tensor([0, 1, 2, 3]))
    with torch.no_grad():
        mixed = model(input_ids=ids).logits

    model.save_pretrained(tmp_path)
    for t, name in enumerate(TASKS):
        ref = PeftModel.from_pretrained(_tiny_sr(), tmp_path / name).eval()
        with torch.no_grad():
            single = ref(input_ids=ids[t : t + 1]).logits
        torch.testing.assert_close(mixed[t : t + 1], single, rtol=1e-4, atol=1e-4)


def test_exports_share_identical_a(tmp_path):
    from safetensors.torch import load_file
    model, _ = _shared_model()
    model.save_pretrained(tmp_path)
    sds = {t: load_file(str(tmp_path / t / "adapter_model.safetensors")) for t in TASKS}
    a_keys = [k for k in sds[TASKS[0]] if "lora_A" in k]
    b_keys = [k for k in sds[TASKS[0]] if "lora_B" in k]
    assert len(a_keys) == len(b_keys) == 2 * len(TARGETS)
    for t in TASKS[1:]:
        assert set(sds[t]) == set(sds[TASKS[0]])
        for k in a_keys:
            assert torch.equal(sds[t][k], sds[TASKS[0]][k])
        assert any(not torch.equal(sds[t][k], sds[TASKS[0]][k]) for k in b_keys)


def test_gradients_route_to_own_b_only():
    model, router = _shared_model()
    model.train()
    ids = torch.randint(0, 300, (2, 6))
    router.set(torch.tensor([1, 1]))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    for layer in _lora_layers(model):
        assert layer.lora_A[TASKS[0]].weight.grad is not None
        assert layer.lora_A[TASKS[0]].weight.grad.abs().sum() > 0
        for t, name in enumerate(TASKS):
            g = layer.lora_B[name].weight.grad
            assert g is not None, "every B must stay in the graph (DDP, no unused params)"
            if t == 1:
                assert g.abs().sum() > 0
            else:
                assert torch.count_nonzero(g) == 0


def test_base_stream_untouched_and_bare_matches_base():
    model, router = _shared_model()
    ids = torch.randint(0, 300, (2, 5))
    base = _tiny_sr()
    with torch.no_grad():
        ref = base(input_ids=ids).logits
        with model.disable_adapter():
            bare = model(input_ids=ids).logits
    torch.testing.assert_close(bare, ref, rtol=0, atol=0)


def test_missing_route_raises():
    model, router = _shared_model()
    router.clear()
    with pytest.raises(RuntimeError, match="task route"):
        model(input_ids=torch.randint(0, 300, (1, 4)))


def test_collator_adds_task_ids():
    def base(features):
        return {"input_ids": torch.tensor([f["input_ids"] for f in features])}
    out = MultiTaskCollator(base)([{"input_ids": [1, 2], "task_id": 3}, {"input_ids": [4, 5], "task_id": 0}])
    assert out["task_ids"].tolist() == [3, 0]


def test_parse_task_specs():
    assert parse_task_specs(["a:/x/t.jsonl:/x/v.jsonl"]) == [("a", "/x/t.jsonl", "/x/v.jsonl")]
    with pytest.raises(ValueError):
        parse_task_specs(["a:/x"])
    with pytest.raises(ValueError):
        parse_task_specs(["a:t:v", "a:t:v"])
