# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the shared-trainable-A / fixed-B step (training/train_shared_a_fixed_b.py)."""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.config_helpers import set_shadow_residual
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.training import multitask
from shadow_residual.training.frozen_a import A_FILE, site_key
from shadow_residual.training.train_shared_a_fixed_b import (
    _patch_setup,
    check_exports,
    export_init,
    install_shared_a_fixed_b,
    parse_fixed_b,
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


def _lora_layers(model):
    from peft.tuners.lora.layer import LoraLayer
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer)]


@pytest.fixture
def setup(tmp_path):
    """A bank of random A's and four source adapters with distinct random A and B."""
    model = get_peft_model(_tiny_sr(), _lora_cfg())
    gen = torch.Generator().manual_seed(7)
    bank = {site_key(n): torch.randn(m.lora_A["default"].weight.shape, generator=gen) * 0.2
            for n, m in _lora_layers(model)}
    (tmp_path / "bank").mkdir()
    save_file(bank, str(tmp_path / "bank" / A_FILE))
    fixed_b = {}
    for t, name in enumerate(TASKS):
        src = get_peft_model(_tiny_sr(), _lora_cfg())
        torch.manual_seed(100 + t)
        with torch.no_grad():
            for n, p in src.named_parameters():
                if "lora_" in n:
                    p.normal_(0, 0.2)
        src.save_pretrained(tmp_path / f"src_{name}")
        fixed_b[name] = str(tmp_path / f"src_{name}")
    return tmp_path, str(tmp_path / "bank"), fixed_b


def _model(bank, fixed_b):
    model = get_peft_model(_tiny_sr(), _lora_cfg(), adapter_name=TASKS[0])
    router = multitask.setup_shared_a(model, TASKS, _lora_cfg())
    install_shared_a_fixed_b(model, TASKS, bank, fixed_b)
    return model.eval(), router


def test_a_shared_trainable_b_frozen(setup):
    _, bank, fixed_b = setup
    model, _ = _model(bank, fixed_b)
    for _, layer in _lora_layers(model):
        a = layer.lora_A[TASKS[0]].weight
        assert a.requires_grad
        for t in TASKS:
            assert layer.lora_A[t].weight is a
            assert not layer.lora_B[t].weight.requires_grad
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(trainable) == len(_lora_layers(model))


def test_step0_equals_bank_a_with_source_b(setup):
    """Each row of the routed model equals the exported init adapter (bank A + source B) alone."""
    tmp, bank, fixed_b = setup
    model, router = _model(bank, fixed_b)
    ids = torch.randint(0, 300, (4, 7))
    router.set(torch.tensor([0, 1, 2, 3]))
    with torch.no_grad():
        mixed = model(input_ids=ids).logits
    export_init(tmp / "init", bank, fixed_b)
    for t, name in enumerate(TASKS):
        ref = PeftModel.from_pretrained(_tiny_sr(), tmp / "init" / name).eval()
        with torch.no_grad():
            single = ref(input_ids=ids[t : t + 1]).logits
        torch.testing.assert_close(mixed[t : t + 1], single, rtol=1e-4, atol=1e-4)


def test_every_task_trains_a_and_no_b(setup):
    _, bank, fixed_b = setup
    model, router = _model(bank, fixed_b)
    model.train()
    for t in range(len(TASKS)):
        model.zero_grad(set_to_none=True)
        ids = torch.randint(0, 300, (2, 6))
        router.set(torch.tensor([t, t]))
        model(input_ids=ids, labels=ids).loss.backward()
        for _, layer in _lora_layers(model):
            assert layer.lora_A[TASKS[0]].weight.grad.abs().sum() > 0
            for name in TASKS:
                assert layer.lora_B[name].weight.grad is None


def test_check_exports(setup):
    tmp, bank, fixed_b = setup
    model, _ = _model(bank, fixed_b)
    with pytest.raises(RuntimeError, match="did not train"):
        model.save_pretrained(tmp / "out0")
        check_exports(tmp / "out0", fixed_b, bank)
    with torch.no_grad():
        for _, layer in _lora_layers(model):
            layer.lora_A[TASKS[0]].weight.add_(0.01)
    model.save_pretrained(tmp / "out")
    stats = check_exports(tmp / "out", fixed_b, bank)
    assert stats["a_unchanged"] == 0 and stats["a_tensors"] == len(_lora_layers(model))
    swapped = dict(fixed_b, ans=fixed_b["guard"])
    with pytest.raises(RuntimeError, match="lora_B differ"):
        check_exports(tmp / "out", swapped, bank)


def test_patched_setup_rejects_untied_and_mismatched_tasks(setup, monkeypatch):
    _, bank, fixed_b = setup
    monkeypatch.setattr(multitask, "setup_shared_a", multitask.setup_shared_a)
    _patch_setup(bank, fixed_b)
    with pytest.raises(ValueError, match="untie"):
        multitask.setup_shared_a(get_peft_model(_tiny_sr(), _lora_cfg(), adapter_name="ans"),
                                 TASKS, _lora_cfg(), tie_a=False)
    _patch_setup(bank, {k: v for k, v in fixed_b.items() if k != "req"})
    with pytest.raises(ValueError, match="--fixed-b tasks"):
        multitask.setup_shared_a(get_peft_model(_tiny_sr(), _lora_cfg(), adapter_name="ans"),
                                 TASKS, _lora_cfg())


def test_parse_fixed_b():
    assert parse_fixed_b(["a:/x/y"]) == {"a": "/x/y"}
    with pytest.raises(ValueError):
        parse_fixed_b(["a"])
    with pytest.raises(ValueError):
        parse_fixed_b(["a:/x", "a:/y"])
