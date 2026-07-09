# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified training configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from shadow_residual.config import TrainingConfig, load_training_config
from shadow_residual.config.training_config import _resolve_env_vars

EXAMPLES = Path(__file__).resolve().parents[2] / "src/shadow_residual/config"


def _minimal(**overrides) -> dict:
    """Produce the smallest valid config payload, with optional top-level overrides."""
    cfg = {
        "model": {"base": "ibm-granite/granite-4.1-3b"},
        "data": {"train_path": "/tmp/t.jsonl", "val_path": "/tmp/v.jsonl"},
        "save": {"output_dir": "/tmp/out"},
    }
    cfg.update(overrides)
    return cfg


# --- Examples round-trip ------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "lora-qkvo-mlp-r16.yaml",
        "alora-all-linear-r16.yaml",
        "sr-qo-mlp-r32-c32-sharedkv.yaml",
    ],
)
def test_examples_validate(monkeypatch, name):
    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("RUN_DIR", "/runs/job1")
    monkeypatch.setenv("EVAL_DATA", "/data/eval.jsonl")
    cfg = load_training_config(EXAMPLES / name)
    assert cfg.model.base == "ibm-granite/granite-4.1-3b"
    assert cfg.save.output_dir == "/runs/job1/checkpoints"


def test_g42_config_validates(monkeypatch):
    """The Granite 4.2 variant: ${BASE_MODEL} interpolation + ChatML marker."""
    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("RUN_DIR", "/runs/job1")
    monkeypatch.setenv("EVAL_DATA", "/data/eval.jsonl")
    monkeypatch.setenv("BASE_MODEL", "/models/granite-4.2-3b")
    cfg = load_training_config(EXAMPLES / "sr-qo-mlp-r32-c32-sharedkv-g42.yaml")
    assert cfg.model.base == "/models/granite-4.2-3b"      # ${BASE_MODEL} resolved
    assert cfg.model.shared_base_kv is True
    assert cfg.adapter.invocation_tokens == "<|im_start|>assistant\n"  # ChatML marker


# --- adapter.target_modules: scalar vs dict -----------------------------------

def test_target_modules_scalar():
    cfg = TrainingConfig.model_validate(_minimal(adapter={"target_modules": 32}))
    assert cfg.adapter.target_modules == 32
    assert cfg.adapter.alpha == 64  # default 2 * rank


def test_target_modules_per_module_dict():
    cfg = TrainingConfig.model_validate(
        _minimal(adapter={"target_modules": {"q_proj": 16, "gate_proj": 64}})
    )
    assert cfg.adapter.target_modules == {"q_proj": 16, "gate_proj": 64}
    # alpha defaults to 2 * max rank
    assert cfg.adapter.alpha == 128


def test_target_modules_alpha_explicit_overrides():
    cfg = TrainingConfig.model_validate(
        _minimal(adapter={"target_modules": 16, "alpha": 64})
    )
    assert cfg.adapter.alpha == 64


def test_target_modules_empty_dict_rejected():
    with pytest.raises(ValidationError, match="cannot be empty"):
        TrainingConfig.model_validate(_minimal(adapter={"target_modules": {}}))


def test_target_modules_negative_rank_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_minimal(adapter={"target_modules": -1}))


def test_target_modules_old_all_linear_string_rejected():
    """The 'all-linear' literal was removed from the schema in favor of int+dict."""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_minimal(adapter={"target_modules": "all-linear"}))


def test_old_r_field_rejected():
    """The standalone `r` field was merged into target_modules; rejected by extra='forbid'."""
    with pytest.raises(ValidationError, match="r"):
        TrainingConfig.model_validate(
            _minimal(adapter={"r": 32, "target_modules": 32})
        )


# --- adapter.invocation_tokens (gated activation) -----------------------------

def test_invocation_tokens_unset_keeps_gradient_checkpointing(caplog):
    cfg = TrainingConfig.model_validate(
        _minimal(batch={"gradient_checkpointing": True})
    )
    assert cfg.adapter.invocation_tokens is None
    assert cfg.batch.gradient_checkpointing is True


def test_invocation_tokens_set_disables_gradient_checkpointing(caplog):
    cfg = _minimal(
        adapter={"invocation_tokens": "<|x|>"},
        batch={"gradient_checkpointing": True},
    )
    with caplog.at_level(logging.WARNING):
        out = TrainingConfig.model_validate(cfg)
    assert out.batch.gradient_checkpointing is False
    assert any("PEFT #2826" in r.message for r in caplog.records)


# --- Removed fields are loudly rejected ---------------------------------------

@pytest.mark.parametrize("payload", [
    {"architecture": "lora"},
    {"alora": {"invocation_tokens": "<|x|>"}},
    {"shadow_residual": {"enabled": True}},
])
def test_removed_top_level_fields_rejected(payload):
    bad = _minimal(**payload)
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(bad)


def test_removed_model_dtype_rejected():
    bad = _minimal()
    bad["model"]["dtype"] = "bfloat16"
    with pytest.raises(ValidationError, match="dtype"):
        TrainingConfig.model_validate(bad)


def test_data_packing_no_longer_in_data_block():
    """`packing` moved from data: to trainer:"""
    bad = _minimal()
    bad["data"]["packing"] = True
    with pytest.raises(ValidationError, match="packing"):
        TrainingConfig.model_validate(bad)


def test_packing_lives_in_trainer_block():
    cfg = TrainingConfig.model_validate(_minimal(trainer={"packing": True}))
    assert cfg.trainer.packing is True


# --- validation block (formerly `eval:`) --------------------------------------

def test_validation_block_replaces_eval():
    cfg = TrainingConfig.model_validate(
        _minimal(validation={"strategy": "no", "early_stopping_patience": None,
                             "load_best_model_at_end": False})
    )
    assert cfg.validation.strategy == "no"
    assert cfg.validation.early_stopping_patience is None


def test_old_eval_key_rejected():
    bad = _minimal(eval={"strategy": "no"})
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(bad)


# --- generation block ---------------------------------------------------------

def test_generation_block_absent_by_default():
    """No generation block in YAML -> cfg.generation is None (skip post-training gen)."""
    cfg = TrainingConfig.model_validate(_minimal())
    assert cfg.generation is None


def test_generation_block_requires_input_path():
    """input_path is required when the block is present (it's the trigger field)."""
    with pytest.raises(ValidationError, match="input_path"):
        TrainingConfig.model_validate(_minimal(generation={"max_new_tokens": 32}))


def test_generation_block_with_defaults():
    cfg = TrainingConfig.model_validate(
        _minimal(generation={"input_path": "/tmp/val.jsonl"})
    )
    assert cfg.generation is not None
    assert cfg.generation.input_path == "/tmp/val.jsonl"
    assert cfg.generation.max_new_tokens == 256
    assert cfg.generation.do_sample is False
    assert cfg.generation.batch_size == 8
    assert cfg.generation.output_filename == "predictions.jsonl"


def test_generation_overrides():
    cfg = TrainingConfig.model_validate(
        _minimal(generation={"input_path": "/tmp/val.jsonl",
                             "max_new_tokens": 32, "do_sample": True,
                             "temperature": 0.7, "top_p": 0.9, "batch_size": 4})
    )
    assert cfg.generation.max_new_tokens == 32
    assert cfg.generation.do_sample is True
    assert cfg.generation.temperature == 0.7


# --- Scheduler quirks ---------------------------------------------------------

def test_warmup_steps_key_rejected():
    """The legacy `warmup_steps: 0.05` quirk must not survive the new schema."""
    bad = _minimal(scheduler={"warmup_steps": 0.05})
    with pytest.raises(ValidationError, match="warmup_steps"):
        TrainingConfig.model_validate(bad)


# --- Unknown keys forbidden ---------------------------------------------------

def test_unknown_top_level_key_rejected():
    bad = _minimal(typo_field=1)
    with pytest.raises(ValidationError, match="typo_field"):
        TrainingConfig.model_validate(bad)


def test_unknown_nested_key_rejected():
    bad = _minimal(optimizer={"learning_rate": 1e-4, "typo": 0.1})
    with pytest.raises(ValidationError, match="typo"):
        TrainingConfig.model_validate(bad)


# --- Env-var resolution -------------------------------------------------------

def test_env_var_resolves(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/data/x")
    out = _resolve_env_vars({
        "p": "${DATA_ROOT}/file.pq",
        "nested": ["${DATA_ROOT}/a", "plain"],
    })
    assert out == {"p": "/data/x/file.pq", "nested": ["/data/x/a", "plain"]}


def test_env_var_unset_raises(monkeypatch):
    monkeypatch.delenv("UNLIKELY_VAR_NAME_XYZ", raising=False)
    with pytest.raises(KeyError, match="UNLIKELY_VAR_NAME_XYZ"):
        _resolve_env_vars("${UNLIKELY_VAR_NAME_XYZ}/x")


def test_load_resolves_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("RUN_DIR", "/runs")
    cfg_yaml = tmp_path / "c.yaml"
    cfg_yaml.write_text(yaml.safe_dump({
        "model": {"base": "x"},
        "data": {"train_path": "${DATA_ROOT}/t.jsonl", "val_path": "${DATA_ROOT}/v.jsonl"},
        "save": {"output_dir": "${RUN_DIR}/ckpt"},
    }))
    cfg = load_training_config(cfg_yaml)
    assert cfg.data.train_path == "/data/t.jsonl"
    assert cfg.save.output_dir == "/runs/ckpt"


# --- Adapters into HF / PEFT objects ------------------------------------------

def test_to_training_arguments_smoke():
    pytest.importorskip("transformers")
    from shadow_residual.config.adapters import to_training_arguments

    cfg = TrainingConfig.model_validate(_minimal())
    args = to_training_arguments(cfg)
    assert args.output_dir == "/tmp/out"
    assert args.learning_rate == 2e-4
    assert args.warmup_ratio == 0.05
    assert args.bf16 is True
    assert args.gradient_checkpointing is True
    assert args.per_device_train_batch_size == 1


def test_to_peft_config_scalar_target_modules():
    pytest.importorskip("peft")
    from shadow_residual.config.adapters import to_peft_config

    cfg = TrainingConfig.model_validate(_minimal(adapter={"target_modules": 16}))
    pc = to_peft_config(cfg)
    assert pc.r == 16
    assert pc.lora_alpha == 32  # default 2 * rank
    assert pc.target_modules == "all-linear"
    # Scalar shape should not produce a rank_pattern
    assert getattr(pc, "rank_pattern", None) in (None, {})


def test_to_peft_config_per_module_target_modules():
    pytest.importorskip("peft")
    from shadow_residual.config.adapters import to_peft_config

    cfg = TrainingConfig.model_validate(
        _minimal(adapter={"target_modules": {"q_proj": 16, "gate_proj": 64}})
    )
    pc = to_peft_config(cfg)
    # PEFT default rank = min of dict; per-module overrides go to rank_pattern.
    assert pc.r == 16
    assert sorted(pc.target_modules) == ["gate_proj", "q_proj"]
    assert pc.rank_pattern == {"q_proj": 16, "gate_proj": 64}


def test_invocation_tokens_to_training_arguments_disables_gc():
    pytest.importorskip("transformers")
    from shadow_residual.config.adapters import to_training_arguments

    cfg = TrainingConfig.model_validate(_minimal(
        adapter={"invocation_tokens": "<|x|>"},
        batch={"gradient_checkpointing": True},
    ))
    args = to_training_arguments(cfg)
    assert args.gradient_checkpointing is False
