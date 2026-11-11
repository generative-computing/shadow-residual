# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI argument parsing and override application in train.main()."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shadow_residual.config import load_training_config
from shadow_residual.training.train import (
    _build_parser,
    apply_cli_overrides,
    apply_tool_formatting,
)


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("RUN_DIR", "/runs/job1")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({
        "model": {"base": "ibm-granite/granite-4.1-3b"},
        "data": {
            "train_path": "${DATA_ROOT}/train.jsonl",
            "val_path": "${DATA_ROOT}/val.jsonl",
        },
        "adapter": {"target_modules": 32},
        "save": {"output_dir": "${RUN_DIR}/checkpoints"},
        "runtime": {"num_train_epochs": 3, "seed": 42},
        "optimizer": {"learning_rate": 2.0e-4},
        "batch": {"per_device_train": 1, "gradient_accumulation": 16},
    }))
    return p


def test_parser_requires_config():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_accepts_minimal(config_path):
    parser = _build_parser()
    args = parser.parse_args(["--config", str(config_path)])
    assert args.config == str(config_path)
    # All overrides default to None / False
    assert args.lr is None
    assert args.lora_r is None
    assert args.quantize is False
    assert args.wandb is False


def test_overrides_no_op_when_unset(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path)])
    # Snapshot before
    before = (
        cfg.optimizer.learning_rate,
        cfg.runtime.num_train_epochs,
        cfg.adapter.target_modules,
        cfg.batch.per_device_train,
    )
    apply_cli_overrides(cfg, args)
    after = (
        cfg.optimizer.learning_rate,
        cfg.runtime.num_train_epochs,
        cfg.adapter.target_modules,
        cfg.batch.per_device_train,
    )
    assert before == after


def test_overrides_apply_when_set(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args([
        "--config", str(config_path),
        "--lr", "1e-5",
        "--epochs", "1",
        "--seed", "7",
        "--lora-r", "8",
        "--lora-alpha", "16",
        "--batch-size", "2",
        "--grad-accum", "4",
        "--max-length", "12288",
        "--output-dir", "/tmp/override",
        "--train-data", "/tmp/t.jsonl",
        "--val-data", "/tmp/v.jsonl",
    ])
    apply_cli_overrides(cfg, args)
    assert cfg.optimizer.learning_rate == 1e-5
    assert cfg.runtime.num_train_epochs == 1
    assert cfg.runtime.seed == 7
    assert cfg.adapter.target_modules == 8
    assert cfg.adapter.alpha == 16
    assert cfg.batch.per_device_train == 2
    assert cfg.batch.gradient_accumulation == 4
    assert cfg.data.max_length == 12288
    assert cfg.save.output_dir == "/tmp/override"
    assert cfg.data.train_path == "/tmp/t.jsonl"
    assert cfg.data.val_path == "/tmp/v.jsonl"


def test_wandb_override_sets_report_to(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path), "--wandb"])
    apply_cli_overrides(cfg, args)
    assert cfg.runtime.report_to == ["wandb"]


def test_thinking_flag_sets_enable_thinking(config_path):
    cfg = load_training_config(config_path)
    assert cfg.data.enable_thinking is False  # default
    args = _build_parser().parse_args(["--config", str(config_path), "--thinking"])
    apply_cli_overrides(cfg, args)
    assert cfg.data.enable_thinking is True


def test_thinking_flag_default_off(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path)])
    apply_cli_overrides(cfg, args)
    assert cfg.data.enable_thinking is False


def test_tool_formatting_default_off(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path)])
    apply_cli_overrides(cfg, args)
    apply_tool_formatting(cfg)
    assert cfg.data.tool_formatting is None
    assert cfg.data.assistant_marker is None
    assert cfg.adapter.last_context_token is None
    assert cfg.adapter.last_token is None


def test_tool_formatting_bare_flag_is_external(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path), "--tool-formatting"])
    apply_cli_overrides(cfg, args)
    apply_tool_formatting(cfg)
    assert cfg.data.tool_formatting == "external_tool_formatting"
    assert cfg.data.assistant_marker == "<tool_response>\n"
    assert cfg.adapter.last_context_token == "\n"
    assert cfg.adapter.last_token == "<|im_end|>"


def test_tool_formatting_autoregressive_markers(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(
        ["--config", str(config_path), "--tool-formatting", "autoregressive_tool_formatting"]
    )
    apply_cli_overrides(cfg, args)
    apply_tool_formatting(cfg)
    assert cfg.data.tool_formatting == "autoregressive_tool_formatting"
    assert cfg.data.assistant_marker == "</tool_call>"
    assert cfg.adapter.last_context_token == "</tool_call>"
    assert cfg.adapter.last_token == "<|im_end|>"


def test_tool_formatting_rejects_unknown_mode(config_path):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--config", str(config_path), "--tool-formatting", "bogus"])


@pytest.mark.parametrize("section, field, value", [
    ("data", "assistant_marker", "<|im_start|>assistant\n"),
    ("adapter", "last_context_token", "</think>"),
    ("adapter", "last_token", "<|im_end|>"),
])
def test_tool_formatting_rejects_explicit_markers(config_path, section, field, value):
    """Any explicitly set marker conflicts with the mode, even one equal to the
    implied value, and the error names the field and the implied value."""
    cfg = load_training_config(config_path)
    setattr(getattr(cfg, section), field, value)
    args = _build_parser().parse_args(["--config", str(config_path), "--tool-formatting"])
    apply_cli_overrides(cfg, args)
    with pytest.raises(ValueError, match=rf"{section}\.{field}=.*mode implies") as exc:
        apply_tool_formatting(cfg)
    assert "external_tool_formatting" in str(exc.value)
    assert "docs/tool_formatting.md" in str(exc.value)


@pytest.fixture
def no_model_build(monkeypatch):
    """Fail the test if train.main gets as far as building the model."""
    from shadow_residual.training import factory

    def _boom(**kw):
        raise AssertionError("train.main built the model before rejecting the config")

    monkeypatch.setattr(factory, "get_shadow_residual_peft_model", _boom)


def _tool_cfg(tmp_path: Path, base: str, **data) -> Path:
    p = tmp_path / "tool.yaml"
    p.write_text(yaml.safe_dump({
        "model": {"base": base},
        "data": {"train_path": str(tmp_path / "t.jsonl"), "val_path": str(tmp_path / "t.jsonl"),
                 "tool_formatting": "external_tool_formatting", **data},
        "adapter": {"target_modules": {"q_proj": 8, "cross_stream": 8}},
        "save": {"output_dir": str(tmp_path / "out")},
    }))
    return p


def test_main_rejects_non_g42_base_before_model_build(tmp_path, no_model_build):
    pytest.importorskip("trl")
    from shadow_residual.training.train import main
    from tests._tokenizers import load_tokenizer

    base = "ibm-granite/granite-4.1-3b"
    load_tokenizer(base)
    with pytest.raises(ValueError, match=r"requires a Granite 4\.2 base model.*granite-4\.1-3b"):
        main(["--config", str(_tool_cfg(tmp_path, base))])


def test_main_rejects_explicit_marker_before_loading_anything(tmp_path, no_model_build):
    from shadow_residual.training.train import main

    cfg = _tool_cfg(tmp_path, "does-not-exist/never-loaded", assistant_marker="<|im_start|>assistant\n")
    with pytest.raises(ValueError, match=r"data\.assistant_marker=.*mode implies '<tool_response>\\n'"):
        main(["--config", str(cfg)])


def test_explicit_markers_allowed_without_tool_formatting(config_path):
    cfg = load_training_config(config_path)
    cfg.data.assistant_marker = "<|im_start|>assistant\n"
    cfg.adapter.last_context_token = "\n"
    apply_tool_formatting(cfg)
    assert cfg.data.assistant_marker == "<|im_start|>assistant\n"
    assert cfg.adapter.last_context_token == "\n"


def test_tool_task_flag_sets_config(config_path):
    cfg = load_training_config(config_path)
    assert cfg.data.tool_task is None
    args = _build_parser().parse_args(
        ["--config", str(config_path), "--tool-formatting", "--tool-task", "guardian_core"]
    )
    apply_cli_overrides(cfg, args)
    apply_tool_formatting(cfg)
    assert cfg.data.tool_task == "guardian_core"


def test_tool_task_rejects_unknown_task(config_path):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--config", str(config_path), "--tool-task", "bogus"])


def test_tool_task_requires_tool_formatting(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path), "--tool-task", "answerability"])
    apply_cli_overrides(cfg, args)
    with pytest.raises(ValueError, match=r"tool_task='answerability' requires data\.tool_formatting"):
        apply_tool_formatting(cfg)


def test_val_frac_replaces_val_path(config_path):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path), "--val-frac", "0.1"])
    apply_cli_overrides(cfg, args)
    assert (cfg.data.val_path, cfg.data.val_frac) == (None, 0.1)


def test_val_frac_and_val_data_are_exclusive(config_path):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--config", str(config_path), "--val-frac", "0.1", "--val-data", "/v.jsonl"]
        )


@pytest.mark.parametrize("frac", ["0", "1", "1.5"])
def test_val_frac_out_of_range(config_path, frac):
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path), "--val-frac", frac])
    with pytest.raises(ValueError, match="--val-frac"):
        apply_cli_overrides(cfg, args)


def test_debug_collator_flag_forms(config_path):
    cfg = load_training_config(config_path)
    apply_cli_overrides(cfg, _build_parser().parse_args(["--config", str(config_path)]))
    assert (cfg.trainer.debug_collator, cfg.trainer.debug_probability) == (False, 0.001)

    cfg = load_training_config(config_path)
    apply_cli_overrides(cfg, _build_parser().parse_args(["--config", str(config_path), "--debug-collator"]))
    assert (cfg.trainer.debug_collator, cfg.trainer.debug_probability) == (True, 0.001)

    cfg = load_training_config(config_path)
    apply_cli_overrides(cfg, _build_parser().parse_args(
        ["--config", str(config_path), "--debug-collator", "0.25"]))
    assert (cfg.trainer.debug_collator, cfg.trainer.debug_probability) == (True, 0.25)

    cfg = load_training_config(config_path)
    with pytest.raises(ValueError, match="--debug-collator"):
        apply_cli_overrides(cfg, _build_parser().parse_args(
            ["--config", str(config_path), "--debug-collator", "2"]))


def test_quantize_flag_does_not_touch_config(config_path):
    """--quantize is consumed inside main(), not by apply_cli_overrides."""
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path), "--quantize"])
    optim_before = cfg.optimizer.name
    apply_cli_overrides(cfg, args)
    assert cfg.optimizer.name == optim_before  # unchanged here
    assert args.quantize is True
