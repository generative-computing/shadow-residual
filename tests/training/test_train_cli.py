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


def test_quantize_flag_does_not_touch_config(config_path):
    """--quantize is consumed inside main(), not by apply_cli_overrides."""
    cfg = load_training_config(config_path)
    args = _build_parser().parse_args(["--config", str(config_path), "--quantize"])
    optim_before = cfg.optimizer.name
    apply_cli_overrides(cfg, args)
    assert cfg.optimizer.name == optim_before  # unchanged here
    assert args.quantize is True
