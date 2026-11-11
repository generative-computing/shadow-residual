# SPDX-License-Identifier: Apache-2.0
"""${VAR} placeholders from CLI flags (--base-model / --data-root / --eval-data /
--run-dir) instead of environment variables, and CLI paths that make a
placeholder unnecessary."""

from __future__ import annotations

from pathlib import Path

import pytest

from shadow_residual.config import load_training_config
from shadow_residual.training.train import (
    _build_parser,
    apply_cli_overrides,
    apply_tool_formatting,
    config_overrides,
    config_variables,
)

REF = Path(__file__).resolve().parents[2] / "src/shadow_residual/config/sr-tool-g42.yaml"
VARS = ("BASE_MODEL", "DATA_ROOT", "EVAL_DATA", "RUN_DIR")


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)


def _train_cfg(*argv, config=REF):
    args = _build_parser().parse_args(["--config", str(config), *argv])
    cfg = load_training_config(args.config, variables=config_variables(args), overrides=config_overrides(args))
    apply_cli_overrides(cfg, args)
    apply_tool_formatting(cfg)
    return cfg


def test_flags_replace_every_env_variable():
    cfg = _train_cfg("--base-model", "ibm-granite/granite-4.2-3b", "--data-root", "/d",
                     "--eval-data", "/e/eval.jsonl", "--run-dir", "/r")
    assert cfg.model.base == "ibm-granite/granite-4.2-3b"
    assert (cfg.data.train_path, cfg.data.val_path) == ("/d/train.jsonl", "/d/val.jsonl")
    assert cfg.generation.input_path == "/e/eval.jsonl"
    assert cfg.save.output_dir == "/r/checkpoints"


def test_flag_wins_over_env(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/env")
    monkeypatch.setenv("BASE_MODEL", "env-model")
    monkeypatch.setenv("EVAL_DATA", "/env/eval.jsonl")
    monkeypatch.setenv("RUN_DIR", "/env/run")
    cfg = _train_cfg("--data-root", "/cli", "--base-model", "cli-model")
    assert cfg.data.train_path == "/cli/train.jsonl" and cfg.model.base == "cli-model"
    assert cfg.save.output_dir == "/env/run/checkpoints"


def test_explicit_paths_need_no_placeholder_values():
    cfg = _train_cfg("--base-model", "m", "--train-data", "/t.jsonl", "--val-frac", "0.1",
                     "--eval-data", "/e.jsonl", "--output-dir", "/out")
    assert (cfg.data.train_path, cfg.data.val_path, cfg.data.val_frac) == ("/t.jsonl", None, 0.1)
    assert cfg.save.output_dir == "/out"


@pytest.mark.parametrize("missing, flag", [
    ("DATA_ROOT", "--data-root"), ("RUN_DIR", "--run-dir"),
    ("EVAL_DATA", "--eval-data"), ("BASE_MODEL", "--base-model"),
])
def test_missing_value_names_its_flag(missing, flag):
    given = {"BASE_MODEL": ["--base-model", "m"], "DATA_ROOT": ["--data-root", "/d"],
             "EVAL_DATA": ["--eval-data", "/e"], "RUN_DIR": ["--run-dir", "/r"]}
    argv = [a for k, v in given.items() if k != missing for a in v]
    with pytest.raises(KeyError, match=rf"\${{{missing}}} is not set \(pass {flag} or set {missing}\)"):
        _train_cfg(*argv)


def test_base_model_flag_overrides_a_hardcoded_base(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model: {base: hardcoded}\ndata: {train_path: /t, val_path: /v}\n"
                 "adapter: {target_modules: 8}\nsave: {output_dir: /o}\n")
    assert _train_cfg("--base-model", "cli", config=p).model.base == "cli"


def test_eval_data_ignored_without_generation_block(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model: {base: m}\ndata: {train_path: /t, val_path: /v}\n"
                 "adapter: {target_modules: 8}\nsave: {output_dir: /o}\n")
    assert _train_cfg("--eval-data", "/e", config=p).generation is None


def test_val_frac_range_checked_before_loading():
    with pytest.raises(ValueError, match=r"--val-frac must be in \(0, 1\)"):
        _train_cfg("--base-model", "m", "--data-root", "/d", "--run-dir", "/r", "--eval-data", "/e",
                   "--val-frac", "1.5")


def test_generate_cli_takes_the_same_flags():
    from shadow_residual.training.generate import _build_parser as gen_parser
    args = gen_parser().parse_args(["--config", "c.yaml", "--checkpoint", "ck", "--base-model", "m",
                                    "--data-root", "/d", "--run-dir", "/r", "--eval-data", "/e"])
    assert (args.base_model, args.data_root, args.run_dir, args.data) == ("m", "/d", "/r", "/e")
    assert config_variables(args) == {"BASE_MODEL": "m", "DATA_ROOT": "/d", "RUN_DIR": "/r"}


def test_train_main_runs_on_flags_alone(monkeypatch, tmp_path):
    """No ${VAR} env variables set: main() gets past config, CLI and the 4.2
    tokenizer gate on flags alone, up to the model build."""
    pytest.importorskip("trl")
    from tests._tokenizers import load_tokenizer
    from shadow_residual.training import factory
    from shadow_residual.training.train import main

    load_tokenizer("ibm-granite/granite-4.2-3b")

    class Reached(Exception):
        pass

    def _stop(**kw):
        raise Reached(kw["base_model_name_or_path"])

    monkeypatch.setattr(factory, "get_shadow_residual_peft_model", _stop)
    with pytest.raises(Reached, match="ibm-granite/granite-4.2-3b"):
        main(["--config", str(REF), "--base-model", "ibm-granite/granite-4.2-3b",
              "--data-root", str(tmp_path), "--eval-data", str(tmp_path / "e.jsonl"),
              "--run-dir", str(tmp_path / "run")])
