# SPDX-License-Identifier: Apache-2.0
"""Tests for post-training generation: signature + FSDP-aware dispatch.

What's testable without a real FSDP launch:

- ``run_generation_under_fsdp`` exists and accepts the documented kwargs.
- ``_run_post_training_generation`` (the dispatch helper in train.py) picks
  ``run_generation`` for a non-FSDP-wrapped model and
  ``run_generation_under_fsdp`` for an FSDP-wrapped model — and passes the
  expected arguments.

End-to-end FSDP coverage stays in the Vela diagnostic
(``scripts/diagnose_fsdp_generate.py --run-attempt-10``) per the
experimental CLAUDE.md note: "end-to-end coverage is the Vela jobs, not
a local heavy-model test."
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from shadow_residual.config import load_training_config


# --- Signature: run_generation_under_fsdp exists with the documented shape ----


def test_run_generation_under_fsdp_signature():
    """The FSDP helper must exist and take the documented kwargs.

    This catches accidental signature drift — the train.py dispatch helper
    binds these by name, so a rename here without updating train.py would
    only show up at runtime under FSDP (rare path, hard to notice).
    """
    from shadow_residual.training.generate import (
        run_generation_under_fsdp,
    )

    sig = inspect.signature(run_generation_under_fsdp)
    params = sig.parameters
    expected = {
        "fsdp_model",
        "accelerator",
        "cfg",
        "peft_config",
        "tokenizer",
        "data_path",
        "out_path",
        "limit",
    }
    assert expected.issubset(set(params.keys())), (
        f"run_generation_under_fsdp must accept {expected}; "
        f"got {set(params.keys())}"
    )
    # All listed args must be keyword-only — the helper has too many args
    # for positional binding to read clearly at the call site, and the
    # test below relies on kwarg names.
    for name in expected:
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name!r} must be keyword-only on run_generation_under_fsdp"
        )


# --- Dispatch: train.py picks the right runner based on FSDP-wrap -------------


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch) -> Path:
    """A minimal TrainingConfig with the generation block populated."""
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
        "generation": {
            "input_path": "/data/eval.jsonl",
            "output_filename": "predictions.jsonl",
            "max_new_tokens": 16,
            "batch_size": 2,
        },
    }))
    return p


class _FakeFsdpModel:
    """Stand-in for a real ``FullyShardedDataParallel`` instance.

    The dispatch in train.py uses ``isinstance(model, FSDP)`` to decide
    which runner to call. We swap the FSDP class symbol used by train.py
    for this stub so ``isinstance`` returns True without importing real
    FSDP machinery (which pulls in a torch.distributed init we don't want
    in unit tests).
    """


class _FakePlainModel:
    """A plain model — should not be detected as FSDP-wrapped."""


def test_dispatch_calls_run_generation_under_fsdp_when_wrapped(
    config_path, monkeypatch, tmp_path,
):
    """When ``trainer.model`` is FSDP-wrapped, the dispatch helper must
    call ``run_generation_under_fsdp`` and pass through the FSDP-relevant
    kwargs (fsdp_model, accelerator, cfg, peft_config). It must NOT call
    the plain ``run_generation``.
    """
    from shadow_residual.training import generate as generate_mod
    from shadow_residual.training import train as train_mod

    cfg = load_training_config(config_path)

    # Fake trainer carrying a fake FSDP-wrapped model + accelerator.
    fake_accel = MagicMock(name="accelerator")
    fake_model = _FakeFsdpModel()
    fake_trainer = SimpleNamespace(
        model=fake_model,
        accelerator=fake_accel,
    )
    fake_peft_config = SimpleNamespace(name="peft_config")
    fake_tok = SimpleNamespace(name="tokenizer")

    # Patch the FSDP class symbol on train module to our stub so isinstance
    # returns True without spinning up real distributed FSDP.
    monkeypatch.setattr(train_mod, "_FSDP_CLASS", _FakeFsdpModel, raising=False)

    plain_runner = MagicMock(return_value=4)
    fsdp_runner = MagicMock(return_value=4)
    monkeypatch.setattr(generate_mod, "run_generation", plain_runner)
    monkeypatch.setattr(generate_mod, "run_generation_under_fsdp", fsdp_runner)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_mod._run_post_training_generation(
        trainer=fake_trainer,
        cfg=cfg,
        peft_config=fake_peft_config,
        tokenizer=fake_tok,
        output_dir=out_dir,
    )

    plain_runner.assert_not_called()
    fsdp_runner.assert_called_once()
    kwargs = fsdp_runner.call_args.kwargs
    assert kwargs["fsdp_model"] is fake_model
    assert kwargs["accelerator"] is fake_accel
    assert kwargs["cfg"] is cfg
    assert kwargs["peft_config"] is fake_peft_config
    assert kwargs["tokenizer"] is fake_tok
    assert Path(kwargs["data_path"]) == Path("/data/eval.jsonl")
    assert Path(kwargs["out_path"]) == out_dir / "predictions.jsonl"
    # The FSDP branch must hand the helper a release_model_refs callback:
    # trainer.model is a second live ref to the FSDP shard, and without
    # nulling it the helper's `del fsdp_model` can't free the ~16GB shard
    # (OOMs the rank-0 rebuild at 30B). The helper invokes it after the
    # gather; here we invoke it directly to confirm it nulls trainer.model.
    release_cb = kwargs["release_model_refs"]
    assert callable(release_cb)
    assert fake_trainer.model is fake_model  # not nulled until the cb fires
    release_cb()
    assert fake_trainer.model is None


def test_dispatch_calls_run_generation_when_not_wrapped(
    config_path, monkeypatch, tmp_path,
):
    """When ``trainer.model`` is NOT FSDP-wrapped (single-GPU / DDP), the
    dispatch helper must call the plain ``run_generation`` on rank 0 and
    NOT call ``run_generation_under_fsdp``.
    """
    from shadow_residual.training import generate as generate_mod
    from shadow_residual.training import train as train_mod

    cfg = load_training_config(config_path)

    fake_model = _FakePlainModel()
    fake_trainer = SimpleNamespace(
        model=fake_model,
        accelerator=MagicMock(name="accelerator"),
    )

    # Force LOCAL_RANK=0 so the rank-0-only branch executes.
    monkeypatch.setenv("LOCAL_RANK", "0")
    # Patch the FSDP-class symbol used by the dispatch to a class our
    # plain model is NOT an instance of.
    monkeypatch.setattr(train_mod, "_FSDP_CLASS", _FakeFsdpModel, raising=False)

    plain_runner = MagicMock(return_value=4)
    fsdp_runner = MagicMock(return_value=4)
    monkeypatch.setattr(generate_mod, "run_generation", plain_runner)
    monkeypatch.setattr(generate_mod, "run_generation_under_fsdp", fsdp_runner)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_mod._run_post_training_generation(
        trainer=fake_trainer,
        cfg=cfg,
        peft_config=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        output_dir=out_dir,
    )

    fsdp_runner.assert_not_called()
    plain_runner.assert_called_once()
    kwargs = plain_runner.call_args.kwargs
    assert kwargs["model"] is fake_model
    assert kwargs["gen_config"] is cfg.generation
    assert Path(kwargs["data_path"]) == Path("/data/eval.jsonl")
    assert Path(kwargs["out_path"]) == out_dir / "predictions.jsonl"


def test_dispatch_skips_non_rank0_when_not_fsdp(
    config_path, monkeypatch, tmp_path,
):
    """For non-FSDP runs, only rank 0 generates (existing behavior).
    Other ranks should call neither runner.
    """
    from shadow_residual.training import generate as generate_mod
    from shadow_residual.training import train as train_mod

    cfg = load_training_config(config_path)

    fake_trainer = SimpleNamespace(
        model=_FakePlainModel(),
        accelerator=MagicMock(),
    )

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(train_mod, "_FSDP_CLASS", _FakeFsdpModel, raising=False)

    plain_runner = MagicMock(return_value=4)
    fsdp_runner = MagicMock(return_value=4)
    monkeypatch.setattr(generate_mod, "run_generation", plain_runner)
    monkeypatch.setattr(generate_mod, "run_generation_under_fsdp", fsdp_runner)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_mod._run_post_training_generation(
        trainer=fake_trainer,
        cfg=cfg,
        peft_config=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        output_dir=out_dir,
    )

    plain_runner.assert_not_called()
    fsdp_runner.assert_not_called()


def test_dispatch_runs_fsdp_path_on_all_ranks(
    config_path, monkeypatch, tmp_path,
):
    """For FSDP runs, EVERY rank must enter the gather collective. The
    dispatch helper must call ``run_generation_under_fsdp`` regardless of
    LOCAL_RANK — the helper itself decides what work happens on which rank.
    """
    from shadow_residual.training import generate as generate_mod
    from shadow_residual.training import train as train_mod

    cfg = load_training_config(config_path)

    fake_trainer = SimpleNamespace(
        model=_FakeFsdpModel(),
        accelerator=MagicMock(),
    )

    monkeypatch.setenv("LOCAL_RANK", "2")  # not rank 0
    monkeypatch.setattr(train_mod, "_FSDP_CLASS", _FakeFsdpModel, raising=False)

    plain_runner = MagicMock(return_value=4)
    fsdp_runner = MagicMock(return_value=0)
    monkeypatch.setattr(generate_mod, "run_generation", plain_runner)
    monkeypatch.setattr(generate_mod, "run_generation_under_fsdp", fsdp_runner)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_mod._run_post_training_generation(
        trainer=fake_trainer,
        cfg=cfg,
        peft_config=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        output_dir=out_dir,
    )

    plain_runner.assert_not_called()
    fsdp_runner.assert_called_once()


def test_dispatch_no_op_when_generation_block_absent(
    tmp_path: Path, monkeypatch,
):
    """If ``cfg.generation`` is None (post-train generate disabled), the
    dispatch helper must no-op (no runner called, no exception).
    """
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
        # NOTE: no generation block.
    }))

    from shadow_residual.training import generate as generate_mod
    from shadow_residual.training import train as train_mod

    cfg = load_training_config(p)
    assert cfg.generation is None

    plain_runner = MagicMock()
    fsdp_runner = MagicMock()
    monkeypatch.setattr(generate_mod, "run_generation", plain_runner)
    monkeypatch.setattr(generate_mod, "run_generation_under_fsdp", fsdp_runner)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_mod._run_post_training_generation(
        trainer=SimpleNamespace(model=_FakePlainModel(), accelerator=MagicMock()),
        cfg=cfg,
        peft_config=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        output_dir=out_dir,
    )

    plain_runner.assert_not_called()
    fsdp_runner.assert_not_called()


# --- Adaptive batch-size halving on generation OOM ---------------------------
# A 30B model with long-document prompts can OOM single-GPU generation at a
# given batch size. run_generation halves the batch and retries the same slice
# on torch.cuda.OutOfMemoryError (down to 1), instead of failing or assuming a
# conservative fixed batch size. This test drives that path with a fake model.


def test_run_generation_halves_batch_on_oom(tmp_path, monkeypatch):
    import json
    import torch
    from shadow_residual.training import generate as gen_mod

    # 5 rows; start batch_size=4. Fake _generate_batch OOMs for any batch > 2,
    # so the loop must halve 4 -> 2 and then complete all rows.
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(5)]
    data_path = tmp_path / "eval.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out_path = tmp_path / "preds.jsonl"

    monkeypatch.setattr(gen_mod, "_load_jsonl_rows", lambda p: list(rows))

    calls = {"oom": 0}

    def fake_generate_batch(model, tokenizer, prompts, cfg_gen, debug_probability=0.0):
        if len(prompts) > 2:
            calls["oom"] += 1
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return [f"gen-{p}" for p in prompts]

    monkeypatch.setattr(gen_mod, "_generate_batch", fake_generate_batch)

    class _Tok:
        padding_side = "right"
        pad_token = "<pad>"
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kw):
            return messages[0]["content"]

    gen_config = SimpleNamespace(batch_size=4, debug_probability=0.0)
    model = SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)

    n = gen_mod.run_generation(
        model=model, tokenizer=_Tok(), gen_config=gen_config,
        data_path=data_path, out_path=out_path,
    )

    assert n == 5, f"expected all 5 rows generated, got {n}"
    assert calls["oom"] >= 1, "the OOM-retry path was not exercised"
    written = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(written) == 5
    assert all("generated_content" in r for r in written)
