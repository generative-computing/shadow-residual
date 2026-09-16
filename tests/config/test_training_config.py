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
        # All five arms of the W-cross ablation. They differ only in the
        # cross-stream key(s) under adapter.target_modules (and, for the two-tap
        # arm, a per-module alpha dict), so validating every one of them is what
        # pins that the schema accepts each tap name and each alpha form.
        "sr-qo-mlp-r32-c32-sharedkv.yaml",
        "sr-qo-mlp-r32-cattn32-sharedkv.yaml",
        "sr-qo-mlp-r32-cboth16-sharedkv.yaml",
        "sr-qo-mlp-r32-cpre2mlp32-sharedkv.yaml",
        "sr-qo-mlp-r32-cmlp2pre32-sharedkv.yaml",
    ],
)
def test_examples_validate(monkeypatch, name):
    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("RUN_DIR", "/runs/job1")
    monkeypatch.setenv("EVAL_DATA", "/data/eval.jsonl")
    cfg = load_training_config(EXAMPLES / name)
    assert cfg.model.base == "ibm-granite/granite-4.1-3b"
    assert cfg.save.output_dir == "/runs/job1/checkpoints"


# The five Granite 4.2 ablation arms → the cross-stream tap(s) each one trains.
# These are the configs the answerability ablation actually runs, so every arm must
# validate AND resolve to the tap set its filename advertises — a copy-paste slip in
# the cross-stream key would otherwise only surface as a mislabelled GPU run.
_G42_ARMS = {
    "sr-qo-mlp-r32-c32-sharedkv-g42.yaml": ("cross_stream",),
    "sr-qo-mlp-r32-cattn32-sharedkv-g42.yaml": ("cross_stream_post_attn",),
    "sr-qo-mlp-r32-cpre2mlp32-sharedkv-g42.yaml": (
        "cross_stream_pre_attn_to_post_mlp",
    ),
    "sr-qo-mlp-r32-cmlp2pre32-sharedkv-g42.yaml": (
        "cross_stream_post_mlp_to_pre_attn",
    ),
    "sr-qo-mlp-r32-cboth32-sharedkv-g42.yaml": (
        "cross_stream",
        "cross_stream_post_attn",
    ),
}


@pytest.mark.parametrize("name", list(_G42_ARMS), ids=list(_G42_ARMS))
def test_g42_config_validates(monkeypatch, name):
    """The Granite 4.2 arms: ${BASE_MODEL} interpolation + ChatML markers.

    Also pins the rank contract for the ablation: every targeted module is r=32
    with a scalar alpha of 64, so ``alpha/r == 2.0`` holds across all five arms and
    effective LoRA scale is never a confound.
    """
    from shadow_residual.shadow_residual.cross_stream import (
        cross_stream_taps_from_target_modules,
    )

    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("RUN_DIR", "/runs/job1")
    monkeypatch.setenv("EVAL_DATA", "/data/eval.jsonl")
    monkeypatch.setenv("BASE_MODEL", "/models/granite-4.2-3b")
    cfg = load_training_config(EXAMPLES / name)
    assert cfg.model.base == "/models/granite-4.2-3b"      # ${BASE_MODEL} resolved
    # ChatML thinking-OFF: generation begins after </think>, so that is the last
    # context token; the collator masks through "<|im_start|>assistant\n<think></think>".
    assert cfg.adapter.last_context_token == "</think>"
    assert cfg.data.assistant_marker == "<|im_start|>assistant\n<think></think>"
    assert cfg.adapter.last_token == "<|im_end|>"

    targets = cfg.adapter.target_modules
    assert set(targets.values()) == {32}, f"{name}: every module must be r=32"
    assert cfg.adapter.alpha == 64, f"{name}: alpha must be 64 (alpha/r == 2.0)"
    assert cross_stream_taps_from_target_modules(list(targets)) == _G42_ARMS[name]


def test_shared_base_kv_key_is_rejected():
    """SR is shared-KV-only — there is no `shared_base_kv` toggle. The key was
    removed from ModelConfig; ModelConfig forbids extras, so a YAML that still
    sets it must fail loudly rather than silently no-op."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            _minimal(model={"base": "ibm-granite/granite-4.1-3b", "shared_base_kv": True})
        )


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


# --- adapter.cross_stream_type ------------------------------------------------

def test_cross_stream_type_defaults_lora():
    cfg = TrainingConfig.model_validate(_minimal(adapter={"target_modules": 32}))
    assert cfg.adapter.cross_stream_type == "lora"


def test_cross_stream_type_linear_accepts_dict_with_cross_stream():
    cfg = TrainingConfig.model_validate(_minimal(adapter={
        "cross_stream_type": "linear",
        "target_modules": {"q_proj": 32, "cross_stream": 8},
    }))
    assert cfg.adapter.cross_stream_type == "linear"
    assert cfg.adapter.target_modules["cross_stream"] == 8
    # alpha excludes the linear d (=8): keyed only off the LoRA ranks (max=32).
    assert cfg.adapter.alpha == 64


def test_cross_stream_type_linear_requires_dict_form():
    with pytest.raises(ValidationError, match="requires the dict form"):
        TrainingConfig.model_validate(_minimal(adapter={
            "cross_stream_type": "linear",
            "target_modules": 32,
        }))


def test_cross_stream_type_linear_requires_cross_stream_entry():
    with pytest.raises(ValidationError, match="cross_stream"):
        TrainingConfig.model_validate(_minimal(adapter={
            "cross_stream_type": "linear",
            "target_modules": {"q_proj": 32},
        }))


def test_cross_stream_type_unknown_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_minimal(adapter={
            "cross_stream_type": "butterfly",
            "target_modules": {"q_proj": 32, "cross_stream": 8},
        }))


def test_cross_stream_type_accepts_extra_tap():
    """The TYPE axis is orthogonal to the WIRING axis: a non-lora type at the
    default tap alongside an ADDITIONAL cross_stream* tap is legal, and each tap
    gets its own type (taps omitted from a dict default to 'lora')."""
    cfg = TrainingConfig.model_validate(_minimal(adapter={
        "cross_stream_type": {"cross_stream": "linear"},
        "target_modules": {"q_proj": 32, "cross_stream": 8, "cross_stream_post_attn": 8},
    }))
    assert cfg.adapter.non_lora_cross_stream_taps() == ["cross_stream"]
    assert cfg.adapter.cross_stream_type_for("cross_stream_post_attn") == "lora"


def test_cross_stream_type_accepts_non_default_tap():
    """A non-lora type at a NON-default tap, with no default cross_stream at all —
    every type is supported at every wiring."""
    cfg = TrainingConfig.model_validate(_minimal(adapter={
        "cross_stream_type": "monarch",
        "target_modules": {"q_proj": 32, "cross_stream_post_attn": 40},
    }))
    assert cfg.adapter.non_lora_cross_stream_taps() == ["cross_stream_post_attn"]
    assert cfg.adapter.cross_stream_tap_types_map() == {
        "cross_stream_post_attn": {"type": "monarch", "num": 40},
    }


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


# --- adapter.last_context_token / last_token ----------------------------------

def test_token_markers_default_none():
    cfg = TrainingConfig.model_validate(_minimal())
    assert cfg.adapter.last_context_token is None
    assert cfg.adapter.last_token is None


def test_token_markers_set():
    cfg = TrainingConfig.model_validate(
        _minimal(adapter={"last_context_token": "<|end_of_role|>", "last_token": "<|end_of_text|>"})
    )
    assert cfg.adapter.last_context_token == "<|end_of_role|>"
    assert cfg.adapter.last_token == "<|end_of_text|>"


def test_token_markers_keep_gradient_checkpointing():
    """The markers don't gate the adapter and don't touch gradient checkpointing
    — it's honored as set (no eval/GC force-off; aLoRA was removed)."""
    cfg = _minimal(
        adapter={"last_context_token": "<|x|>"},
        batch={"gradient_checkpointing": True},
    )
    out = TrainingConfig.model_validate(cfg)
    assert out.batch.gradient_checkpointing is True


def test_invocation_tokens_field_rejected():
    """The old aLoRA field is gone; AdapterConfig forbids extras."""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_minimal(adapter={"invocation_tokens": "<|x|>"}))


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
    # Scalar maps to the fixed SR-safe target set — NOT "all-linear", which would
    # wrap the forbidden K/V projections (rejected in factory._reject_kv_lora).
    assert sorted(pc.target_modules) == [
        "cross_stream",
        "down_proj",
        "gate_proj",
        "o_proj",
        "q_proj",
        "up_proj",
    ]
    assert "k_proj" not in pc.target_modules
    assert "v_proj" not in pc.target_modules
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


def test_scalar_alpha_produces_no_alpha_pattern():
    pytest.importorskip("peft")
    from shadow_residual.config.adapters import to_peft_config

    cfg = TrainingConfig.model_validate(
        _minimal(adapter={"target_modules": {"q_proj": 16, "gate_proj": 64}, "alpha": 32})
    )
    pc = to_peft_config(cfg)
    assert pc.lora_alpha == 32
    assert getattr(pc, "alpha_pattern", None) in (None, {})


def test_per_module_alpha_emits_alpha_pattern():
    """A dict `alpha` projects to PEFT's `alpha_pattern`, which is what makes a
    parameter-aligned ablation possible: LoRA scales by alpha / r, so halving a
    module's rank without halving its alpha would change its effective scale too."""
    pytest.importorskip("peft")
    from shadow_residual.config.adapters import to_peft_config

    cfg = TrainingConfig.model_validate(_minimal(adapter={
        "target_modules": {"cross_stream": 16, "cross_stream_post_attn": 16, "q_proj": 32},
        "alpha": {"cross_stream": 32, "cross_stream_post_attn": 32, "q_proj": 64},
    }))
    pc = to_peft_config(cfg)
    assert pc.alpha_pattern == {
        "cross_stream": 32, "cross_stream_post_attn": 32, "q_proj": 64,
    }
    # PEFT resolves ONE target_name_key across chain(rank_pattern, alpha_pattern)
    # and indexes BOTH with it, so the key sets must be identical.
    assert set(pc.alpha_pattern) == set(pc.rank_pattern)
    # Scalar fallbacks are the minima of each dict.
    assert pc.r == 16
    assert pc.lora_alpha == 32
    # alpha / r == 2.0 for every module — the aligned-ablation property.
    for name, r in pc.rank_pattern.items():
        assert pc.alpha_pattern[name] / r == pytest.approx(2.0)


def test_dict_alpha_requires_dict_target_modules():
    with pytest.raises(ValidationError, match="dict form of"):
        TrainingConfig.model_validate(_minimal(adapter={
            "target_modules": 32,
            "alpha": {"q_proj": 64},
        }))


def test_dict_alpha_key_set_must_match_target_modules():
    """An omitted module silently falling back to the scalar alpha is exactly the
    uncontrolled scale change an aligned ablation is trying to remove."""
    with pytest.raises(ValidationError, match="must match"):
        TrainingConfig.model_validate(_minimal(adapter={
            "target_modules": {"q_proj": 32, "cross_stream": 16},
            "alpha": {"q_proj": 64},
        }))
    with pytest.raises(ValidationError, match="must match"):
        TrainingConfig.model_validate(_minimal(adapter={
            "target_modules": {"q_proj": 32},
            "alpha": {"q_proj": 64, "o_proj": 64},
        }))


def test_to_peft_config_lora_cross_stream_is_lora_target():
    """Default (lora) cross-stream: it's a LoRA target, no modules_to_save."""
    pytest.importorskip("peft")
    from shadow_residual.config.adapters import to_peft_config

    cfg = TrainingConfig.model_validate(_minimal(adapter={
        "target_modules": {"q_proj": 32, "cross_stream": 32},
    }))
    pc = to_peft_config(cfg)
    assert "cross_stream" in pc.target_modules
    assert "cross_stream" in (pc.rank_pattern or {})
    assert getattr(pc, "modules_to_save", None) in (None, [])


def test_to_peft_config_linear_cross_stream_uses_modules_to_save():
    """linear cross-stream: NOT a LoRA target — routed to modules_to_save, and
    absent from target_modules / rank_pattern (its d is not a LoRA rank)."""
    pytest.importorskip("peft")
    from shadow_residual.config.adapters import to_peft_config

    cfg = TrainingConfig.model_validate(_minimal(adapter={
        "cross_stream_type": "linear",
        "target_modules": {"q_proj": 32, "o_proj": 32, "cross_stream": 8},
    }))
    pc = to_peft_config(cfg)
    assert pc.modules_to_save == ["cross_stream"]
    assert "cross_stream" not in pc.target_modules
    assert sorted(pc.target_modules) == ["o_proj", "q_proj"]
    assert "cross_stream" not in (pc.rank_pattern or {})
    # r/rank_pattern keyed only off genuine LoRA targets (both 32 here).
    assert pc.r == 32


def test_to_training_arguments_keeps_gc():
    """to_training_arguments passes gradient_checkpointing through as configured.
    Note: train.py separately sets the Trainer's flag False and manages
    checkpointing on the model itself — this test only pins that the config layer
    doesn't mutate it."""
    pytest.importorskip("transformers")
    from shadow_residual.config.adapters import to_training_arguments

    cfg = TrainingConfig.model_validate(_minimal(
        adapter={"last_context_token": "<|x|>"},
        batch={"gradient_checkpointing": True},
    ))
    args = to_training_arguments(cfg)
    assert args.gradient_checkpointing is True
