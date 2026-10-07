# SPDX-License-Identifier: Apache-2.0
"""Project a TrainingConfig into HF/PEFT-shaped objects.

Keep this module thin: each function takes a validated TrainingConfig and
returns a single object the trainer needs. No I/O, no model loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .training_config import TrainingConfig

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import TrainerCallback, TrainingArguments


# SR-safe target set for a uniform scalar rank. Excludes k_proj/v_proj (shared
# base-only K/V — an adapter-side K/V delta has nowhere to land) and includes
# cross_stream (engages the dual-stream SR forward). See
# factory._reject_kv_lora, which forbids the "all-linear" shortcut for the same
# reason.
_SR_SCALAR_TARGETS = [
    "q_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "cross_stream",
]


def to_training_arguments(cfg: TrainingConfig) -> "TrainingArguments":
    """Build transformers.TrainingArguments from a unified config.

    Always uses bfloat16 mixed precision (no `dtype` knob in the schema).
    """
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = dict(
        output_dir=cfg.save.output_dir,
        num_train_epochs=cfg.runtime.num_train_epochs,
        seed=cfg.runtime.seed,
        # Optimizer
        optim=cfg.optimizer.name,
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        max_grad_norm=cfg.optimizer.max_grad_norm,
        # Scheduler
        lr_scheduler_type=cfg.scheduler.type,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        # Batch
        per_device_train_batch_size=cfg.batch.per_device_train,
        per_device_eval_batch_size=cfg.batch.per_device_eval,
        gradient_accumulation_steps=cfg.batch.gradient_accumulation,
        gradient_checkpointing=cfg.batch.gradient_checkpointing,
        gradient_checkpointing_kwargs=cfg.batch.gradient_checkpointing_kwargs,
        # Validation (formerly `eval:` in the schema)
        eval_strategy=cfg.validation.strategy,
        load_best_model_at_end=cfg.validation.load_best_model_at_end,
        metric_for_best_model=cfg.validation.metric_for_best_model,
        greater_is_better=cfg.validation.greater_is_better,
        # Save
        save_strategy=cfg.save.strategy,
        save_total_limit=cfg.save.total_limit,
        # Runtime
        logging_steps=cfg.runtime.logging_steps,
        logging_nan_inf_filter=not cfg.validation.stop_on_nan,
        dataloader_num_workers=cfg.runtime.dataloader_num_workers,
        report_to=cfg.runtime.report_to,
        # Mixed precision: always bf16.
        bf16=True,
    )

    # FSDP — bool or dict
    if isinstance(cfg.runtime.fsdp, dict):
        kwargs["fsdp"] = cfg.runtime.fsdp.get("sharding_strategy", "full_shard")
    elif cfg.runtime.fsdp:
        kwargs["fsdp"] = "full_shard"

    return TrainingArguments(**kwargs)


def to_peft_config(cfg: TrainingConfig) -> "LoraConfig":
    """Build peft.LoraConfig from the adapter block.

    `cfg.adapter.target_modules` is either a scalar rank (uniform) or a
    dict[name, rank] (per-module). Both shapes need to be projected into
    PEFT's three-field surface: `r`, `target_modules`, and optional
    `rank_pattern`.
    """
    from peft import LoraConfig

    a = cfg.adapter

    if isinstance(a.target_modules, dict):
        # Per-module: PEFT wants a scalar default in `r` plus a list of
        # module names in `target_modules` and per-name overrides in
        # `rank_pattern`. Use the smallest rank as the default and list every
        # explicit module (including ones equal to the default — harmless,
        # avoids surprises if PEFT changes how it merges).
        target_names = list(a.target_modules.keys())
        r_scalar = min(a.target_modules.values())
        rank_pattern: dict[str, int] | None = dict(a.target_modules)
    else:
        # Uniform scalar: apply the same rank to the fixed SR-safe module set.
        # NOT "all-linear" — that would wrap the forbidden K/V projections
        # (rejected in factory._reject_kv_lora). Listing modules explicitly keeps
        # K/V out and engages the dual-stream forward via "cross_stream".
        target_names = _SR_SCALAR_TARGETS
        r_scalar = a.target_modules
        rank_pattern = None

    # Per-module alpha projects to PEFT's `alpha_pattern` (same shape as
    # rank_pattern). The schema guarantees a dict alpha's keys equal
    # target_modules' keys exactly, which matters: PEFT resolves ONE
    # `target_name_key` from chain(rank_pattern, alpha_pattern) and indexes BOTH
    # with it, so key sets that differ would silently mis-resolve. `lora_alpha` is
    # the scalar fallback — set to the min so an (impossible, per the validator)
    # unlisted module can't inflate scale; every listed module is in alpha_pattern.
    if isinstance(a.alpha, dict):
        alpha_scalar = min(a.alpha.values())
        alpha_pattern: dict[str, int] | None = dict(a.alpha)
    else:
        alpha_scalar = a.alpha
        alpha_pattern = None

    kwargs: dict[str, Any] = dict(
        r=r_scalar,
        lora_alpha=alpha_scalar,
        lora_dropout=a.dropout,
        bias=a.bias,
        target_modules=target_names,
        task_type=a.task_type,
    )
    if rank_pattern is not None:
        kwargs["rank_pattern"] = rank_pattern
    if alpha_pattern is not None:
        kwargs["alpha_pattern"] = alpha_pattern

    return LoraConfig(**kwargs)


def build_callbacks(cfg: TrainingConfig) -> list["TrainerCallback"]:
    """Construct the optional callbacks (early stopping, NaN guard)."""
    from transformers import TrainerCallback

    callbacks: list[TrainerCallback] = []

    if (
        cfg.validation.early_stopping_patience is not None
        and cfg.validation.strategy != "no"
    ):
        from transformers import EarlyStoppingCallback
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.validation.early_stopping_patience
            )
        )

    if cfg.validation.stop_on_nan:
        callbacks.append(_make_stop_on_nan_callback())

    return callbacks


def _make_stop_on_nan_callback():
    """Build the StopOnNan callback. Defined as a factory so the
    `transformers.TrainerCallback` import stays lazy (the schema module is
    importable without transformers installed)."""
    from transformers import TrainerCallback

    class _StopOnNanCallback(TrainerCallback):
        """Stop training when train loss is NaN or Inf.

        Trainer's default behavior is to log and continue; for production
        runs that silently corrupts checkpoints. Mirrors the adapter-team
        intrinsics setup.
        """

        def on_log(self, args, state, control, logs=None, **kwargs):
            import math
            if logs is None:
                return control
            loss = logs.get("loss")
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                control.should_training_stop = True
            return control

    return _StopOnNanCallback()
