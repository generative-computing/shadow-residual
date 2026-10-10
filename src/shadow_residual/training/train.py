# SPDX-License-Identifier: Apache-2.0
"""Unified adapter-training entry point.

Usage:
    python -m shadow_residual.training.train --config path/to/config.yaml [overrides]

The YAML is required. Named CLI flags override matching YAML fields. The
adapter is always active (plain LoRA, no gated activation). Two optional
adapter fields shape the training data/labels and are recorded into the saved
adapter_config.json: `adapter.last_context_token` (the final context token;
the supervised region must start right after it — validated) and
`adapter.last_token` (end-of-completion marker; appended to rows that lack it).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from shadow_residual.config import load_training_config
from shadow_residual.config.adapters import (
    build_callbacks,
    to_peft_config,
    to_training_arguments,
)
from shadow_residual.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shadow_residual.training.train",
        description="Unified adapter-training entry point.",
    )
    p.add_argument("--config", required=True, type=str, help="Path to unified training config YAML.")

    # Hand-picked overrides for the most-edited fields.
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--train-data", type=str, default=None)
    p.add_argument("--val-data", type=str, default=None)
    p.add_argument("--lr", type=float, default=None, help="Override optimizer.learning_rate")
    p.add_argument("--epochs", type=int, default=None, help="Override runtime.num_train_epochs")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap training to N optimizer steps (TRL max_steps; takes precedence "
                        "over epochs). Use for short diagnostic runs that still exercise the "
                        "post-training generate path.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None,
                   help="Override adapter.target_modules to a uniform scalar rank")
    p.add_argument("--lora-alpha", type=int, default=None, help="Override adapter.alpha")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch.per_device_train")
    p.add_argument("--grad-accum", type=int, default=None, help="Override batch.gradient_accumulation")

    p.add_argument("--skip-post-train-generate", action="store_true",
                   help="Skip the in-process post-training generation step. Use when generation "
                        "runs as a separate process (a fresh `python -m ...generate` against the "
                        "saved checkpoint) — avoids the post-training CUDA residue that OOMs the "
                        "in-process FSDP gather+rebuild at 30B.")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="Disable non-reentrant gradient checkpointing for gated SR models. "
                        "Diagnostic only — safe when activations fit memory (3B/DDP). Used to "
                        "isolate whether a training issue is a checkpoint-recompute interaction.")
    p.add_argument("--quantize", action="store_true", help="Enable 4-bit QLoRA via bitsandbytes")
    p.add_argument("--thinking", "--enable-thinking", dest="enable_thinking", action="store_true",
                   help="Set data.enable_thinking=True — forwarded to "
                        "apply_chat_template(enable_thinking=...) when rendering training rows.")
    p.add_argument("--instruction-as-user-message", dest="instruction_as_user_message",
                   action="store_true",
                   help="Set data.instruction_as_user_message=True — inject RAG documents "
                        "before the first assistant turn (judge/guardian data) instead of "
                        "after the last user turn. ChatML (Granite 4.2/5.0) only.")
    p.add_argument("--cross-lr", type=float, default=None,
                   help="Separate learning rate for cross_stream (W-cross) LoRA params. When set, "
                        "builds a two-group optimizer: cross_stream params at --cross-lr, all other "
                        "trainable params at the base learning_rate. Diagnostic for the SR-overfit "
                        "hypothesis (regularize W-cross with a lower LR, or probe with a higher one).")
    p.add_argument("--wandb", action="store_true", help="Set runtime.report_to=['wandb']")
    p.add_argument("--debug-collator", action="store_true",
                   help="Periodically log a decoded training example + labels from the collator. "
                        "Also enables generation debug printing at the same probability "
                        "(unless generation.debug_probability is already set in the config).")
    return p


def apply_cli_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> None:
    """Mutate cfg in place with whatever CLI flags the user supplied."""
    if args.output_dir is not None:
        cfg.save.output_dir = args.output_dir
    if args.train_data is not None:
        cfg.data.train_path = args.train_data
    if args.val_data is not None:
        cfg.data.val_path = args.val_data
    if args.lr is not None:
        cfg.optimizer.learning_rate = args.lr
    if args.epochs is not None:
        cfg.runtime.num_train_epochs = args.epochs
    if args.seed is not None:
        cfg.runtime.seed = args.seed
    if args.lora_r is not None:
        # CLI override sets a uniform scalar rank, replacing any per-module dict.
        cfg.adapter.target_modules = args.lora_r
    if args.lora_alpha is not None:
        cfg.adapter.alpha = args.lora_alpha
    if args.batch_size is not None:
        cfg.batch.per_device_train = args.batch_size
    if args.grad_accum is not None:
        cfg.batch.gradient_accumulation = args.grad_accum
    if args.wandb:
        cfg.runtime.report_to = ["wandb"]
    if args.enable_thinking:
        cfg.data.enable_thinking = True
    if args.instruction_as_user_message:
        cfg.data.instruction_as_user_message = True
    if args.debug_collator:
        cfg.trainer.debug_collator = True
        # Also turn on generation debug printing at the same rate, unless the
        # config already set it explicitly (don't clobber an explicit choice).
        # No-op when the generation block is absent (post-training gen disabled).
        if cfg.generation is not None and cfg.generation.debug_probability == 0.0:
            cfg.generation.debug_probability = 0.001


# --- Helpers ------------------------------------------------------------------


# Used as the response_template fallback only if `cfg.data.assistant_marker`
# is unset AND the chat template lacks `{% generation %}` tags. Granite-family
# value; for other model families set `data.assistant_marker` in the YAML.
_DEFAULT_ASSISTANT_MARKER = "<|start_of_role|>assistant<|end_of_role|>"


# Module-level alias for FullyShardedDataParallel — kept as an attribute on
# this module so unit tests can monkeypatch it to a stub class without
# importing real FSDP machinery. Resolved lazily inside the dispatch helper
# to keep this module importable on machines without torch.distributed.
_FSDP_CLASS = None


def _resolve_fsdp_class():
    """Return the FullyShardedDataParallel class, importing if needed.

    Uses the module-level ``_FSDP_CLASS`` slot so unit tests can substitute
    a stub via ``monkeypatch.setattr(train, "_FSDP_CLASS", StubClass)``.
    """
    global _FSDP_CLASS
    if _FSDP_CLASS is None:
        from torch.distributed.fsdp import FullyShardedDataParallel
        _FSDP_CLASS = FullyShardedDataParallel
    return _FSDP_CLASS


def _run_post_training_generation(
    *,
    trainer,
    cfg: TrainingConfig,
    peft_config,
    tokenizer,
    output_dir: Path,
) -> None:
    """Run post-training generation, dispatching on FSDP-wrapped state.

    No-ops if ``cfg.generation`` is None.

    For FSDP-wrapped models, calls ``run_generation_under_fsdp`` on EVERY
    rank — the gather inside is a collective. After the helper returns
    the FSDP model has been torn down; we null out ``trainer.model`` so
    the trainer's shutdown path doesn't try to use the dead reference.

    For non-FSDP runs, only rank 0 generates (existing behavior).
    """
    if cfg.generation is None:
        return

    fsdp_cls = _resolve_fsdp_class()
    is_fsdp = isinstance(trainer.model, fsdp_cls)
    gen_out = output_dir / cfg.generation.output_filename

    # Lazy import — keeps this module importable in environments without
    # torch installed (the test suite stubs the runners directly).
    from shadow_residual.training import generate as generate_mod

    if is_fsdp:
        logger.info(
            "Running post-training generation under FSDP "
            "(gather + rebuild on rank 0): input=%s out=%s",
            cfg.generation.input_path, gen_out,
        )
        # The helper's `fsdp_model` arg is only ONE binding to the FSDP
        # module; `trainer.model` holds another live ref. If we don't null
        # it, the helper's `del fsdp_model` can't drop the ~16GB shard and
        # the rank-0 rebuild OOMs at 30B ([CENSUS] confirmed the residue is
        # the flat-param shard, kept alive by trainer.model). So hand the
        # helper a callback that nulls trainer.model — it fires AFTER the
        # gather (which needs the live model) and BEFORE teardown. FSDP-only:
        # the non-FSDP (3b) flow returns past this branch and is untouched.
        def _release_model_refs():
            trainer.model = None

        generate_mod.run_generation_under_fsdp(
            fsdp_model=trainer.model,
            accelerator=trainer.accelerator,
            cfg=cfg,
            peft_config=peft_config,
            tokenizer=tokenizer,
            data_path=cfg.generation.input_path,
            out_path=gen_out,
            release_model_refs=_release_model_refs,
        )
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        logger.info("Skipping post-training generation on rank %d", local_rank)
        return

    logger.info(
        "Running post-training generation: input=%s out=%s",
        cfg.generation.input_path, gen_out,
    )
    # Disable gradient checkpointing before generation. Checkpointing is a
    # training-only memory optimization; leaving it on for generate() is
    # wasteful (forward-without-backward gains nothing from recompute). Also
    # switch to eval mode so no checkpointing path is taken regardless.
    if hasattr(trainer.model, "gradient_checkpointing_disable"):
        trainer.model.gradient_checkpointing_disable()
    trainer.model.eval()
    generate_mod.run_generation(
        model=trainer.model,
        tokenizer=tokenizer,
        gen_config=cfg.generation,
        data_path=cfg.generation.input_path,
        out_path=gen_out,
    )


def _chat_template_supports_generation_tags(tokenizer) -> bool:
    """True iff the tokenizer's chat template uses {% generation %} markers.

    TRL's `SFTConfig(assistant_only_loss=True)` requires those markers to know
    which token positions belong to assistant turns. If absent, we must fall
    back to a response-template-string approach.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return False
    return "{% generation %}" in template or "{%- generation %}" in template


def _build_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# --- Main ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _build_parser().parse_args(argv)
    if args.quantize:
        raise NotImplementedError(
            "--quantize (4-bit QLoRA via bitsandbytes) is not currently "
            "supported on the shadow-residual + PEFT training path."
        )

    # Opt-in chunked cross-entropy (SR_CHUNKED_CE=1): bounds the full-vocab softmax
    # memory for tasks with long untruncated rows (is-vulnerable OOM'd at 21.67 GiB
    # on a ~37k-token row x 100k vocab). No-op unless the env flag is set.
    from shadow_residual.training._chunked_ce import maybe_patch_fixed_cross_entropy
    maybe_patch_fixed_cross_entropy()
    cfg = load_training_config(args.config)
    apply_cli_overrides(cfg, args)

    logger.info("Adapter activation: unconditional (always active)")
    logger.info("Base model:   %s", cfg.model.base)
    logger.info("Train data:   %s", cfg.data.train_path)
    logger.info("Val data:     %s", cfg.data.val_path)
    logger.info("Output dir:   %s", cfg.save.output_dir)

    # Lazy imports — keeps argparse fast, lets unit tests import the module
    # without pulling in transformers/peft.
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from shadow_residual.training.factory import (
        get_shadow_residual_peft_model,
    )

    from shadow_residual.training.data import (
        DebugPrintingCollator,
        ResponseOnlyCollator,
        load_jsonl_dataset,
        validate_last_context_token_boundary,
    )

    # 1. Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # 2. Resolve the two optional single-token markers (each must encode to
    # exactly one token id). They shape the data/labels and are recorded into
    # the saved adapter_config.json; they do NOT gate the (always-active) adapter.
    def _resolve_single_token(field_value: str | None, field_name: str) -> int | None:
        if field_value is None:
            return None
        ids = tok.encode(field_value, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"adapter.{field_name} {field_value!r} must encode to exactly one "
                f"token id for this tokenizer, got {ids}."
            )
        logger.info("%s: %r -> %d", field_name, field_value, ids[0])
        return ids[0]

    last_context_token_id = _resolve_single_token(
        cfg.adapter.last_context_token, "last_context_token"
    )
    last_token_id = _resolve_single_token(cfg.adapter.last_token, "last_token")

    # 3. PEFT config (stock LoRA — no gated/aLoRA activation).
    peft_config = to_peft_config(cfg)

    # 4. Model — built via the shadow-residual PEFT factory.
    # param_dtype selects how the model is materialized. "bf16" (legacy
    # default) loads base + adapters in bf16. "fp32" loads everything in
    # fp32; combined with accelerate mixed_precision: bf16 this yields bf16
    # compute via AMP while keeping fp32 master weights / grads / optimizer
    # state — the PEFT-recommended recipe that keeps adapter updates in fp32
    # for stability at scale (see ModelConfig.param_dtype).
    import torch
    _DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
    model_dtype = _DTYPES[cfg.model.param_dtype]
    logger.info("Loading model %s (param_dtype=%s) ...", cfg.model.base, cfg.model.param_dtype)
    if cfg.model.attn_implementation:
        logger.info("Attention backend: %s", cfg.model.attn_implementation)
    model = get_shadow_residual_peft_model(
        base_model_name_or_path=cfg.model.base,
        lora_config=peft_config,
        torch_dtype=model_dtype,
        attn_implementation=cfg.model.attn_implementation,
        share_moe_routing=cfg.adapter.share_moe_routing,
    )
    model.print_trainable_parameters()

    # 4a. Activation checkpointing.
    # The SR decoder is a GradientCheckpointingLayer; HF non-reentrant gradient
    # checkpointing drives it via __call__. Without this, a large SR forward keeps
    # all unsharded activations (FSDP shards params/grads/optimizer, NOT
    # activations) and OOMs in backward. enable_input_require_grads is required so
    # the checkpointed inputs carry grad. cfg.batch.gradient_checkpointing is the
    # single source of truth (default True); --no-grad-checkpoint is a diagnostic
    # override. The adapter is plain LoRA (no aLoRA offset hooks), so gradient
    # checkpointing is compatible with mid-training eval — no eval force-off.
    gradient_checkpointing_on = (
        cfg.batch.gradient_checkpointing
        and not getattr(args, "no_grad_checkpoint", False)
    )
    if gradient_checkpointing_on:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        logger.info(
            "Activation checkpointing: enabled non-reentrant gradient "
            "checkpointing (use_reentrant=False)."
        )
    else:
        logger.info(
            "Activation checkpointing DISABLED (cfg.batch.gradient_checkpointing=%s, "
            "--no-grad-checkpoint=%s). Only safe when activations fit memory.",
            cfg.batch.gradient_checkpointing, getattr(args, "no_grad_checkpoint", False),
        )

    # 5. Datasets. last_token (when set) is appended to rows that don't already
    # end with it — done at the text level so both masking paths inherit it.
    train_ds = load_jsonl_dataset(
        cfg.data.train_path, tok,
        enable_thinking=cfg.data.enable_thinking,
        last_token=cfg.adapter.last_token,
        instruction_as_user_message=cfg.data.instruction_as_user_message,
    )
    val_ds = load_jsonl_dataset(
        cfg.data.val_path, tok,
        enable_thinking=cfg.data.enable_thinking,
        last_token=cfg.adapter.last_token,
        instruction_as_user_message=cfg.data.instruction_as_user_message,
    )
    logger.info("Train rows: %d, Val rows: %d", len(train_ds), len(val_ds))

    # 6. SFTConfig — start from to_training_arguments, add SFT-specific keys
    ta = to_training_arguments(cfg)
    sft_kwargs = {k: v for k, v in vars(ta).items() if not k.startswith("_")}
    sft_kwargs["max_length"] = cfg.data.max_length
    sft_kwargs["packing"] = cfg.trainer.packing
    # Checkpointing is managed manually above (model.gradient_checkpointing_enable
    # with use_reentrant=False + enable_input_require_grads). Keep the Trainer's
    # own flag OFF so it doesn't re-enable it with default kwargs / double-wrap.
    sft_kwargs["gradient_checkpointing"] = False

    # 6a. Response-only label masking strategy.
    # Preferred: TRL's `assistant_only_loss=True` (driven by `{% generation %}`
    # markers in the chat template). Fallback: our own ResponseOnlyCollator
    # (TRL-version-independent; doesn't depend on TRL's removed
    # DataCollatorForCompletionOnlyLM).
    use_assistant_only_loss = _chat_template_supports_generation_tags(tok)
    data_collator = None
    if use_assistant_only_loss:
        sft_kwargs["assistant_only_loss"] = True
        logger.info("Response-only masking via SFTConfig(assistant_only_loss=True).")
    else:
        marker = cfg.data.assistant_marker or _DEFAULT_ASSISTANT_MARKER
        data_collator = ResponseOnlyCollator(tokenizer=tok, response_template=marker)
        logger.info(
            "Chat template lacks {%% generation %%} tags; using "
            "ResponseOnlyCollator(response_template=%r).", marker,
        )

    # Diagnostic short-run: cap optimizer steps. TRL/transformers treats
    # max_steps > 0 as taking precedence over num_train_epochs, so a few
    # steps still produce a trained model + the post-training generate path
    # (and its post-training CUDA residue) without a full epoch.
    if getattr(args, "max_steps", None) is not None:
        sft_kwargs["max_steps"] = args.max_steps
        logger.info("Capping training to max_steps=%d (overrides epochs).", args.max_steps)

    sft_args = SFTConfig(**_filter_to_sftconfig_fields(sft_kwargs))

    # 6b. Optional per-module LR for cross_stream (W-cross). Default None → single
    # LR as before. When --cross-lr is set, build a two-group AdamW: cross_stream
    # trainable params at cross_lr, everything else at the base LR. Diagnostic for
    # the SR-overfit hypothesis (W-cross capacity hurts generalization at the
    # shared LR; a lower cross LR regularizes it).
    optimizers = (None, None)
    cross_lr = getattr(args, "cross_lr", None)
    if cross_lr is not None:
        import torch
        base_lr = cfg.optimizer.learning_rate
        cross_params, other_params = [], []
        for n, pm in model.named_parameters():
            if not pm.requires_grad:
                continue
            (cross_params if "cross_stream" in n else other_params).append(pm)
        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": cross_params, "lr": cross_lr},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr, weight_decay=cfg.optimizer.weight_decay,
        )
        optimizers = (optimizer, None)
        logger.info(
            "Per-module LR: cross_stream=%g (%d params), base=%g (%d params).",
            cross_lr, sum(p.numel() for p in cross_params),
            base_lr, sum(p.numel() for p in other_params),
        )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_args,
        processing_class=tok,
        data_collator=data_collator,  # None → SFTTrainer uses its default
        callbacks=build_callbacks(cfg),
        optimizers=optimizers,
    )

    # 7a. last_context_token boundary check — fail loudly unless the supervised
    # region (labels != -100) starts immediately after last_context_token. Run
    # against trainer.data_collator (the REAL collator TRL settled on), so it
    # covers both masking paths uniformly: the assistant_only_loss path (where
    # TRL owns masking — we can only validate) and our ResponseOnlyCollator.
    if cfg.adapter.last_context_token is not None:
        validate_last_context_token_boundary(
            train_ds, tok, trainer.data_collator, cfg.adapter.last_context_token,
        )

    # Optional: wrap whatever collator SFTTrainer ended up with so we
    # periodically log a decoded example + its labels. Covers both the
    # assistant_only_loss path (TRL-internal collator) and the
    # ResponseOnlyCollator fallback.
    if cfg.trainer.debug_collator:
        trainer.data_collator = DebugPrintingCollator(
            trainer.data_collator, tokenizer=tok, probability=0.001
        )
        logger.info("Debug collator enabled (logs ~0.1%% of batches).")

    # 8. Train + save
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    out = Path(cfg.save.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    logger.info("Saved adapter + tokenizer to %s", out)

    # Guard: a separate-process generate step (and any downstream reload) needs
    # a consolidated PEFT adapter on disk. Under FSDP this only happens when the
    # accelerate config uses FULL_STATE_DICT — with SHARDED_STATE_DICT,
    # save_model writes .distcp shards and NO adapter_model.safetensors, which
    # loads as nothing. Fail loudly on rank 0 rather than ship an unloadable
    # checkpoint silently.
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        missing = [
            name for name in ("adapter_config.json", "adapter_model.safetensors")
            if not (out / name).exists()
        ]
        if missing:
            raise RuntimeError(
                f"Saved checkpoint at {out} is missing {missing} — got a sharded/"
                "non-consolidated save. A fresh generate process can't load this. "
                "Set fsdp_state_dict_type: FULL_STATE_DICT in the accelerate config "
                "(src/shadow_residual/config/accelerate/fsdp_4gpu.yaml) so the FSDP save consolidates."
            )

        # Record SR-specific metadata into the saved adapter_config.json as extra
        # keys (stock LoraConfig doesn't serialize them). Stock
        # PeftConfig.from_pretrained tolerates unknown keys, so this doesn't
        # break reload. Two kinds:
        #   - the two token markers (string form + resolved id for the training
        #     tokenizer);
        #   - share_moe_routing: the MoE routing mode this adapter was trained
        #     under. It is a forward-path choice, not a weight, so the saved
        #     adapter_model.safetensors is identical either way — without recording
        #     it here the adapter has no memory of its routing mode, and a served
        #     model built with the wrong build_sr_base(share_moe_routing=...) would
        #     silently diverge from training. Persisting it lets the serving path
        #     read it back (see training.generation_utils.
        #     read_share_moe_routing_from_adapter) so the
        #     adapter self-describes.
        adapter_cfg_path = out / "adapter_config.json"
        if adapter_cfg_path.exists():
            import json as _json
            with adapter_cfg_path.open("r") as f:
                adapter_cfg = _json.load(f)
            if cfg.adapter.last_context_token is not None:
                adapter_cfg["last_context_token"] = cfg.adapter.last_context_token
                adapter_cfg["last_context_token_id"] = last_context_token_id
            if cfg.adapter.last_token is not None:
                adapter_cfg["last_token"] = cfg.adapter.last_token
                adapter_cfg["last_token_id"] = last_token_id
            adapter_cfg["share_moe_routing"] = bool(cfg.adapter.share_moe_routing)
            with adapter_cfg_path.open("w") as f:
                _json.dump(adapter_cfg, f, indent=2)
            logger.info(
                "Patched adapter_config.json with last_context_token=%r "
                "last_token=%r share_moe_routing=%r.",
                cfg.adapter.last_context_token, cfg.adapter.last_token,
                bool(cfg.adapter.share_moe_routing),
            )

    # 9. Post-training generation, gated on the generation block being present.
    # Generation methodology lives in generate.run_generation — train.py just
    # invokes it with the in-memory model so we don't pay for a base+adapter
    # reload from disk. Under FSDP, the helper dispatches into
    # run_generation_under_fsdp (gather + rebuild on rank 0) instead.
    #
    # --skip-post-train-generate disables this so generation can run as a
    # separate process against the saved checkpoint — a fresh interpreter gives
    # generate a clean CUDA context, sidestepping the post-training residue that
    # OOMs the in-process FSDP gather+rebuild at 30B.
    if args.skip_post_train_generate:
        logger.info(
            "Skipping in-process post-training generation "
            "(--skip-post-train-generate); run generate as a separate process."
        )
    else:
        _run_post_training_generation(
            trainer=trainer,
            cfg=cfg,
            peft_config=peft_config,
            tokenizer=tok,
            output_dir=out,
        )

    return 0


def _filter_to_sftconfig_fields(kwargs: dict) -> dict:
    """Drop keys SFTConfig doesn't accept (it's a TrainingArguments subset+)."""
    from trl import SFTConfig
    valid = set(SFTConfig.__dataclass_fields__.keys()) if hasattr(SFTConfig, "__dataclass_fields__") else None
    if valid is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in valid}


if __name__ == "__main__":
    sys.exit(main())
