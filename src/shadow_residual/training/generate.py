# SPDX-License-Identifier: Apache-2.0
"""Post-training generation: reusable function + standalone CLI.

The ``run_generation`` function is the single source of truth for
generation methodology. It is called from two places:

  1. ``train.py`` after training completes, with the in-memory PEFT model.
  2. The standalone CLI in this module, which loads a saved checkpoint
     from disk and applies it to the base model first.

Standalone usage:

    python -m shadow_residual.training.generate \
        --config path/to/training_config.yaml \
        --checkpoint /path/to/saved/adapter \
        [--data /path/to/eval.jsonl]   # defaults to cfg.generation.input_path
        [--out /path/to/predictions.jsonl]   # defaults to <checkpoint>/<cfg.generation.output_filename>

Input row shape (JSONL): ``{"messages": [...], "tools": [...]?,
"documents": [...]?, "ground_truth": "..."?}``. ``messages`` should NOT
include a trailing gold-answer assistant turn — eval rows put the gold
in a separate ``ground_truth`` key (the scorer reads it). The generator
renders ``messages`` as-is with ``add_generation_prompt=True``; the
model fills in what comes after the empty assistant role-tag.

Output row shape (JSONL): the input row preserved verbatim, with one
extra key ``generated_content`` holding the newly-generated text. A
downstream scorer reads gold from ``ground_truth`` and prediction from
``generated_content``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from shadow_residual.config import load_training_config
from shadow_residual.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shadow_residual.training.generate",
        description="Generate predictions from a trained adapter on an eval set.",
    )
    p.add_argument("--config", required=True, help="Training config YAML (same one used for training).")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a saved LoRA adapter directory (output of train.py).")
    p.add_argument("--data", default=None,
                   help="Override eval JSONL path (defaults to cfg.data.val_path).")
    p.add_argument("--out", default=None,
                   help="Override output JSONL path (defaults to <checkpoint>/<cfg.generation.output_filename>).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional row cap for quick smoke runs.")
    p.add_argument("--debug-generate", type=float, default=None,
                   metavar="PROB",
                   help="Override generation.debug_probability (e.g. 0.01 logs "
                        "decoded prompt + generated text for ~1%% of batches).")
    return p


# --- Helpers ------------------------------------------------------------------


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL of ``{messages, tools?, documents?}`` rows.

    No assumption is made about whether the trailing message is an
    assistant turn — that's the caller's concern (``run_generation``
    strips it role-aware when building the prompt).
    """
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"{path}: row missing 'messages' field: {obj!r}")
            if not obj["messages"]:
                raise ValueError(f"{path}: empty 'messages' list")
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _resolve_paths(cfg: TrainingConfig, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Return (data_path, checkpoint_path, output_path) after applying overrides.

    Requires ``cfg.generation`` to be present (CLI rejects None earlier).
    """
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = Path(cfg.generation.input_path)
    checkpoint_path = Path(args.checkpoint)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = checkpoint_path / cfg.generation.output_filename
    return data_path, checkpoint_path, out_path


# --- Generation loop ---------------------------------------------------------


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    cfg_gen,
    debug_probability: float = 0.0,
) -> list[str]:
    """Run model.generate on a list of pre-rendered prompt strings.

    Returns the *newly generated* text only (prompt is sliced off).

    When ``debug_probability > 0``, periodically logs a decoded prompt +
    generated text for one example in the batch — mirrors
    ``DebugPrintingCollator`` so the prediction path is as inspectable as
    training is.
    """
    import random

    import torch

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,  # chat template already added them
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=cfg_gen.max_new_tokens,
        do_sample=cfg_gen.do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if cfg_gen.do_sample:
        gen_kwargs["temperature"] = cfg_gen.temperature
        gen_kwargs["top_p"] = cfg_gen.top_p

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    # Slice off the prompt portion of each row to get only newly generated tokens.
    # We left-pad (set above), so every row's prompt occupies positions
    # [seq_len - prompt_len_i, seq_len) and generated tokens begin at seq_len.
    # Slicing at `prompt_lens[i]` (the unpadded length) would include the
    # leftmost padded chunk of the prompt — for short prompts that means
    # decoded predictions start with chunks of the prompt itself. The right
    # slice point is the padded length (same for every row in the batch).
    padded_prompt_len = enc["input_ids"].shape[1]
    predictions: list[str] = []
    for i in range(out.shape[0]):
        new_tokens = out[i, padded_prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions.append(text)

    if debug_probability > 0.0 and random.random() < debug_probability:
        i = 0
        prompt_ids = enc["input_ids"][i].tolist()
        new_ids = out[i, padded_prompt_len:].tolist()
        logger.info("=" * 70)
        logger.info("DEBUG GENERATION OUTPUT")
        logger.info("=" * 70)
        logger.info("Example %d from batch (batch_size=%d):", i, len(prompts))
        logger.info("\nPrompt text:\n%s", prompts[i])
        logger.info("\nGenerated text:\n%s", predictions[i])
        logger.info("\nPrompt IDs (padded, len=%d): %s", len(prompt_ids), prompt_ids)
        logger.info("\nGenerated IDs (len=%d): %s", len(new_ids), new_ids)
        logger.info("=" * 70)

    return predictions


# --- Reusable run --------------------------------------------------------------


def run_generation(
    *,
    model,
    tokenizer,
    gen_config,
    data_path: str | Path,
    out_path: str | Path,
    limit: int | None = None,
) -> int:
    """Run generation given an already-loaded model + tokenizer.

    This is the single source of truth for generation methodology — both
    train.py (post-training, with the in-memory PEFT model) and the
    standalone CLI in this module call into here. Don't replicate this
    loop elsewhere.

    Args:
        model: a model with a ``.generate(...)`` method (PEFT-wrapped or not).
        tokenizer: must be left-padded (we set this up here).
        gen_config: a ``GenerationConfig`` instance.
        data_path: JSONL of ``{messages, tools?, documents?}`` rows. If a
            row's last message is an assistant turn, it's treated as the
            gold target and stripped before prompt rendering.
        out_path: where to write predictions JSONL. Each output row is the
            input row preserved verbatim with one extra
            ``generated_content`` key.
        limit: optional cap on number of rows.

    Returns:
        Number of predictions written.
    """
    import torch

    data_path = Path(data_path)
    out_path = Path(out_path)

    # Generation requires left-padding so every sequence ends at the same
    # position. Stash + restore in case the caller (train.py) wants
    # right-padding back afterwards for any reason.
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        rows = _load_jsonl_rows(data_path)
        if limit is not None:
            rows = rows[:limit]
        logger.info("Loaded %d generation rows from %s", len(rows), data_path)
        debug_prob = getattr(gen_config, "debug_probability", 0.0)
        if debug_prob > 0.0:
            logger.info(
                "Debug generation enabled (logs ~%.2f%% of batches).",
                debug_prob * 100,
            )

        prompts: list[str] = []
        for row in rows:
            # Eval rows are expected to NOT include a trailing gold-answer
            # assistant turn — the gold lives in a separate ``ground_truth``
            # field if present (read by the scorer). ``messages`` is
            # rendered as-is here; ``add_generation_prompt=True`` appends
            # the empty assistant role-tag so the model fills in what
            # comes next.
            prompts.append(tokenizer.apply_chat_template(
                row["messages"],
                tools=row.get("tools"),
                documents=row.get("documents"),
                tokenize=False,
                add_generation_prompt=True,
            ))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        bs = gen_config.batch_size
        n_done = 0
        was_training = model.training
        model.eval()
        try:
            with out_path.open("w") as f:
                i = 0
                while i < len(rows):
                    batch_prompts = prompts[i : i + bs]
                    # Adaptive batch size: a large 30B + long-document prompts can
                    # OOM a single-GPU generation at a given batch size. Rather than
                    # hardcode a conservative batch_size (which assumes prompt
                    # lengths) or fail, halve the batch and retry the same slice on
                    # OOM, down to 1. This adapts to any prompt length without data
                    # assumptions and supports larger models / longer contexts.
                    try:
                        batch_predictions = _generate_batch(
                            model, tokenizer, batch_prompts, gen_config,
                            debug_probability=getattr(gen_config, "debug_probability", 0.0),
                        )
                    except torch.cuda.OutOfMemoryError:
                        if bs == 1:
                            raise
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        new_bs = max(1, bs // 2)
                        logger.warning(
                            "Generation OOM at batch_size=%d; halving to %d and retrying.",
                            bs, new_bs,
                        )
                        bs = new_bs
                        continue
                    for j, pred in enumerate(batch_predictions):
                        row_idx = i + j
                        # Preserve the input row verbatim, append the new key.
                        out_row = dict(rows[row_idx])
                        out_row["generated_content"] = pred
                        f.write(json.dumps(out_row) + "\n")
                        n_done += 1
                    i += len(batch_prompts)
                    logger.info("  generated %d / %d", n_done, len(rows))
        finally:
            if was_training:
                model.train()
        logger.info("Wrote %d predictions to %s", n_done, out_path)
        return n_done
    finally:
        tokenizer.padding_side = saved_padding_side


# --- FSDP-aware variant -------------------------------------------------------


def _cuda_mem_snapshot(tag: str, *, device: int | None = None) -> dict[str, float]:
    """Log + return a CUDA memory snapshot for ``device`` (default: current).

    Captures allocated / reserved / max-allocated and the
    allocator-vs-driver gap so we can attribute a shortage to live tensors
    (allocated), allocator fragmentation (reserved - allocated), or
    out-of-band driver usage (driver - reserved).
    """
    import torch

    if not torch.cuda.is_available():
        return {}
    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_alloc = torch.cuda.max_memory_allocated(device) / 1e9
    max_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    free_b, total_b = torch.cuda.mem_get_info(device)
    free = free_b / 1e9
    total = total_b / 1e9
    # Driver-side usage = total - free. Memory held outside the torch
    # caching allocator (other processes, NCCL buffers, cuda context) =
    # driver_used - reserved.
    driver_used = total - free
    out_of_allocator = driver_used - reserved

    logger.info(
        "[MEMSNAP %-22s] cuda:%d alloc=%.2f reserved=%.2f (frag=%.2f) "
        "max_alloc=%.2f max_reserved=%.2f driver_used=%.2f out_of_allocator=%.2f free=%.2f total=%.2f GB",
        tag, device, allocated, reserved, reserved - allocated, max_alloc,
        max_reserved, driver_used, out_of_allocator, free, total,
    )
    return {
        "tag": tag, "device": device, "allocated": allocated,
        "reserved": reserved, "frag": reserved - allocated,
        "max_alloc": max_alloc, "max_reserved": max_reserved,
        "driver_used": driver_used, "out_of_allocator": out_of_allocator,
        "free": free, "total": total,
    }


def _cuda_tensor_census(tag: str, *, device: int | None = None, top: int = 15) -> None:
    """Walk the Python GC graph and tally live CUDA tensors on ``device``.

    Groups tensors by (dtype, requires_grad, is_leaf) and reports the
    largest single tensors. This is how we attribute residue to model
    params vs grads vs optimizer state vs autograd-saved activations:
      - requires_grad=True, is_leaf=True  -> parameters
      - the ``.grad`` of those             -> gradients
      - requires_grad=False, large         -> optimizer exp_avg/exp_avg_sq
        or detached buffers / saved tensors.
    """
    import gc

    import torch

    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.cuda.current_device()

    buckets: dict[tuple, list[int]] = {}
    big: list[tuple[int, str]] = []
    total_bytes = 0
    n = 0
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj):
                # Also catch tensors wrapped in nn.Parameter / .data
                t = getattr(obj, "data", None)
                if not torch.is_tensor(t):
                    continue
                obj = t
            if not obj.is_cuda or obj.device.index != device:
                continue
        except Exception:
            continue
        nbytes = obj.element_size() * obj.nelement()
        if nbytes == 0:
            continue
        total_bytes += nbytes
        n += 1
        key = (str(obj.dtype), bool(obj.requires_grad), bool(obj.is_leaf))
        b = buckets.setdefault(key, [0, 0])
        b[0] += 1
        b[1] += nbytes
        if nbytes >= 50 * 1024 * 1024:  # >= 50MB
            big.append((nbytes, f"{tuple(obj.shape)} {obj.dtype} "
                                f"req_grad={obj.requires_grad} leaf={obj.is_leaf}"))

    logger.info("[CENSUS %-22s] cuda:%d live tensors=%d total=%.2f GB",
                tag, device, n, total_bytes / 1e9)
    for key in sorted(buckets, key=lambda k: -buckets[k][1]):
        cnt, nb = buckets[key]
        logger.info("[CENSUS %-22s]   dtype=%-14s req_grad=%-5s leaf=%-5s : "
                    "%4d tensors, %.2f GB", tag, key[0], key[1], key[2],
                    cnt, nb / 1e9)
    big.sort(reverse=True)
    for nbytes, desc in big[:top]:
        logger.info("[CENSUS %-22s]   BIG %.2f GB  %s", tag, nbytes / 1e9, desc)


def run_generation_under_fsdp(
    *,
    fsdp_model,
    accelerator,
    cfg,
    peft_config,
    tokenizer,
    data_path: str | Path,
    out_path: str | Path,
    limit: int | None = None,
    release_model_refs=None,
) -> int:
    """Post-training generate when ``trainer.model`` is FSDP-wrapped.

    Recipe verified end-to-end at 30B / 4-rank / 4×80GB H100 by
    ``scripts/diagnose_fsdp_generate.py --run-attempt-10``:

      1. ALL RANKS: gather FULL_STATE_DICT (rank0_only=True, offload_to_cpu=True).
      2. ALL RANKS: ``release_model_refs()`` — null the CALLER's reference to
         the FSDP module (e.g. ``trainer.model``). This is load-bearing: the
         helper's ``fsdp_model`` is only one binding; the trainer holds another
         live ref, so ``del fsdp_model`` alone leaves the ~16GB shard pinned
         (confirmed by [CENSUS after-teardown] — the rebuild then OOMs at 30B).
      3. ALL RANKS: ``accelerator.free_memory()`` — drops the ~29GB CUDA
         caching-allocator reserved arena that survives a plain ``empty_cache``.
      4. ALL RANKS: ``del fsdp_model + gc.collect + empty_cache`` — with the
         caller ref already gone, this drops the last ref to the flat-param
         shard (rank 0 returns to ~0GB).
      5. RANK 0 ONLY: rebuild fresh non-FSDP SR+PEFT (meta init) →
         load_state_dict → .to(cuda:0) → ``run_generation(...)``.
      6. ALL RANKS: barrier (others wait at the post-gather barrier).

    NOTE: this destroys the FSDP-wrapped model. The caller cannot resume
    training afterwards. Right shape for END-OF-TRAINING generate; for
    mid-training eval that needs to keep training, use a different recipe
    (gather + ``device_map='auto'`` rebuild that keeps the FSDP shards alive).

    ``release_model_refs``: optional zero-arg callable invoked AFTER the gather
    (which needs the live model) and BEFORE teardown. It must null every
    caller-side reference to the FSDP module. Without it, the rebuild OOMs at
    30B. Optional only so existing tests/callers that already hold no extra ref
    don't break; production callers (train.py) MUST pass it.

    Scale: single-GPU rebuild on rank 0. The 30B unsharded model is ~60GB,
    fits on an 80GB H100 with ~20GB headroom for KV cache + activations.
    For 500B+, swap ``.to('cuda:0')`` for ``device_map='auto'`` (TP-shards
    the rebuilt model across all ranks; would also need all ranks to
    participate in the rebuild instead of just rank 0).

    Returns the number of predictions written (0 on non-rank-0 ranks).
    """
    import gc

    import torch
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    from shadow_residual.peft_shadow_residual.factory import (
        _build_sr_config,
        _build_sr_peft_model_meta,
    )

    is_rank0 = accelerator.process_index == 0

    # MEMORY ANALYTICS: snapshot on ENTRY, before any teardown. On rank 0
    # this captures the post-training residue that the clean-card diagnostic
    # never reproduced (optimizer state / grads / autograd graph left by
    # trainer.train()). The census attributes it to params vs grads vs
    # optimizer state.
    _cuda_mem_snapshot("entry/pre-gather")
    if is_rank0:
        _cuda_tensor_census("entry/pre-gather")

    logger.info(
        "[FSDP-generate] gathering FULL_STATE_DICT (rank0_only=True, "
        "offload_to_cpu=True)..."
    )
    gathered = None
    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        sd = fsdp_model.state_dict()
        if is_rank0:
            gathered = sd
            logger.info("[FSDP-generate] gathered %d tensors on rank 0 (CPU).", len(gathered))
        else:
            del sd
    accelerator.wait_for_everyone()

    # Drop the FSDP-wrapped model. Three distinct holders of the shard,
    # each needs its own kill:
    #   - the CALLER's reference (e.g. trainer.model). The helper's
    #     `fsdp_model` is a SECOND binding to the same module; nulling only
    #     the local leaves the shard pinned (~16GB at 30B) via the caller
    #     ref. release_model_refs() nulls it. MUST run after the gather.
    #   - accelerator._models / the ~29GB reserved allocator arena —
    #     accelerator.free_memory() clears the lists and empty_caches.
    #   - the helper's local `fsdp_model` ref — del + gc + empty_cache.
    _cuda_mem_snapshot("after-gather")

    if release_model_refs is not None:
        logger.info("[FSDP-generate] release_model_refs() (nulls caller's model ref, e.g. trainer.model)...")
        release_model_refs()
    else:
        logger.warning(
            "[FSDP-generate] no release_model_refs callback given; if the "
            "caller holds another reference to the FSDP module (e.g. "
            "trainer.model), the shard will stay pinned and the rebuild may OOM."
        )

    logger.info("[FSDP-generate] accelerator.free_memory() (drops the reserved arena)...")
    accelerator.free_memory()
    _cuda_mem_snapshot("after-free_memory")

    logger.info("[FSDP-generate] del fsdp_model + gc + empty_cache (drops the FSDP shard)...")
    del fsdp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    # MEMORY ANALYTICS: what survived teardown. If `allocated` is still
    # large here, the teardown is NOT reclaiming everything — the census
    # tells us which tensors are pinned (the suspected ~16GB residue).
    _cuda_mem_snapshot("after-teardown")
    if is_rank0:
        _cuda_tensor_census("after-teardown")

    # GUARD: the teardown is supposed to return rank 0 to ~0GB allocated so
    # the unsharded rebuild (~58GB at 30B) fits with headroom for the KV
    # cache. If a holder of the FSDP shard was missed (e.g. trainer.optimizer
    # not dropped by the caller), allocated stays ~16GB and the rebuild OOMs.
    # Warn loudly here so a future regression is diagnosable from the log
    # rather than surfacing as an opaque OOM mid-generation. Threshold is
    # generous (2GB) to tolerate small allocator/context residue.
    if is_rank0 and torch.cuda.is_available():
        residue_gb = torch.cuda.memory_allocated(0) / 1e9
        if residue_gb > 2.0:
            logger.warning(
                "[FSDP-generate] cuda:0 still holds %.2f GB allocated after "
                "teardown (expected ~0). A reference to the FSDP shard was "
                "not released (likely the optimizer) — the rebuild may OOM. "
                "See [CENSUS after-teardown] above for the pinned tensors.",
                residue_gb,
            )

    n_done = 0
    if is_rank0:
        logger.info("[FSDP-generate] rebuilding non-FSDP SR+PEFT model on rank 0 (meta init)...")
        sr_config = _build_sr_config(
            cfg.model.base,
            torch_dtype=torch.bfloat16,
            attn_implementation=cfg.model.attn_implementation,
            shared_base_kv=cfg.model.shared_base_kv,
        )
        rebuilt = _build_sr_peft_model_meta(
            sr_config,
            peft_config,
            torch_dtype=torch.bfloat16,
            adapter_name="default",
        )
        logger.info("[FSDP-generate] materializing on CPU + load_state_dict...")
        rebuilt.to_empty(device="cpu")
        missing, unexpected = rebuilt.load_state_dict(gathered, strict=False)
        logger.info(
            "[FSDP-generate] load_state_dict: missing=%d unexpected=%d",
            len(missing), len(unexpected),
        )
        if missing:
            logger.warning("[FSDP-generate] missing keys (first 5): %s", missing[:5])
        if unexpected:
            logger.warning("[FSDP-generate] unexpected keys (first 5): %s", unexpected[:5])
        del gathered
        gc.collect()
        _cuda_mem_snapshot("rank0/pre-.to(cuda:0)", device=0)

        logger.info("[FSDP-generate] moving rebuilt model to cuda:0...")
        rebuilt = rebuilt.to("cuda:0")
        rebuilt.eval()
        # MEMORY ANALYTICS: the full unsharded model is now resident. This
        # snapshot + the entry residue tells us the headroom left for
        # generation (KV cache + activations). If headroom < a few GB, the
        # generate() OOM is explained by residue, not by the model size.
        _cuda_mem_snapshot("rank0/after-.to(cuda:0)", device=0)
        _cuda_tensor_census("rank0/after-.to(cuda:0)", device=0)

        logger.info("[FSDP-generate] running generation on rank 0...")
        try:
            n_done = run_generation(
                model=rebuilt,
                tokenizer=tokenizer,
                gen_config=cfg.generation,
                data_path=data_path,
                out_path=out_path,
                limit=limit,
            )
            _cuda_mem_snapshot("rank0/after-generation", device=0)
        except Exception:
            # Capture the memory state at the moment of failure (e.g. OOM)
            # before the stack unwinds — this is the snapshot that matters.
            _cuda_mem_snapshot("rank0/at-generation-FAILURE", device=0)
            _cuda_tensor_census("rank0/at-generation-FAILURE", device=0)
            raise
        finally:
            del rebuilt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    return n_done


# --- Main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    cfg = load_training_config(args.config)
    if cfg.generation is None:
        raise ValueError(
            f"{args.config}: missing 'generation:' block. The standalone "
            "generate CLI needs generation parameters (max_new_tokens, "
            "batch_size, ...) — add a generation block to the config."
        )
    data_path, checkpoint_path, out_path = _resolve_paths(cfg, args)
    if args.debug_generate is not None:
        cfg.generation.debug_probability = args.debug_generate

    logger.info("Base model:    %s", cfg.model.base)
    logger.info("Checkpoint:    %s", checkpoint_path)
    logger.info("Eval data:     %s", data_path)
    logger.info("Output:        %s", out_path)
    logger.info("Generation:    max_new_tokens=%d batch_size=%d do_sample=%s temp=%s top_p=%s",
                cfg.generation.max_new_tokens, cfg.generation.batch_size,
                cfg.generation.do_sample, cfg.generation.temperature, cfg.generation.top_p)

    # Lazy imports — keep argparse fast and the schema importable without
    # transformers/peft installed.
    import torch
    from transformers import AutoTokenizer

    from shadow_residual.peft_shadow_residual import (
        load_shadow_residual_peft_model,
    )

    # Load tokenizer (prefer the one saved next to the adapter; some adapters
    # carry tokenizer changes worth preserving).
    tokenizer_src = checkpoint_path if (checkpoint_path / "tokenizer_config.json").exists() else cfg.model.base
    logger.info("Loading tokenizer from %s ...", tokenizer_src)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_src))

    # Load via the SR loader unconditionally. For LoRA/aLoRA checkpoints
    # (no "cross_stream" in target_modules), the SR model's runtime gate
    # takes the single-stream early-exit path — semantically equivalent
    # to upstream Granite + LoRA, but not bit-identical due to unfused
    # projections (see load.py warning). For SR checkpoints, this is the
    # only correct path: the cross-stream weights need CrossStream sites
    # to bind to, which only exist on a ShadowResidualForCausalLM.
    logger.info("Loading SR base + adapter from %s ...", checkpoint_path)
    model = load_shadow_residual_peft_model(
        cfg.model.base,
        str(checkpoint_path),
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
        shared_base_kv=cfg.model.shared_base_kv,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    run_generation(
        model=model,
        tokenizer=tok,
        gen_config=cfg.generation,
        data_path=data_path,
        out_path=out_path,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
