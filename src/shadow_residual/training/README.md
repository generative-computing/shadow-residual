# Unified training

Two entry points that consume the unified Pydantic-validated YAML schema:

- `train.py` — build an SR + LoRA model and fine-tune it on a base model (CLI + YAML).
- `generate.py` — run a trained adapter against an eval set, write predictions JSONL.

The schema deliberately does not name specific architectures (LoRA, aLoRA,
shadow-residual). Training behavior is determined by which fields are set —
most importantly `adapter.invocation_tokens` (gated activation when non-null,
plain unconditional adapter when null) and whether `"cross_stream"` appears in
`adapter.target_modules` (turns on the SR dual-stream forward).

## Quick start

```bash
# Train
python -m shadow_residual.training.train \
    --config src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv.yaml \
    --train-data /data/train.jsonl \
    --val-data   /data/val.jsonl \
    --output-dir ./checkpoints

# Multi-GPU via accelerate
accelerate launch --num_processes 4 --multi_gpu --mixed_precision bf16 \
    -m shadow_residual.training.train \
    --config src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv.yaml \
    --train-data /data/train.jsonl --val-data /data/val.jsonl \
    --output-dir ./checkpoints

# After training, generate predictions on the eval set
python -m shadow_residual.training.generate \
    --config src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv.yaml \
    --checkpoint ./checkpoints --data /data/eval.jsonl --out ./checkpoints/predictions.jsonl
```

The YAML supplies everything; named CLI flags are escape hatches for the
most-edited fields.

## train.py — CLI reference

```
--config PATH           required — unified-config YAML path
--output-dir DIR        override save.output_dir
--train-data PATH       override data.train_path
--val-data PATH         override data.val_path
--lr FLOAT              override optimizer.learning_rate
--epochs INT            override runtime.num_train_epochs
--max-steps INT         cap training to N optimizer steps (diagnostic short runs)
--seed INT              override runtime.seed
--lora-r INT            override adapter.target_modules to a uniform scalar rank
--lora-alpha INT        override adapter.alpha
--batch-size INT        override batch.per_device_train
--grad-accum INT        override batch.gradient_accumulation
--thinking              set data.enable_thinking=True (apply_chat_template enable_thinking)
--wandb                 set runtime.report_to=['wandb']
--debug-collator        periodically log a decoded training example + labels
--skip-post-train-generate   skip the in-process post-training generation step
--no-grad-checkpoint    disable non-reentrant gradient checkpointing (diagnostic)
```

For overrides outside this set, edit the YAML.

## generate.py — CLI reference

```
--config PATH           required — same training config YAML used for train.py
--checkpoint PATH       required — directory of a saved LoRA adapter
--data PATH             override eval JSONL (default: cfg.data.val_path)
--out PATH              override output JSONL (default: <checkpoint>/<cfg.generation.output_filename>)
--limit N               cap rows for quick smoke runs
```

Output: each input eval row preserved verbatim, plus one added key
`generated_content` holding the model's output. Gold stays in the input row's
`ground_truth` key; a downstream scorer reads gold from `ground_truth` and
prediction from `generated_content`. Sampling parameters (`max_new_tokens`,
`temperature`, `top_p`, `do_sample`, `batch_size`) come from the `generation:`
block of the YAML.

## Determining adapter activation

- `adapter.invocation_tokens: null` (default) → unconditional adapter.
  Every forward pass routes through the LoRA delta.
- `adapter.invocation_tokens: "<some token sequence>"` → gated activation
  (aLoRA-style). PEFT scans each sequence for the token IDs; the LoRA delta is
  zeroed until they appear, then applied for the remainder. The schema
  validator auto-forces `batch.gradient_checkpointing=False` (PEFT #2826);
  `train.py` re-enables non-reentrant checkpointing explicitly for gated SR.

Intrinsics (answerability, citations, etc.) typically use the literal Granite
assistant marker `"<|start_of_role|>assistant<|end_of_role|>"` as the
invocation tokens — the adapter activates exactly at the start of every
assistant response.

## Shadow-residual selection

- Add `"cross_stream"` to `adapter.target_modules` → the SR dual-stream forward
  engages (frozen base stream + trainable adapter stream + per-layer
  cross-stream injection). Rank is the value next to `cross_stream` in the dict.
- Omit `"cross_stream"` → SR takes the single-stream early-exit path (compute
  equivalent to plain LoRA on Granite, modulo unfused-projection numerics).
- `model.shared_base_kv: true` → adapter attends a single base-only K/V (one
  cache); forbids LoRA on `k_proj` / `v_proj`. Leave it unset (the default) to
  auto-resolve: `true` when `"cross_stream"` is in `target_modules` (the SR
  production default), `false` otherwise (plain LoRA/aLoRA → K/V on the adapter
  stream; a warning is logged). An explicit `true`/`false` always wins.

## Per-module rank

`adapter.target_modules` carries both the module names and the LoRA ranks:

```yaml
# Uniform: rank 32 applied to "all-linear"
adapter:
  target_modules: 32

# Per-module: explicit names + ranks (+ cross_stream to enable SR)
adapter:
  target_modules:
    q_proj: 32
    o_proj: 32
    gate_proj: 32
    up_proj: 32
    down_proj: 32
    cross_stream: 32
```

`alpha` defaults to `2 * rank` (scalar) or `2 * max(ranks)` (dict) unless set.

## `enable_thinking`

`data.enable_thinking` (default `false`) is forwarded to
`apply_chat_template(..., enable_thinking=...)` when rendering each training
row. Set it in the YAML or pass `--thinking` on the CLI (CLI wins). Templates
that don't understand the kwarg ignore it, so the default is a no-op.

## Response-only label masking

Loss is computed only on assistant turns. Two paths, picked automatically:

1. **Preferred — `SFTConfig(assistant_only_loss=True)`** when the tokenizer's
   chat template contains `{% generation %}` tags. TRL handles everything.
2. **Fallback — `ResponseOnlyCollator`** (`data.py`) when the template lacks
   the tags. Scans for the assistant-marker token sequence and masks everything
   before the LAST occurrence. Marker defaults to Granite's
   `<|start_of_role|>assistant<|end_of_role|>`; override via
   `data.assistant_marker` for other model families.

## Data format

JSONL only, one chat conversation per row. `tools` / `documents` (both
optional) are passed verbatim to `apply_chat_template`.

**Training row** — the last message MUST be the assistant turn; that turn is
the supervised target.

```json
{"messages": [ ...turns..., {"role": "assistant", "content": "<gold target>"}],
 "documents": [...], "tools": [...]}
```

**Eval / generation row** — same shape, but gold lives in a separate
`ground_truth` key, NOT as a trailing message. Only `messages` is rendered via
`apply_chat_template(..., add_generation_prompt=True)`, so the prompt never
leaks the answer.

```json
{"messages": [ ...prompt turns... ], "ground_truth": "<gold>",
 "documents": [...], "tools": [...]}
```

A 50-row synthetic file lives at `data_samples/tiny_qa.jsonl`.

## What this module reuses

- `shadow_residual.config.load_training_config` — YAML + env-var resolution +
  Pydantic validation.
- `shadow_residual.config.adapters.to_training_arguments` — schema →
  `transformers.TrainingArguments`.
- `shadow_residual.config.adapters.to_peft_config` — schema → `peft.LoraConfig`
  (translates the `target_modules` scalar/dict into PEFT's `r` +
  `target_modules` + `rank_pattern` triple).
- `shadow_residual.config.adapters.build_callbacks` — early stopping + NaN guard.
- `shadow_residual.peft_shadow_residual.get_shadow_residual_peft_model` — builds
  the SR base and PEFT-wraps it.

## Tests

```bash
pytest tests/config/ tests/training/ -v --tb=short
```

The `transformers`/`peft`-dependent cases skip when those packages aren't
installed locally.
