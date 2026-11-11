# Unified training

Two entry points that consume the unified Pydantic-validated YAML schema:

- `train.py` — build an SR + LoRA model and fine-tune it on a base model (CLI + YAML).
- `generate.py` — run a trained adapter against an eval set, write predictions JSONL.

The schema deliberately does not name specific architectures (LoRA,
shadow-residual). The adapter is always active (plain LoRA). Training behavior is
determined by which fields are set — whether `"cross_stream"` appears in
`adapter.target_modules` (turns on the SR dual-stream forward), and the optional
`adapter.last_context_token` / `adapter.last_token` markers that shape the
data/labels (see below).

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
--val-frac F            hold out a seeded fraction F of the train rows as validation (data.val_frac;
                        exclusive with --val-data)
--lr FLOAT              override optimizer.learning_rate
--epochs INT            override runtime.num_train_epochs
--max-steps INT         cap training to N optimizer steps (diagnostic short runs)
--seed INT              override runtime.seed
--lora-r INT            override adapter.target_modules to a uniform scalar rank
--lora-alpha INT        override adapter.alpha
--batch-size INT        override batch.per_device_train
--grad-accum INT        override batch.gradient_accumulation
--max-length INT        override data.max_length (longer rows are truncated, dropping their labels)
--thinking              set data.enable_thinking=True (apply_chat_template enable_thinking)
--tool-formatting [MODE]  opt in to SR-adapter-as-tool training (data.tool_formatting, Granite 4.2);
                        bare flag = external_tool_formatting, or autoregressive_tool_formatting
--tool-task TASK        rows are TASK source rows, converted to tool rows on load (data.tool_task;
                        answerability | guardian_core | requirement_check | query_clarification)
--vanilla-frac F        tool training: the seeded fraction F of the source rows trained as plain
                        vanilla-SR chat rows, the rest as tool rows (data.vanilla_frac; needs
                        --tool-task; default 0.5; 0 = tool rows only)
--wandb                 set runtime.report_to=['wandb']
--debug-collator [PROB] log a decoded training example + labels: the first batch, then ~PROB of
                        batches (default trainer.debug_probability = 0.001)
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
--tool-task TASK        eval rows are TASK source rows; convert on load (overrides data.tool_task)
```

Output: each input eval row preserved verbatim, plus one added key
`generated_content` holding the model's output. Gold stays in the input row's
`ground_truth` key; a downstream scorer reads gold from `ground_truth` and
prediction from `generated_content`. Sampling parameters (`max_new_tokens`,
`temperature`, `top_p`, `do_sample`, `batch_size`) come from the `generation:`
block of the YAML.

## Adapter activation and the token markers

The adapter is **always active** (plain LoRA) — every forward routes through the
LoRA delta on every position. There is no gated/aLoRA activation. Two optional
single-token markers shape the training data/labels (enforced in `train.py`) and
are recorded into the saved `adapter_config.json` for downstream tooling:

- `adapter.last_context_token` — the final context/prompt token. `train.py`
  validates that the supervised region (`labels != -100`) starts immediately
  after it (validate + error) and covers both masking paths (TRL
  `assistant_only_loss` and the `ResponseOnlyCollator` fallback).
- `adapter.last_token` — end-of-completion marker. `train.py` appends it to any
  training row whose tokens don't already end with it (at the text level, so both
  masking paths inherit it).

Each must encode to exactly one token id for the tokenizer. Gradient
checkpointing is honored as set (no force-off); mid-training eval is compatible.

## Shadow-residual selection

- Add `"cross_stream"` to `adapter.target_modules` → the SR dual-stream forward
  engages (frozen base stream + trainable adapter stream + per-layer
  cross-stream injection). Rank is the value next to `cross_stream` in the dict.
- Omit `"cross_stream"` → SR takes the single-stream early-exit path (compute
  equivalent to plain LoRA on Granite, modulo unfused-projection numerics).
- `model.shared_base_kv: true` → adapter attends a single base-only K/V (one
  cache); forbids LoRA on `k_proj` / `v_proj`. Leave it unset (the default) to
  auto-resolve: `true` when `"cross_stream"` is in `target_modules` (the SR
  production default), `false` otherwise (plain LoRA → K/V on the adapter
  stream; a warning is logged). An explicit `true`/`false` always wins.

## Per-module rank

`adapter.target_modules` carries both the module names and the LoRA ranks:

```yaml
# Uniform: rank 32 applied to the SR-safe target set
# (q_proj, o_proj, gate/up/down_proj, cross_stream — never K/V)
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

## `tool_formatting` (SR adapter invoked as a tool — opt-in)

`data.tool_formatting` defaults to unset, which means standard chat SFT with no
tool logic. Setting it (YAML or `--tool-formatting`) trains the adapter to be
called as a tool: it produces the `<tool_response>` payload and stops. It never
writes the assistant answer. **Granite 4.2 only**: any other base is rejected
at startup.

With `data.tool_task` set (`--tool-task`; the reference config sets
`answerability`), the rows are the task's source rows. The trainer converts
them to tool rows on load, so training is one command. Without it, rows must be
pre-converted tool rows: `messages` + `tools` (+ `documents`) ending with the
trigger tool call and the gold `tool` response
(`python -m shadow_residual.training.tool_data --task <task> ...`). The trainer
renders the rows for the mode, so one data file trains either mode:

| mode | adapter takes over after | adapter emits | markers (`assistant_marker` / `last_context_token` / `last_token`) |
|---|---|---|---|
| `external_tool_formatting` *(bare flag)* | env-written `<tool_response>\n` | the payload, then EOS | `<tool_response>\n` / `\n` / `<|im_end|>` |
| `autoregressive_tool_formatting` | base's `</tool_call>` | the whole tool-response turn, ending at its `<|im_end|>` | `</tool_call>` / `</tool_call>` / `<|im_end|>` |

The mode sets the three markers itself. Setting any of them as well is an
error. The mode is recorded in `adapter_config.json`, and generation renders
prompts for it. Reference config: `config/sr-tool-g42.yaml`. Full design:
[`docs/tool_formatting.md`](../../../docs/tool_formatting.md).

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
