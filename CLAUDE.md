# CLAUDE.md — working with Shadow Residual

Operational guide for Claude agents in this repo. Read
[`docs/sr_architecture_explainer.md`](docs/sr_architecture_explainer.md) for the
deep dive.

## What SR is

Shadow Residual runs two parallel streams per decoder layer: a **frozen base
stream** (bit-exact with the unadapted base) and a **trainable adapter stream**
(LoRA deltas + a per-layer rank-R `cross_stream` injection from the base). Only
`norm(h_adapt)` reaches the LM head. It's a PEFT adapter using **100% stock**
`peft` LoRA — no custom LoRA classes and no `_register_custom_module`. The
`cross_stream` site is a real frozen zero-init `nn.Linear` that stock
`lora.Linear` wraps like any other target. The adapter is plain LoRA and
**always active** — the delta fires on every position (no gated/aLoRA activation).

**Cross-stream taps are a registry, not one hard-coded site.**
`cross_stream.py::CROSS_STREAM_TAPS` maps a module name to a (base source,
adapter destination) pair — `cross_stream` = post-MLP → post-MLP (the default, and
the legacy name kept for back-compat), `cross_stream_post_attn` = post-attention →
post-attention, plus the two cross-position wirings
`cross_stream_pre_attn_to_post_mlp` and `cross_stream_post_mlp_to_pre_attn`. The
taps *built* on the model are **derived from `target_modules`** (an untargeted tap
is a frozen-zero no-op, so building one is pure dead weight); there is no separate
config field. A tap whose destination *precedes* its source (only
`cross_stream_post_mlp_to_pre_attn` today) makes the decoder select
`_forward_base_ahead` — the base stream's whole layer runs before the adapter's —
decided once at construction so it is identical on every FSDP rank. It is
arithmetically equivalent to the interleaved path but cannot fuse the two
layernorms, so every other topology keeps the interleaved default and reproduces
bit-for-bit. Each tap costs a frozen, permanently-zero `H×H` per layer. For
parameter-aligned ablations `adapter.alpha` accepts a per-module dict (→ PEFT
`alpha_pattern`) over exactly the `target_modules` keys, so `alpha/r` can be held
constant while rank varies. See §1.4b of the explainer.

**Single topology: shared base-only K/V.** K and V are computed once from the
base stream and shared with both streams' Q; there is one KV cache. LoRA on
`k_proj` / `v_proj` is forbidden (rejected at build time — no place for a K/V
delta to land). There is no disjoint-K/V mode and no plain-LoRA-on-SR mode.

## Repo layout

```
src/shadow_residual/
├── shadow_residual/        # the HF model
│   ├── modeling_hf.py          ShadowResidualModel / ForCausalLM; dual-stream vs bare gate
│   ├── decoder_hf.py           GradientCheckpointingLayer; dual-stream forward + forward_bare
│   ├── attention_hf.py         shared-base-K/V attention (+ forward_bare)
│   ├── cross_stream.py         frozen zero-init nn.Linear cross-stream site
│   ├── model_config.py         vendored ShadowResidualConfig
│   ├── config_helpers.py       validate / set_shadow_residual
│   ├── build.py                SR config/model construction + RoPE reinit (no PEFT)
│   ├── weight_transfer.py      fused→unfused QKV / gate-up slicing
│   └── _stream_gated_linear.py   _StreamGatedLinear marker + _base_only helper
│                           #   (100% stock PEFT-adaptable — no custom LoRA classes; the
│                           #    publishable HF model, needs nothing from training/)
├── config/                 # training_config.py schema, adapters.py, reference *.yaml
├── training/               # train.py, generate.py, data.py, collator.py, add_labels.py,
│   │                       #   pack_stats.py, chat_render.py, README.md
│   └── factory.py          get_shadow_residual_peft_model: FSDP meta-init/materialize + PEFT-wrap
└── eval/                   # answerability_eval.py (standalone --mode peft eval)
tests/                      # run with pytest (CPU-only cases pass without a GPU)
examples/vela/              # Vela job YAMLs
```

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[train]"      # add [dev] for pytest
```

## Training

Entry point: `sr-train` (== `python -m shadow_residual.training.train`), YAML-driven.

```bash
# single GPU
sr-train --config src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv.yaml \
         --train-data /data/train.jsonl --val-data /data/eval.jsonl --output-dir ./ckpt

# multi-GPU (FSDP)
accelerate launch --config_file src/shadow_residual/config/accelerate/fsdp_4gpu.yaml \
  --num_processes 8 -m shadow_residual.training.train --config <cfg>.yaml \
  --train-data … --val-data … --output-dir ./ckpt
```

Config essentials:
- **`"cross_stream"` in `adapter.target_modules`** engages the dual-stream SR
  forward. Do NOT list `k_proj` / `v_proj` (K/V LoRA is rejected).
- **`adapter.last_context_token`** (optional, single token) → train.py validates
  that the supervised region (`labels != -100`) starts immediately after it, and
  records it into the saved `adapter_config.json`. Does NOT gate the adapter.
- **`adapter.last_token`** (optional, single token) → end-of-completion marker;
  train.py appends it to any row that doesn't already end with it, and records it
  into `adapter_config.json`.
- **`data.enable_thinking` / `--thinking`** → forwarded to
  `apply_chat_template(enable_thinking=...)`. CLI wins over YAML. Default False.

**How checkpointing works.** The decoder layer is a `GradientCheckpointingLayer`
with a single stacked-tensor (`[2,B,S,H]`) dual-stream forward and a pure
`base_layer(x)` base-stream gate. Under HF gradient checkpointing the dual-stream
forward is pure, so recompute is exact — no manual checkpointing or custom
recompute-stable variant. The adapter is plain LoRA (no aLoRA offset hooks), so
**mid-training eval is compatible** with gradient checkpointing.

**Post-training generation under FSDP — pass `--skip-post-train-generate`.** In an
FSDP run the in-job generation path gathers/rebuilds the model on rank 0 and
generates there while the other ranks wait at a collective barrier. A full eval
set takes longer than the default 30-minute collective timeout, so the idle ranks
abort and the job dies *after* training completes (the checkpoint is already
saved). Autoregressive generation gains nothing from sharding anyway. The pattern:
train with `--skip-post-train-generate`, then run generation/eval as a **separate
single-GPU process** against the saved adapter (no distributed group, no barrier).
Single-GPU / non-FSDP runs are unaffected and can generate in-job.

Vela: use the YAMLs in `examples/vela/`. Pin `trl<1.7` (already in pyproject).
`eval_loss: nan` per epoch is expected for prompt-only eval rows.

## Serving / evaluation

No custom loader — build an SR base, then attach the adapter with stock PEFT:

```python
from shadow_residual.shadow_residual.build import build_sr_base
from peft import PeftModel
base = build_sr_base(base_id, torch_dtype=...)
model = PeftModel.from_pretrained(base, adapter_path)
```

Answerability eval: `python -m shadow_residual.eval.answerability_eval --mode peft
--base-model <id> --adapter <path> …`.

## Invariants — do not break these

1. **Frozen base stream.** The base stream runs on `base_layer` (raw frozen
   weights, no LoRA delta), so it is bit-identical to the unadapted base. Never
   let a delta reach it. `merge_and_unload` is disabled on purpose.
2. **Shared base-only K/V.** One cache, K/V computed once from the base stream;
   K/V LoRA rejected at build time.
3. **Bare model == unadapted base.** With no adapter attached (or under a
   top-level `disable_adapter()`), the model runs the single-stream `forward_bare`
   path and produces base-identical logits.

## Testing

```bash
pytest                    # CPU-only invariant/config/data cases run without a GPU
```

If you touch the model or PEFT wiring, run the invariant tests
(`tests/shadow_residual/`, `tests/peft_shadow_residual/`).

## Version notes

- Vendored `ShadowResidualConfig` uses `model_type = "shadow_residual"`.
- **transformers pinned `>=5.5.1,<5.10.0`** (≥5.10 renames GraniteMoeHybrid
  attention `layer_types` and strict-validates, breaking the config/tests).

## Git protocol

**Never commit and never push.** Stage only (`git add`); the maintainer reviews,
commits, and pushes. **Never sign commits as Claude** (no `Co-Authored-By`,
`Generated with Claude Code`, or similar).
