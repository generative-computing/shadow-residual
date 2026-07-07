# CLAUDE.md — working with Shadow Residual

Guidance for Claude agents operating in this repo. It covers what SR is, how the
code is laid out, and the exact flows for **training** and **serving** SR
adapters. Read [`docs/sr_architecture_explainer.md`](docs/sr_architecture_explainer.md)
for the deep dive; this file is the operational summary.

## What SR is (in one paragraph)

Shadow Residual runs two parallel streams per decoder layer: a **frozen base
stream** (bit-exact with the unadapted base — Q/K/V/O/MLP are the base weights,
no delta) and a **trainable adapter stream** (LoRA deltas + a per-layer rank-R
`cross_stream` injection from the base). Only `norm(h_adapt)` reaches the LM
head. It's a PEFT adapter under the hood; SR just needs two custom PEFT module
types and a dedicated loader.

## Repo layout

```
src/shadow_residual/
├── shadow_residual/        # the HF model
│   ├── modeling_hf.py          ShadowResidualModel / ShadowResidualForCausalLM
│   ├── decoder_hf.py           dual-stream + single-stream early-exit
│   ├── attention_hf.py         disjoint vs shared-base K/V
│   ├── cross_stream.py         no-op site PEFT wraps into CrossStreamLora
│   ├── model_config.py         VENDORED ShadowResidualConfig (GraniteMoeHybridConfig subclass)
│   ├── config_helpers.py       validate_shadow_residual_config / set_shadow_residual
│   └── weight_transfer.py      fused→unfused QKV / gate-up slicing
├── peft_shadow_residual/   # PEFT integration
│   ├── factory.py              get_shadow_residual_peft_model + _build_sr_base (TRAINING)
│   ├── load.py                 load_shadow_residual_peft_model (SERVING)
│   ├── cross_stream_lora.py    CrossStreamLora (rank-R B·A, no base contribution)
│   └── stream_gated_lora.py    ShadowResidualLora (LoRA gated off in base stream)
├── config/                 # unified training-config schema + reference YAMLs
│   ├── training_config.py      Pydantic schema; DataConfig.enable_thinking lives here
│   ├── adapters.py             to_peft_config / to_training_arguments / build_callbacks
│   └── *.yaml                  lora / alora / sr reference configs (+ schema/, accelerate/)
├── training/               # train.py, generate.py, data.py, collator.py, add_labels.py, pack_stats.py
└── eval/answerability_eval.py  # peft mode = standalone; switch/vllm need granite-switch
tests/                      # mirrors src; run with pytest (CPU-only cases pass without a GPU)
examples/vela/              # Vela job YAMLs (fill placeholders before submitting)
```

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[train]"      # add [dev] for pytest; [eval-switch]/[eval-vllm] for optional eval modes
```

Fully self-contained — no `granite-switch` dependency on the train/serve path.

## Training

Entry point: `sr-train` (== `python -m shadow_residual.training.train`), YAML-driven.

```bash
# local, single GPU
sr-train --config src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv.yaml \
         --train-data /data/train.jsonl --val-data /data/eval.jsonl --output-dir ./ckpt

# multi-GPU (DDP)
accelerate launch --num_processes 4 --multi_gpu --mixed_precision bf16 \
  -m shadow_residual.training.train --config <cfg>.yaml \
  --train-data … --val-data … --output-dir ./ckpt
```

What determines behavior (no architecture name — it's all fields):
- **`"cross_stream"` in `adapter.target_modules`** → dual-stream SR forward
  engages. Omit it → single-stream early-exit (≈ plain LoRA on Granite).
- **`adapter.invocation_tokens` set** → aLoRA-style gated activation (delta off
  until the token sequence appears). Requires `task_type="CAUSAL_LM"` (the schema
  sets it). Gated runs force HF gradient_checkpointing off (PEFT #2826); train.py
  re-enables non-reentrant checkpointing explicitly for gated SR.
- **`model.shared_base_kv: true`** → one base-only K/V cache shared with the
  adapter's Q; **forbids LoRA on k_proj/v_proj** (rejected at build time).
- **`data.enable_thinking` / `--thinking`** → forwarded to
  `apply_chat_template(enable_thinking=...)`. CLI wins over YAML. Default False.

Vela: use the YAMLs in `examples/vela/`. They clone this repo in-pod, `pip install -e
".[train]"`, then `accelerate launch` the trainer against COS-mounted data.
**Pin `trl<1.7`** (already in pyproject) — TRL 1.7 crashes on Granite configs
(`config.num_experts`). The Vela image ships torch 2.6 + CUDA 12.4. `eval_loss:
nan` per epoch is expected for prompt-only eval rows. Placeholders (`<changeme>`,
`<YOUR_GIT_HOST>`) MUST be filled (or use a k8s Secret) before submitting.

## Serving / evaluation

Load through the SR loader — **not** `PeftModel.from_pretrained` directly, which
can't re-register SR's custom modules:

```python
from shadow_residual.peft_shadow_residual import load_shadow_residual_peft_model
model = load_shadow_residual_peft_model(base_id, adapter_path,
                                        torch_dtype=..., shared_base_kv=<same as training>)
```

`shared_base_kv` is not serialized in `adapter_config.json` — the caller must
pass the value used at training time.

Answerability eval: `python -m shadow_residual.eval.answerability_eval --mode peft
--base-model <id> --adapter <path> …`. `--mode peft` is standalone; `--mode
switch` / `--mode vllm` raise a clear ImportError unless `granite-switch` is
installed (`[eval-switch]` / `[eval-vllm]`).

## Invariants — do not break these

1. **Frozen base stream.** Never let a LoRA delta reach a `stream_context("base")`
   call. `merge_and_unload` is disabled on purpose (it would fold the delta into
   the base linear).
2. **Disjoint vs shared K/V.** Disjoint = two caches; shared = one base-only
   cache, no K/V LoRA. The `test_dual_kv_cache_invariant.py` tests pin this.
3. **ALORA three-knob rule.** `alora_invocation_tokens` + `task_type="CAUSAL_LM"`
   + `"cross_stream"` must all be set for gated SR; missing any one silently
   degrades to unconditional LoRA. `test_alora_invocation_invariant.py` pins the
   pre-invocation logits/KV equality.

## Testing

```bash
pytest                    # 134 tests; CPU-only invariant/config/data cases run without a GPU
pytest -m deep            # expensive code-theory tests (opt-in)
```

If you touch the model or PEFT wiring, run the invariant tests
(`tests/shadow_residual/`, `tests/peft_shadow_residual/`) — they're fast and
catch silent breakage of the frozen-base / KV / ALORA guarantees.

## Version notes

- Vendored `ShadowResidualConfig` keeps `model_type = "granite_switch"` for
  checkpoint round-trips.
- **transformers is pinned `>=5.5.1,<5.10.0`** to match the granite-switch-internal
  environment. On this range, `GraniteMoeHybridConfig` keeps attention
  `layer_types` as `"attention"` — which `validate_shadow_residual_config` and the
  test fixtures rely on. transformers ≥5.10 renames it to `"full_attention"` and
  strict-validates, which breaks the config and tests. Do not bump the upper
  bound without re-running the invariant tests and adapting the `layer_types`
  handling.

## Git protocol in this repo

**Never commit and never push.** Stage only (`git add`) — the maintainer reviews
staged changes, then commits and pushes themselves. Do not run `git commit` or
`git push` under any circumstances, even if asked in passing; if a commit seems
warranted, stage the files and say so instead.

**Never sign commits as Claude.** Do not add `Co-Authored-By: Claude`,
`Generated with Claude Code`, or any similar attribution/trailer. Since you must
not commit at all, this also means never preparing commit messages that carry
such a signature.
