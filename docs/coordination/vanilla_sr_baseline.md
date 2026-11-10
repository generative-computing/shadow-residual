# Vanilla SR baseline for the 4 adapter-bench intrinsics — shared results

**To session `8889d7da` (read-out-stream design) and any other session on `feature/different-arch`:
do NOT train the vanilla SR baselines yourself.** Session `3a884bc0` (shared-A design,
`docs/sr_shared_a_design.html`) is training them and will update this file with the scores.

## What is being trained (granite-4.2-3b, Vela, 4 GPUs DDP each)

| field | value |
|---|---|
| base | `/danieloh_cos/antonp_hf_cache/granite-4.2-3b-prerelease-r260622a` |
| config | `src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv-g42.yaml` (q,o,gate,up,down + cross_stream, r=32, α=64, lr 2e-4, 2 epochs, eff. batch 32 = 2×4 accum×4 GPUs, seed 2) |
| data | `/danieloh_cos/granite-switch/sr_shared_a/data_v1/<intrinsic>/{train,val}.jsonl` — the adapter-bench datasets (`adapter-bench/datasets/train/...`) rendered with the bench's own `prompts.with_documents(..., "tool_json_after_question")`, so training prompts are byte-identical to the eval prompts. Script: `scripts/prepare_bench_sr_data.py`. |
| eval | `scripts/bench_eval.py` = the bench's `hf_generate.py` (HF+PEFT, `column=sr`) + bench scorers, harness `granite-switch@2506b7c` copied to `/danieloh_cos/granite-switch/sr_shared_a/bench_harness_2506b7c`; eval sets `adapter-bench/granite-4.2-3b/v1/eval/<intrinsic>/evaluation.jsonl` |
| outputs | `/danieloh_cos/granite-switch/sr_shared_a/runs/v1/<variant>/run_<TS>/{checkpoints,eval/<intrinsic>/score.json}` |

To evaluate your own adapter identically:

```bash
python scripts/bench_eval.py --bench-harness /danieloh_cos/granite-switch/sr_shared_a/bench_harness_2506b7c \
  --base-model /danieloh_cos/antonp_hf_cache/granite-4.2-3b-prerelease-r260622a \
  --adapter <adapter dir> --intrinsic <answerability|guardian_core|query_clarification|requirement_check> \
  --eval-root /danieloh_cos/granite-switch/adapter-bench/granite-4.2-3b/v1/eval --out <dir>
```

(`hf_generate` loads `column=sr` with `build_sr_base` + `PeftModel.from_pretrained`; a read-out-stream
adapter needs that loader to build your architecture.)

## Results (benchmark headline metrics)

**All 4 done (2026-11-10).**

| intrinsic (headline) | vanilla SR | bench's staged SR | run (under `/danieloh_cos/granite-switch/sr_shared_a/runs/v1/`) |
|---|---|---|---|
| answerability (accuracy) | **86.34** (wF1 84.41, unans F1 85.49, ans F1 90.13, n=3565) | 63.39 (older aLoRA-token recipe, not comparable) | `vanilla-sr-qo-mlp-r32-c32-skv-g42-ans/run_20261110T103731Z` (750 steps) |
| guardian_core (accuracy, mean over ood_safety datasets) | **80.24** (f1 76.62, bal. acc 79.74, pooled 82.82, 0 parse failures, n=11347) | 80.20 | `vanilla-sr-qo-mlp-r32-c32-skv-g42-guard/run_20261110T103753Z` (750 steps, train_loss 0.0352) |
| query_clarification (overall_accuracy) | **92.47** (underspecified 82.28 / clear_random 100.0 / clear_hard 86.0, n=279) | 93.19 | `vanilla-sr-qo-mlp-r32-c32-skv-g42-qc/run_20261110T103755Z` (136 steps, train_loss 0.317) |
| requirement_check (balanced_accuracy) | **64.90** (acc 65.09, wF1 65.82, 0 invalid, n=2409) | 65.23 | `vanilla-sr-qo-mlp-r32-c32-skv-g42-req/run_20261110T113720Z` (744 steps, train_loss 0.0793) |

Each run dir has `checkpoints/` (stock SR adapter) and `eval/<intrinsic>/{predictions.jsonl,score.json,job.json}`.
guardian / qc / req reproduce the benchmark's staged SR adapters (same recipe) within noise, so the
pipeline is a faithful baseline. The answerability number is new: the bench had no comparable SR adapter.

### Shared-A result (same data / recipe / eval), for reference

`shareda-sr-qo-mlp-r32-c32-skv-g42-4task/run_20261110T093316Z` — one jointly trained LoRA A shared by
the 4 adapters, per-task B: ans **85.69** (−0.65), guard **79.73** (−0.50, sig.), qc **92.83** (+0.36),
req **63.53** (−1.37, sig.); paired-bootstrap CIs in `docs/sr_shared_a_design.html` §5 and
`/danieloh_cos/granite-switch/sr_shared_a/runs/v1/paired_bootstrap_shareda_vs_vanilla.json`.

Published bench HF+PEFT reference for comparison (run 20261011T193236Z): LoRA r16 84.40 / 81.33 / 92.11 / 68.20;
aLoRA r32 81.80 / 79.89 / 91.76 / 67.78; base 73.83 / 64.33 / 30.11 / 54.76.
