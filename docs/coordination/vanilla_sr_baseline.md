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

| intrinsic (headline) | vanilla SR | status |
|---|---|---|
| answerability (accuracy) | — | training |
| guardian_core (accuracy, mean over ood_safety datasets) | — | training |
| query_clarification (overall_accuracy) | — | training |
| requirement_check (balanced_accuracy) | — | training |

Published bench HF+PEFT reference for comparison (run 20261011T193236Z): LoRA r16 84.40 / 81.33 / 92.11 / 68.20;
aLoRA r32 81.80 / 79.89 / 91.76 / 67.78; base 73.83 / 64.33 / 30.11 / 54.76.
