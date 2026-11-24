# Runbook: train and test an SR tool adapter (Granite 4.2)

This runbook describes how to train an SR adapter for one of the four
adapter-bench intrinsics, check its data, and score it in both formats. It
names every data path. It reflects the runs of 2026-11-11 (branch
`experiments/sr-tool-hybrid`, PR #7).

> **Internal paths.** This file names paths on the `danieloh-cos` PVC. The
> repository is public. Do not commit this file without that decision.

## 1. What the adapter learns

Tool training (`--tool-formatting` with `--tool-task`) renders each source row
in one of two formats. By default, a seeded half of the rows goes to each format
(`data.vanilla_frac`, default 0.5):

| row kind | prompt ends with | adapter writes (answerability example) |
|---|---|---|
| tool, external mode | `...</tool_call>\n<|im_end|>\n<|im_start|>user\n<tool_response>\n` | `{"answerable": false}<|im_end|>` |
| tool, autoregressive mode | `...</tool_call>` | `\n<|im_start|>user\n<tool_response>\n{"answerable": false}\n</tool_response>\n<|im_end|>` |
| vanilla | `...<|im_start|>assistant\n<think></think>` | `"unanswerable"<|im_end|>` |

So one adapter serves as a tool and as a plain SR adapter. The other three
intrinsics write the same JSON dict in both formats. Only the prompt differs:
the tool row moves the `<guardian>` / `<requirements>` instruction into the
tool-call arguments. Design: [`docs/tool_formatting.md`](tool_formatting.md).

## 2. Paths

All paths are on the PVC `danieloh-cos`, mounted at `/danieloh_cos`.
`AB=/danieloh_cos/granite-switch/adapter-bench`.

### 2.1 Base model and code

| what | path |
|---|---|
| base model (all runs below) | `/danieloh_cos/antonp_hf_cache/granite-4.2-3b-prerelease-r260622a` |
| public equivalent (same tokenizer and template) | `ibm-granite/granite-4.2-3b` |
| code | `github.com/generative-computing/shadow-residual`, branch `experiments/sr-tool-hybrid` |
| benchmark harness (scorers, `hf_generate`) | `/danieloh_cos/granite-switch/sr_shared_a/bench_harness_2506b7c` (granite-switch @ 2506b7c) |
| eval tooling (not in the repo) | `/danieloh_cos/granite-switch/adapter_training/shadow_residual_runs/tool_hybrid/tooling/` |

The tooling directory holds:

| script | does |
|---|---|
| `check_hybrid_formats.py` | data check before training (section 4.2) |
| `eval_tool_bench.py` | tool-format eval (section 6.1) |
| `bench_eval.py` | vanilla-format eval through the benchmark's HF+PEFT runner (section 6.2) |
| `strict_rescore.py` | per-row strict rescoring of tool-format dumps (section 6.3) |
| `format_zoom.py` | combined score and parsing failures for all cells (section 6.3) |
| `gen_vela_yamls.py` | generator of the Vela job YAMLs used for these runs (section 7) |

### 2.2 Training data (raw source rows)

Pass the raw rows directly. `--tool-task` converts them on load.

| intrinsic | `--tool-task` | train / val | path |
|---|---|---|---|
| answerability | `answerability` | 11,979 / 1,351 | `$AB/datasets/train/answerability/g42-alora-4.1-split/{train,val}.jsonl` |
| guardian_core | `guardian_core` | 12,000 / 577 | `$AB/datasets/train/guardian_core/guardian-ood-12k/{train,val}.jsonl` |
| query_clarification | `query_clarification` | 2,153 / 240 | `$AB/datasets/train/query_clarification/unified-compare-data-v4/{train,val}.jsonl` |
| requirement_check | `requirement_check` | 11,882 / 500 | `$AB/datasets/train/requirement_check/core-library-train-v2/{train,val}.jsonl` |

Row shapes:

- **answerability:** `messages` only. The documents are inline `tool` turns
  before the last user turn. The converter moves them into `documents`. Labels:
  6,062 answerable / 5,917 unanswerable.
- **guardian_core:** `messages` + `ground_truth`. The last user turn is the
  `<guardian>` instruction. No documents.
- **query_clarification:** `messages` + `documents` (1,593 rows have documents).
- **requirement_check:** `messages` only. The last user turn is the
  `<requirements>` instruction. No documents.

Reference data (used only by the format check):

| what | path |
|---|---|
| vanilla-baseline training rows (benchmark renderer) | `/danieloh_cos/granite-switch/sr_shared_a/data_v1/<intrinsic>/{train,val}.jsonl` |
| answerability rows with a `documents` key (earlier tool runs) | `/danieloh_cos/granite-switch/granite-intrinsics-training-data/output/unified_compare/data_v4/{train,val}.jsonl` |

### 2.3 Eval sets

`$AB/granite-4.2-3b/v1/eval/<intrinsic>/evaluation.jsonl`. This is the only
bench eval version on the PVC (no `bench/<model>/v<N>/eval/` tree yet).

| intrinsic | rows | headline metric | benchmark scorer |
|---|---|---|---|
| answerability | 3,565 | accuracy | `answerability` |
| guardian_core | 11,347 | accuracy (mean over the ood_safety datasets) | `guardian` |
| query_clarification | 279 | overall_accuracy | `query_clarification` |
| requirement_check | 2,409 | balanced_accuracy | `requirements` |

Do not score on the training `val.jsonl`. In vanilla format its prompt contains
the gold assistant turn.

### 2.4 Outputs

New runs: `/danieloh_cos/granite-switch/adapter_training/shadow_residual_runs/tool_hybrid/<intrinsic>/<mode>/run_<UTC ts>/`

| file | content |
|---|---|
| `checkpoints/` | the adapter (`adapter_model.safetensors`, `adapter_config.json` with `tool_formatting`, `vanilla_frac`, `last_context_token`, `vanilla_last_context_token`) |
| `format_check_<intrinsic>_<mode>.{json,log}` | data check result |
| `eval_tool_full.json`, `predictions_tool_full.jsonl`, `eval_tool_full.log` | tool-format eval |
| `eval_vanilla_full/{score.json,predictions.jsonl,status.json,job.json}`, `eval_vanilla_full.log` | vanilla-format eval |

Do not write runs into `$AB/training/`. That tree is the benchmark's managed
store (`launch.py cos`).

## 3. Environment

Vela: namespace `security`, image `us.icr.io/cil15-shared-registry/sec-rl-neo:latest`,
secrets `antonp-hf-token` (key `token`) and `antonp-ghp-token` (key `token`).
4 GPUs, 8 CPUs, 400Gi per training pod.

Install, inside the pod:

```sh
git clone https://x-access-token:${GHP_PERSONAL_TOKEN}@github.com/generative-computing/shadow-residual.git
cd shadow-residual && git checkout experiments/sr-tool-hybrid
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -e ".[train]"
pip install "trl>=1.0,<1.7"
pip uninstall -y torchvision || true
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0"
```

Pin a commit with `git checkout <sha>`, not a branch, for a reproducible run.

## 4. Before training

### 4.1 Unit tests (CPU, local or in the pod)

```sh
SR_REQUIRE_TOKENIZERS=1 pytest -q -rs tests/
```

Expected: `316 passed`, 0 skipped. `SR_REQUIRE_TOKENIZERS=1` turns every
tokenizer-dependent skip into a failure. It needs the `ibm-granite/granite-4.2-3b`,
`ibm-granite/granite-4.1-3b` and `hf-internal-testing/tiny-random-LlamaForCausalLM`
tokenizers in the HF cache.

### 4.2 Data format check (CPU, in the pod)

The check compares the new rows with the old code and with the vanilla
baseline data, on all rows. It needs the pre-change code as a reference:

```sh
mkdir -p /workspace/ref && git archive 8d19557 src | tar -x -C /workspace/ref
T=/danieloh_cos/granite-switch/adapter_training/shadow_residual_runs/tool_hybrid/tooling
python $T/check_hybrid_formats.py --task guardian_core --mode external_tool_formatting \
  --base-model $BASE \
  --src $AB/datasets/train/guardian_core/guardian-ood-12k/train.jsonl \
  --tool-ref $AB/datasets/train/guardian_core/guardian-ood-12k/train.jsonl \
  --vanilla-ref /danieloh_cos/granite-switch/sr_shared_a/data_v1/guardian_core/train.jsonl \
  --eval $AB/granite-4.2-3b/v1/eval/guardian_core/evaluation.jsonl \
  --harness /danieloh_cos/granite-switch/sr_shared_a/bench_harness_2506b7c \
  --ref-src /workspace/ref/src --max-length 12288 \
  --work /workspace/fc --out $RUN_DIR/format_check.json
```

For answerability, set `--tool-ref` to the `data_v4/train.jsonl` above.

The script prints `FORMAT_CHECK {...}`. Gate training on `"ok": true`. It checks:

- tool rows == old-code tool rows (`tool_exact`; for answerability
  `tool_docid_norm`, after normalizing `document_id` and the empty-document
  encoding);
- vanilla rows == the vanilla-baseline rows (`vanilla_exact`);
- the supervised span is the expected gold on every row (`span_ok`);
- benchmark eval prompts == our vanilla rendering (`eval_prompt_bench_eq_ours`).

Results of 2026-11-11 (both modes the same):

| intrinsic | rows | tool rows | vanilla rows | spans | longest row (tokens) |
|---|---|---|---|---|---|
| answerability | 11,979 | 5989/5989 (normalized) | 5990/5990 | 11979/11979 | 11,179 |
| guardian_core | 12,000 | 6000/6000 | 6000/6000 | 12000/12000 | 1,246 |
| query_clarification | 2,153 | 1077/1077 | 1076/1076 | 2153/2153 | 11,326 |
| requirement_check | 11,882 | 5941/5941 | 5941/5941 | 11882/11882 | 4,304 |

### 4.3 Smoke run (optional, recommended after code changes)

Add `--max-steps 40 --debug-collator 0.03` to the training command (section 5).
In the log, check:

- `hybrid split` / `split (vanilla_frac=0.5, seed=2): N ... tool rows + M vanilla rows`;
- `Validating the tool rows' boundary` and `Validating the vanilla rows'
  boundary`, each followed by `validated on 8/8 sampled rows`, on every rank;
- debug prints of both kinds: `Last context token: '</think>'` with
  `Supervised text: '"answerable"<|im_end|>\n<|im_end|>'`, and
  `Last context token: '\n'` with `Supervised text: '{"answerable": true}<|im_end|>'`.

## 5. Training

One command per intrinsic and mode. 4 GPUs, DDP:

```sh
accelerate launch --num_machines 1 --main_process_port 0 --num_processes 4 --multi_gpu \
  --mixed_precision bf16 -m shadow_residual.training.train \
  --config src/shadow_residual/config/sr-tool-g42.yaml \
  --tool-formatting external_tool_formatting --tool-task guardian_core \
  --base-model /danieloh_cos/antonp_hf_cache/granite-4.2-3b-prerelease-r260622a \
  --train-data $AB/datasets/train/guardian_core/guardian-ood-12k/train.jsonl \
  --val-data   $AB/datasets/train/guardian_core/guardian-ood-12k/val.jsonl \
  --run-dir $RUN_DIR --eval-data /unused \
  --max-length 12288 --epochs 2 --skip-post-train-generate --debug-collator 0.001
rc_train=$?
```

- Use `--tool-formatting autoregressive_tool_formatting` for the other mode.
- `--vanilla-frac 0` trains tool rows only. The default is 0.5.
- `--run-dir` writes the adapter to `$RUN_DIR/checkpoints`. `--eval-data` only
  fills the config's placeholder, because `--skip-post-train-generate` skips
  in-job generation. Evaluate as separate processes (section 6).
- Recipe (`sr-tool-g42.yaml`): SR r32 on q/o/gate/up/down + `cross_stream`,
  alpha 64, lr 2e-4 cosine, 2 epochs, effective batch 32 (1 × 8 accumulation ×
  4 GPUs), seed 2.
- Tool rows of at least `max_length` tokens are dropped and logged. At 12288 no
  row of the four datasets is dropped. At 4096 a large share of the
  answerability rows would be dropped.
- Wall time on 4 GPUs: guardian_core about 75 min (750 steps, ~6 s/step).
  Answerability about 2.7 h (~13 s/step) plus ~16 min of data loading per rank.
- End the Vela command list with `exit ${rc_train}`. Otherwise a crashed run
  shows `Succeeded`.
- Check after training: `grep -E "vanilla_frac|tool_formatting" $RUN_DIR/checkpoints/adapter_config.json`.

## 6. Evaluation

Run both formats as separate single-GPU processes on the saved adapter. They can
run at the same time on different GPUs.

### 6.1 Tool format

```sh
CUDA_VISIBLE_DEVICES=0 python $T/eval_tool_bench.py --intrinsic guardian_core \
  --mode external_tool_formatting --base-model $BASE --adapter $RUN_DIR/checkpoints \
  --eval $AB/granite-4.2-3b/v1/eval/guardian_core/evaluation.jsonl \
  --harness /danieloh_cos/granite-switch/sr_shared_a/bench_harness_2506b7c \
  --batch-size 16 --max-new-tokens 32 \
  --out $RUN_DIR/eval_tool_full.json --dump $RUN_DIR/predictions_tool_full.jsonl
```

- The script converts each eval row to a tool row and renders it up to the
  mode's handoff point. It generates greedily in bf16 until EOS and extracts the
  payload. The benchmark's scorer then scores it.
- It prints `EVAL_RESULT {...}` with `headline_value`, `structural_rate`
  (well formed) and `eos_stop_rate`.
- `--max-new-tokens`: 32 for answerability, guardian_core and
  requirement_check; **use 160 for query_clarification**. The runs of
  2026-11-11 used 96. That cut off 4–6 of 279 long clarification lists, while
  the vanilla eval allows 160.
- Answerability scores 3,545 of 3,565 rows. The 20 `conversational` rows have no
  tool label.
- The tool call is given in the prompt, so this eval does not test whether the
  base decides to call the tool.

### 6.2 Vanilla format

```sh
CUDA_VISIBLE_DEVICES=1 python $T/bench_eval.py \
  --bench-harness /danieloh_cos/granite-switch/sr_shared_a/bench_harness_2506b7c \
  --base-model $BASE --adapter $RUN_DIR/checkpoints --intrinsic guardian_core \
  --eval-root $AB/granite-4.2-3b/v1/eval --out $RUN_DIR/eval_vanilla_full
```

- This is the benchmark's own HF+PEFT runner (`hf_generate`, column `sr`).
  Documents are placed as `tool_json_after_question`, with thinking off.
- `max_new_tokens` is fixed per intrinsic: answerability 200, guardian_core 24,
  query_clarification 160, requirement_check 200.
- It prints `SCORE <intrinsic> <headline>=...` and writes `score.json`.

### 6.3 Combined score and parsing failures

The benchmark scorers for guardian and query_clarification parse leniently. A
malformed output can still score as correct. Report the **combined score**: an
output counts as correct only if it is well formed (tool format: it also stops
on EOS) and the scorer marks its content correct.

- `strict_rescore.py <harness> <runs root>`: per-run plain vs strict accuracy,
  malformed rows, missing EOS stops.
- `format_zoom.py <harness>`: combined score and parsing-failure counts for the
  new adapters and the baselines.

Both scripts glob the run directories of 2026-11-11
(`run_20261110T2351*`). Edit the glob for new runs.

Well formed means:
- **Tool, external:** the whole output up to EOS is exactly the task's JSON dict.
- **Tool, autoregressive:** the output is exactly
  `\n<|im_start|>user\n<tool_response>\n{dict}\n</tool_response>\n<|im_end|>`.
- **Vanilla:** the output is exactly `"answerable"` / `"unanswerable"`, or the
  one-key dict.

## 7. Vela jobs

`$T/gen_vela_yamls.py <commit>` writes the job values files (one smoke job and
eight full jobs) for the `mlbatch/pytorchjob-generator` chart. Submit:

```sh
helm template -f <job>.yaml --set-string "environmentVariables[0].value=$(date '+%Y-%m-%d_%H.%M')" \
  mlbatch/pytorchjob-generator | oc create -f -
```

Each full job runs, in order: clone and install, the format check (gate), the
training, a check of `adapter_config.json`, then both evals on GPU 0 and GPU 1.
The generator predates the `--base-model` / `--data-root` / `--eval-data` /
`--run-dir` flags. It sets `BASE_MODEL`, `DATA_ROOT`, `EVAL_DATA` and `RUN_DIR`
as environment variables, which still work.

Rules:

- **Commands run under `/bin/sh` (dash).** Do not use bash-only syntax such as
  `PIPESTATUS`.
- **Exit code:** training jobs end with `exit ${rc_train}`.
- **Check the artifacts,** not the AppWrapper status.
- **False alarms:** debug-collator prints can contain the text `Error:` from the
  training rows (C# code in requirement_check). A failed `oc get` poll can show
  a job as gone for one poll.

## 8. Reference results (2026-11-11)

Granite 4.2-3b prerelease, `max_length` 12288, bench v1 eval sets, combined score.

| intrinsic | ext · tool | ext · vanilla | ar · tool | ar · vanilla | tool-only ext | tool-only ar | vanilla-only |
|---|---|---|---|---|---|---|---|
| answerability | 0.8426 | 0.8412 | 0.8403 | 0.8337 | 0.8432 | n/a (older span) | 0.8634 |
| guardian_core | 0.7987 | 0.8011 | 0.8016 | 0.8043 | 0.7996 | 0.8009 | 0.8024 |
| query_clarification | 0.9176 | 0.9319 | 0.9104 | 0.9176 | 0.8996 | 0.9140 | 0.9211 |
| requirement_check | 0.6653 | 0.6640 | 0.6519 | 0.6530 | 0.6862 | 0.6690 | 0.6490 |

Parsing failures: 0 everywhere, except query_clarification tool format (4 / 279
ext, 6 / 279 ar, 4 / 279 both tool-only) and 1 / 279 vanilla-only.

Adapters (`checkpoints/` under each run directory):

| adapter | path |
|---|---|
| new, `<intrinsic>`, `<mode>` | `.../shadow_residual_runs/tool_hybrid/<intrinsic>/<mode>/run_20261110T2351*/checkpoints` |
| tool-only, guardian / qc / req | `.../shadow_residual_runs/tool_bench/<intrinsic>/<mode>/run_<ts>/checkpoints` (ar requirement_check: `run_20261110T173516Z`) |
| tool-only answerability, external | `.../shadow_residual_runs/tooladapter_external_42_3b/sr-tooladapter-ext-ans-42-3b/run_20261104T080120Z/checkpoints` |
| tool-only answerability, autoregressive | `.../shadow_residual_runs/tooladapter_42_3b/sr-tooladapter-ans-42-3b/run_20261103T220821Z/checkpoints` |
| vanilla-only | `/danieloh_cos/granite-switch/sr_shared_a/runs/v1/vanilla-sr-qo-mlp-r32-c32-skv-g42-{ans,guard,qc,req}/run_<ts>/checkpoints` |

`...` = `/danieloh_cos/granite-switch/adapter_training`.

Notes for comparisons:

- The answerability baselines trained at `max_length` 4096, which truncated
  many rows. The new runs used 12288.
- The tool-only answerability adapter trained on `data_v4`. That is the same
  rows as `g42-alora-4.1-split`, with documents in a `documents` key.
- In vanilla format, the answerability adapter is 2.2–3.0 pt below the
  vanilla-only adapter. It refuses more answerable questions (233 vs 160 of
  2,045). The cause is not confirmed.
