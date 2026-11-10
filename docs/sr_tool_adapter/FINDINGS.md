# SR answerability: tool-adapter vs. plain — findings

Two SR answerability adapters trained and evaluated on the same base, data, and
scale, to compare the **adapter-as-tool** flow against the **plain** (label-only)
flow. Runs done locally on Apple MPS (2026-10-19/20), branch
`experiments/sr-tool-adapter`.

## Setup (identical across both runs)

- **Base:** `ibm-research/granite-4.2-3b-prerelease-r260622a` (Granite 4.2-3b,
  `model_type=granite`, `tie_word_embeddings=False`).
- **Data source:** `granite-switch-internal/scratch/answerability_evaluation.jsonl`
  (3565 rows: `messages` + `documents` + `ground_truth`). The **`ground_truth`
  field is the gold answerability verdict** and the sole supervised signal — the
  dataset has no gold *prose* answer, only the verdict label.
- **Scale:** 120 balanced train rows (60 answerable / 60 unanswerable), 60
  balanced val rows, 1 epoch, 15 optimizer steps, ~32–37 min each. SR r32 on
  q/o/MLP + cross_stream, thinking OFF.
- **Eval:** free-generation (stock `generate()`, greedy) over the 60 val rows.

> Scale caveat: 120 rows / 1 epoch is a **flow-validation** scale, not a quality
> scale. Absolute numbers are indicative, not final. The prior full-scale
> answerability run (11,979 rows, Vela GPU) reached ~0.89.

## Results

| metric | plain SR | tool-adapter SR |
|---|---|---|
| free-gen verdict accuracy | **0.80** | 0.58 (0.66 of structurally-valid) |
| structural validity | 1.00 | 0.88 |
| held-out `eval_loss` (teacher-forced) | 0.071 | 0.103 |
| held-out `eval_mean_token_accuracy` | 0.966 | 0.961 |
| class balance | balanced errors | biased → unanswerable |

### Plain SR confusion (60 val)

|  | →answerable | →unanswerable |
|---|---|---|
| **answerable** | 22 | 8 |
| **unanswerable** | 4 | 26 |

Balanced error profile (8 false-refusals, 4 false-accepts).

### Tool-adapter SR confusion (53 structurally-valid)

|  | →answerable | →unanswerable |
|---|---|---|
| **answerable** | 10 | 18 |
| **unanswerable** | 0 | 25 |

Strong bias toward `unanswerable` (nails unanswerable, over-refuses answerable);
7/60 generations were structurally malformed (collapsed the two-part
tool_response + answer into one turn).

## What each flow learns

- **Plain:** reads question + retrieved docs (docs injected as a `tool` message),
  answers directly with the verdict label `"answerable"`/`"unanswerable"`. The
  completion is one word, so the model fits the format fast and spends its
  capacity on the *decision*. Standard config `sr-qo-mlp-r32-c32-sharedkv-g42.yaml`
  (marker `<|im_start|>assistant\n<think></think>`, `last_context_token=</think>`).
- **Tool-adapter:** the base emits a `<tool_call>` for the `answerability` tool;
  the switch routes to the SR adapter, which generates the `<tool_response>`
  verdict **and** the answer turn. It must learn the decision *and* the structural
  scaffolding. Config `sr-tool-adapter-answerability-g42.yaml`
  (marker `</tool_call>`, `last_context_token=</tool_call>`). See
  [`SR_TOOL_ADAPTER_BOOTSTRAP.md`](SR_TOOL_ADAPTER_BOOTSTRAP.md).

## Key takeaways

1. **At this scale the plain flow is both more accurate (0.80 vs 0.66) and more
   robust (no structural failures, no class bias).** Expected: it only learns a
   1-word decision; the tool-adapter must also learn the tool_response/answer
   scaffolding, which dilutes the decision signal and occasionally breaks.
2. **The adapter-as-tool capability is proven to work** — a correctly-trained
   SR adapter reliably generates the full tool_response + answer at inference
   under stock `generate()`. The 12% structural-failure rate and the class bias
   are scale artifacts (120 rows, 1 epoch), not pipeline defects.
3. **The single-`<|im_end|>` design is load-bearing.** An earlier tool-adapter
   attempt trained on the fully-stock ChatML stream (which closes the
   tool_response turn with an intermediate `<|im_end|>`) produced an adapter that
   emitted `\n<|im_end|>` and halted immediately at inference — `<|im_end|>` is the
   stop token. Dropping the intermediate `<|im_end|>` so the adapter emits exactly
   one terminal `<|im_end|>` took structural validity from 0 → 1.0 (then 0.88 once
   the completion shrank to the bare verdict label). See the design doc.
4. **`ground_truth` is the correct gold.** A first tool-adapter run that paired
   the verdict with a *synthesized prose answer* scored ~chance (0.52) on the
   verdict — the synthetic answer was noise. Switching the completion to the gold
   verdict label (`"answerable"`/`"unanswerable"`) lifted verdict accuracy to 0.66.

## Full-scale autoregressive Vela run (2026-11-04, job `antonp-sr-tooladapter-42-3b`, commit `dc837c8`)

The autoregressive mode (adapter emits the `<tool_response>` block + the next
assistant-turn opener, then stops at `<|im_start|>assistant\n`) re-run at full
scale, so the flow-check numbers above can be read against a real quality scale.
Granite 4.2-3b prerelease, SR r32 q/o/MLP + cross_stream, 4-GPU DDP, 2 epochs (750
steps). Data generated in-pod from `unified_compare/data_v4` (11,978 train / 1,350
val); verdict derived from the final assistant turn. `--thinking off`. Boundary
validated 8/8 per rank.

The in-job eval hung (the known FSDP/accelerate-barrier trap — generation must run
as a separate single-GPU process, per CLAUDE.md); re-run standalone
(`antonp-sr-eval-both-42-3b`) over a **500-row sample** of the val set.

| metric | 120-row flow-check | full-scale (500-row sample) |
|---|---|---|
| free-gen verdict accuracy | 0.58 (0.66 of valid) | **0.95** |
| structural validity | 0.88 | **1.00** (500/500) |
| consistency rate | 0.93 | **1.00** |
| held-out `eval_loss`, ep1 → ep2 | — | 0.00596 → 0.00516 |
| held-out `eval_mean_token_accuracy`, ep2 | — | 0.998 |

|  | →answerable | →unanswerable |
|---|---|---|
| **answerable** (264) | 249 | 15 |
| **unanswerable** (236) | 10 | 226 |

Balanced error profile (15 false-refusals, 10 false-accepts) — the flow-check's
heavy unanswerable bias is **gone** at scale, and structural validity is perfect
(every generation produced a parseable `<tool_response>` + the assistant opener).
This is the like-for-like counterpart to the external-mode run below (same
`data_v4` split, same `eval_tool_adapter.py`): autoregressive **0.95** vs external
**0.9541** — effectively tied, so emitting the full scaffolding (autoregressive)
costs essentially nothing in accuracy versus the minimal external payload.

### Plain-flow baseline on the same split (2026-11-10)

The plain (vanilla) answerability adapter — reads question + docs, emits the
one-word verdict, no tool framing — evaluated on the same `data_v4` val set (full
1,350 rows, `eval_plain_answerability.py`). (The first standalone attempt died on a
transient `CUDA: unspecified launch failure` after 1 row; the retry on a fresh GPU
ran clean.)

| metric | plain | tool autoregressive | tool external |
|---|---|---|---|
| verdict accuracy | **0.9444** (1,350) | **0.95** (500 sample) | **0.9541** (1,350) |

|  | →answerable | →unanswerable |
|---|---|---|
| **answerable** (665) | 626 | 39 |
| **unanswerable** (685) | 36 | 649 |

Balanced (39 false-refusals, 36 false-accepts). **Headline: all three flows land at
~0.94–0.95 — wrapping the answerability capability in the tool protocol costs
essentially nothing versus the plain adapter.** The tool framing buys native
tool-calling ergonomics for free.

Artifacts: `/danieloh_cos/granite-switch/adapter_training/shadow_residual_runs/tooladapter_42_3b/sr-tooladapter-ans-42-3b/run_20261103T220821Z/`
(`checkpoints/` adapter, `eval.json`, `eval_dump.jsonl` — 500 rows).

## External tool formatting (`external_tool_formatting`)

A second mode in which the environment owns the template: it writes
`<tool_response>\n`, the adapter emits only the verdict dict and stops on EOS, and
the environment appends `\n</tool_response>\n`. The supervised span is just
`{"answerable": <bool>}<|im_end|>`, so the adapter learns the decision with no
structural scaffolding. In principle this is the plain flow's advantage, kept
inside the tool protocol. See `SR_TOOL_ADAPTER_BOOTSTRAP.md` → *Tool formatting
modes*.

### Vela run (2026-11-04, job `antonp-sr-tooladapter-ext-42-3b`, commit `b29a377`)

Granite 4.2-3b prerelease, SR r32 q/o/MLP + cross_stream, 4-GPU DDP, 2 epochs
(750 optimizer steps). Data was generated in-pod from `unified_compare/data_v4`
(11,978 train / 1,350 val), the same source as the autoregressive Vela run.
Boundary validated on 8/8 rows per rank.

| metric | value |
|---|---|
| free-gen verdict accuracy (1,350 val) | **0.9541** |
| structural validity (generation parses as the dict) | **1.00** |
| EOS stop rate (ended on `<|im_end|>`, not max tokens) | **1.00** |
| held-out `eval_loss`, epoch 1 → 2 | 0.0191 → 0.0167 |
| held-out `eval_mean_token_accuracy`, epoch 1 → 2 | 0.9927 → 0.9936 |

|  | →answerable | →unanswerable |
|---|---|---|
| **answerable** (665) | 630 | 35 |
| **unanswerable** (685) | 27 | 658 |

Balanced error profile: 35 false-refusals, 27 false-accepts. Every generation was
exactly the dict followed by EOS, so the environment-side contract (emit the
payload, stop, let the env close the tag) holds with no structural failures.

> **Comparability caveat.** This val set is the `data_v4` held-out split
> (in-distribution with training), scored by `eval_tool_adapter.py`. It is **not**
> the canonical answerability eval (`evaluation_data/answerability/`, scored by
> `answerability_eval`) behind the ~0.89 numbers. The like-for-like comparison is
> the autoregressive Vela run on the same `data_v4` split.

Artifacts: `/danieloh_cos/granite-switch/adapter_training/shadow_residual_runs/tooladapter_external_42_3b/sr-tooladapter-ext-ans-42-3b/run_20261104T080120Z/`
(`checkpoints/` adapter, `eval.json`, `eval_dump.jsonl`).

## Next levers (not pipeline changes)

- **Scale:** full 3191-row set, multiple epochs, on GPU — both flows should
  approach the ~0.89 band.
- **Tool-adapter structure robustness:** more rows reinforce the two-part
  structure; alternatively constrain decoding or add a structure-repair step.
- **Class bias:** larger/again-balanced data and longer training.

## Artifacts

- Plain:  `runs/plain_answerability_g42_3b/` (adapter, `eval.json`, `eval_dump.jsonl`)
- Tool:   `runs/tool_adapter_g42_3b/` (adapter, `eval.json`, `eval_dump.jsonl`)
- Generators: `scripts/make_plain_answerability_data.py`,
  `scripts/make_tool_adapter_data.py`
- Evals: `scripts/eval_plain_answerability.py`, `scripts/eval_tool_adapter.py`
- Launchers: `scripts/run_plain_answerability_local.sh`,
  `scripts/run_tool_adapter_local.sh`
- Configs: `src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv-g42.yaml`
  (plain), `sr-tool-adapter-answerability-g42.yaml` (tool)
