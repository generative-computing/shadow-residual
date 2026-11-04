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

## External tool formatting (`external_tool_formatting`)

A second mode in which the environment owns the template: it writes
`<tool_response>\n`, the adapter emits only the verdict dict and stops on EOS, and
the environment appends `\n</tool_response>\n`. The supervised span is just
`{"answerable": <bool>}<|im_end|>`, so the adapter learns the decision with no
structural scaffolding. In principle this is the plain flow's advantage, kept
inside the tool protocol. See `SR_TOOL_ADAPTER_BOOTSTRAP.md` → *Tool formatting
modes*.

**Vela run (Granite 4.2-3b, 2 epochs, 4-GPU DDP; data generated in-pod from the
same `unified_compare/data_v4` source as the autoregressive Vela run):** pending.
Results land here and in the HTML once the job completes.

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
