# SR adapters as tools — training bootstrap plan (Granite 4.2)

## Goal

Expose Shadow Residual (SR) adapters to Granite as **tools**. The model's native
tool-invocation capability becomes the switching trigger: some of the "tools"
registered with the model are really SR adapters. When the model decides to call
one (e.g. the standard **answerability** adapter), the switch runtime invokes the
adapter *under the hood* right after the model emits its `<tool_call>` — **before
an end-of-text token is produced** — and the adapter generates a real response.

The switching mechanism itself is **out of scope**. This plan covers the
**training bootstrap**: what the adapter must learn to generate, how to shape the
data, where the supervised boundary falls, and how to wire it into the existing
`shadow-residual` training pipeline.

Target base: **Granite 4.2** (ChatML template, `<|im_start|>` / `<|im_end|>`,
`<tool_call>` / `<tool_response>` tags).

**Two formatting modes** (`data.tool_formatting` / `--tool-formatting`; see
[Tool formatting modes](#tool-formatting-modes) below):
- `autoregressive_tool_formatting` *(default when tool training is on)*: the
  adapter takes over after `</tool_call>` and emits the `<tool_response>` turn
  plus the final answer. Most of this document describes this mode.
- `external_tool_formatting`: the environment writes `<tool_response>\n`; the
  adapter emits only the dict and stops on EOS; the environment closes the tag.

Tool training is **opt-in for the repo**: with `tool_formatting` unset (every
config except `sr-tool-adapter-answerability-g42.yaml`) training is standard chat
SFT and pre-rendered `{"text"}` rows are rejected.

---

## What the adapter must generate (confirmed design)

**Decision: the adapter generates the FULL span after `</tool_call>`** — the
`<tool_response>` block and the subsequent assistant answer turn. The switch does
nothing but route tokens to the adapter; it does not synthesize the tool result.
This keeps the runtime trivial at the cost of the adapter having to emit the
structural `<tool_response>` scaffolding.

**Critical constraint: the adapter emits exactly ONE `<|im_end|>` — the terminal
stop.** `<|im_end|>` is the generation stop token. We assume the switch calls a
**stock `generate()` with default EOS handling and no custom stopping criterion**,
so the decode loop halts at the *first* `<|im_end|>` the adapter produces. A
faithful ChatML stream would contain a *second* `<|im_end|>` closing the
`<tool_response>` turn before the answer turn — but emitting that would stop
generation prematurely, before the answer. **So we drop the intermediate turn-
closing `<|im_end|>`:** the adapter flows directly from `</tool_response>\n` into
`<|im_start|>assistant`, and the single `<|im_end|>` at the end of the answer is
its genuine, first-and-only stop.

> **Trade-off.** The emitted stream is *slightly* off-spec ChatML — the
> `<tool_response>` (user) turn is not closed with `<|im_end|>` before the
> assistant turn opens. This is harmless if nothing re-parses the adapter output
> as structured turns (shown to user / logged). If the base later re-ingests the
> transcript as multi-turn context, the explicit `<|im_start|>assistant` tag still
> marks the boundary, but validate coherence in the dry run (open question 5).
>
> **Alternative (only if the switch owns the decode loop):** install a stopping
> criterion that ignores the first N−1 `<|im_end|>` tokens and stops on the Nth.
> Then the adapter can emit fully-stock ChatML including the intermediate
> `<|im_end|>`. Not assumed here, since the runtime is out of scope and we default
> to stock `generate()`.

### Empirical token boundary (rendered with the real 4.2 ChatML template)

What the **base model generates** for the answerability tool-call turn (verified
by diffing `add_generation_prompt=True` prefix against the rendered call turn):

```
\n<tool_call>\n<function=answerability>\n<parameter=question>\n...\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n
```

The base's turn naturally ends with `</tool_call>\n` followed by **`<|im_end|>`,
which is the stop token** — if the base emits it, generation halts and the adapter
never runs. **Therefore the switch must intercept at `</tool_call>` (before
`<|im_end|>` is committed)**, parse the emitted `tool_call`, see that it names an
SR adapter, and flip generation to that adapter.

So the **last context token is `</tool_call>`** (its final sub-token / the newline
immediately after it). The adapter **generates the turn-closing `<|im_end|>`
itself**, plus everything after.

> **Correction note.** An earlier draft put the boundary at the turn-closing
> `<|im_end|>`. That is wrong: `<|im_end|>` is the generation stop token, so the
> base never gets past it. The boundary is `</tool_call>`; the adapter owns
> `<|im_end|>` onward.

> **On `<tool_call>` vs. `tool_calls`.** The literal `<tool_call><function=...>`
> tags exist only in the **rendered template string**. In the message object a
> tool call is the structured **`tool_calls`** key (see
> `example_conversations.json`); the chat template materializes it into the tags,
> and an inference-time parser does the inverse (`</tool_call>` text →
> `tool_calls` object). Training rows therefore store `tool_calls` as structured
> objects, never as literal tag text.

**Adapter-generated span** (what SR is trained to emit — the stock single-
`<|im_end|>` design; see `adapter_span.txt`). The adapter starts at
`<|im_start|>user` (the base's final context token is `</tool_call>`, see below)
and the intermediate turn-closing `<|im_end|>` is **omitted**:

```
<|im_start|>user\n
<tool_response>\n
{"answerable": true, "rationale": "The HR handbook states 16 weeks of paid parental leave is available to all parents regardless of gender, which covers new fathers."}\n
</tool_response>\n
<|im_start|>assistant\n
<think></think>Yes. New fathers are eligible for 16 weeks of paid parental leave, which can be taken any time within the first 12 months after your child's birth or adoption. The policy applies to all parents regardless of gender or caregiver role.<|im_end|>
```

(Exactly one `<|im_end|>`, at the very end. Contrast the fully-stock ChatML
stream, which would also place an `<|im_end|>` after `</tool_response>` — removed
here so stock `generate()` doesn't stop early.)

Breakdown of the three parts the user described:

| Part | Tokens | Who emits it |
|---|---|---|
| **tool_response prefix** | `<|im_start|>user\n<tool_response>\n` | adapter |
| **tool_response body** | the verdict JSON `{"answerable": ..., "rationale": ...}` | adapter (this is the adapter's real "work") |
| **tool_response suffix** | `\n</tool_response>\n` *(no `<|im_end|>` — would stop generation)* | adapter |
| **assistant closure prefix** | `<|im_start|>assistant\n<think></think>` | adapter |
| **assistant closure body** | the user-facing natural-language answer | adapter |
| **assistant closure suffix** | `<|im_end|>` *(the single terminal stop)* | adapter |

> Note on `<think></think>`: with `enable_thinking=False` the template emits an
> empty think block inline. The adapters here are trained thinking-off (consistent
> with the existing answerability adapters — see memory
> `project_answerability_shared_kv_default`). Do **not** pass `--enable-thinking`
> at train or eval.

### Why `<tool_response>` renders as a `user` turn

In the 4.2 template a `role: "tool"` message is wrapped as
`<|im_start|>user\n<tool_response>...</tool_response>\n<|im_end|>`. The adapter
reproduces this with the `user` role tag — the downstream base model reads the
tool result as a user-role tool_response, which is how 4.2 natively represents
retrieved/tool context. (It omits the trailing `<|im_end|>`; see the single-
`<|im_end|>` constraint above.)

---

## Files produced by this bootstrap step

```
docs/sr_tool_adapter/
├── SR_TOOL_ADAPTER_BOOTSTRAP.md   # this plan
├── example_conversations.json          # 2 user / 2 assistant-answer / 2 tool turns, coherent agentic flow
├── rendered_full.txt                   # full conversation rendered through the 4.2 ChatML template
├── adapter_span.txt                    # supervised span, 1st (answerable) call; one <|im_end|>, at the very end
└── adapter_span_unanswerable.txt       # supervised span, 2nd (unanswerable) call; one <|im_end|>, at the very end
```

The example conversation is a RAG/answerability flow:
`search_kb` (plain tool) → `answerability` (SR adapter) → answer → follow-up →
`answerability` (unanswerable) → refusal. The second answerability call lands on
an **unanswerable** question, so the adapter sees both verdict polarities.

---

## Training-data construction

### 1. Row shape

Each training row is a full conversation (`{"messages": [...], "tools": [...]}`)
rendered with `apply_chat_template(..., tools=..., add_generation_prompt=False,
enable_thinking=False)`. This is exactly the shape the existing `data.py` already
consumes (`tools` is passed through verbatim — see `data.py:19`).

Key difference from today's answerability data: today the answerability *verdict*
is the whole completion. Here the completion is the **concatenation of the
tool_response block + the user-facing assistant answer**, because the adapter owns
both.

**The stock template cannot produce the target span directly.** If you render the
full `messages` list (tool msg + assistant answer msg) with
`apply_chat_template`, the template inserts a turn-closing `<|im_end|>` after the
`<tool_response>` block (see `rendered_full.txt`) — which we must NOT train on
(the adapter would stop there). The data generator therefore does **not** feed the
completion through the template as separate `tool` + `assistant` messages. Instead
it builds the supervised completion **by string assembly** to match
`adapter_span.txt` exactly:

```
<|im_start|>user\n<tool_response>\n{verdict_json}\n</tool_response>\n<|im_start|>assistant\n<think></think>{answer}<|im_end|>
```

Only the **context** (everything through `</function>\n</tool_call>\n`) is produced
by `apply_chat_template(messages[:trigger_call+1], add_generation_prompt=False)`;
the completion is appended as the hand-assembled string above. This is the single
non-stock step in the pipeline, and it exists solely to omit the intermediate
`<|im_end|>`. (If the switch later turns out to own the decode loop, drop the
assembly and render stock — see the alternative in the design section.)

### 2. Supervised boundary (the critical piece)

The adapter must be supervised on the span starting **immediately after
`</tool_call>\n`** — i.e. the first supervised token is `<|im_start|>` (opening the
`<tool_response>` turn). The base's last emitted content before the switch
intercepts is `</tool_call>` (it must NOT reach the stop token `<|im_end|>`), so
`</tool_call>` is the last *context* token. The adapter then generates the whole
span above, ending with its single terminal `<|im_end|>`. In the existing pipeline
the boundary is handled by `ResponseOnlyCollator` / `last_context_token`, which
masks everything up to and including the **last** occurrence of a marker and
supervises what follows (`data.py:180`, `data.py:248`).

**The marker is `</function>\n</tool_call>\n`.** It is unique to an adapter
tool-call close and avoids the `<|im_end|>` ambiguity entirely (`<|im_end|>` recurs
at every turn boundary — it must NOT be the marker: it is non-unique, and the one
`<|im_end|>` in the span is the terminal token we need to *supervise*, not mask).

- **(A) Multi-token response template.** Set the collator's `response_template` to
  **`</function>\n</tool_call>\n`** (tokenized to its id sequence). The collator's
  `_find_last_subsequence` already supports multi-token markers, and "last
  occurrence" semantics correctly target the *final* adapter tool-call in a
  multi-call conversation. No model/collator code change — just configuration (plus
  the schema tweak noted below if it only accepts a single-token
  `last_context_token`).
- **(B) Dedicated boundary token.** Introduce a sentinel after `</tool_call>`.
  Rejected: diverges from stock 4.2 tool semantics and would require the switch to
  inject the sentinel at inference, defeating the "100% stock tool invocation" goal.

Go with **(A)**. Validate (adapting `validate_last_context_token_boundary` to a
multi-token marker) that `labels != -100` starts exactly at the `<|im_start|>`
immediately after `</function>\n</tool_call>\n` — matching `adapter_span.txt`,
which begins with `<|im_start|>user`.

> **Single-call-per-row simplification for v1.** The "last occurrence" rule means
> a conversation with *two* adapter tool-calls (like the example) would only
> supervise the *second* one. For the first training pass, emit **one training row
> per adapter tool-call** by truncating each conversation at the end of the
> adapter's answer turn. The example conversation thus yields two rows:
> row 1 context = `msgs[:5]` + answerable completion (`msgs[5]` verdict, `msgs[6]`
> answer); row 2 context = `msgs[:9]` + unanswerable completion (`msgs[9]` verdict,
> `msgs[10]` answer). This guarantees every adapter call is supervised and keeps
> the boundary rule simple.

### 3. What to mask

- Mask (`-100`): system + tools block, all user turns, and the trigger
  tool-call turn itself — everything through **`</function>\n</tool_call>\n`**.
- Supervise: the `<tool_response>` block **and** the assistant answer turn (the
  full adapter span above, which begins at `<|im_start|>user` and ends at the
  single terminal `<|im_end|>`). The intermediate turn-closing `<|im_end|>` is
  **not present** in the data, so it is neither masked nor supervised.

**One flat text span crossing three rendered turns.** A training row is *not*
stored as separate `tool_calls` / `tool` / `assistant` message objects — it is the
literal rendered text, because that is exactly the token sequence present at
inference. The supervised + masked regions together span **three** consecutive
rendered turns:

1. `<|im_start|>assistant\n<think></think>\n<tool_call>…</tool_call>\n` — the base's
   tool-call turn. **MASKED** (the base emits this at inference, not the adapter).
2. `<|im_start|>user\n<tool_response>…</tool_response>\n` — the adapter's verdict.
   **SUPERVISED.**
3. `<|im_start|>assistant\n<think></think>{answer}<|im_end|>` — the adapter's answer.
   **SUPERVISED.**

So the row's text includes **both** the tool-call generation and the tool_response
inline, but the mask boundary (right after `</tool_call>\n`) splits them: the
tool-call is masked context, the tool_response + answer is the supervised target.
It is a flat string that *crosses* turns 1→2→3, not a single assistant message
object. Verbatim `[M]`/`[S]` annotation:

```
[M] <|im_start|>assistant\n
[M] <think></think>\n
[M] <tool_call>\n
[M] <function=answerability>\n
[M] <parameter=question>\n
[M] What is our company's parental leave policy for new fathers?\n
[M] </parameter>\n
[M] </function>\n
[M] </tool_call>\n
    >>> MASK BOUNDARY (right after </tool_call>\n) <<<
[S] <|im_start|>user\n
[S] <tool_response>\n
[S] {"answerable": true, "rationale": "…"}\n
[S] </tool_response>\n
[S] <|im_start|>assistant\n
[S] <think></think>Yes. New fathers are eligible for 16 weeks…<|im_end|>
```

---

## Config / pipeline wiring

Reuse the existing SR training path (`sr-train`, YAML-driven). Starting point:
`src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv.yaml`, pointed at a 4.2
base.

Changes vs. the stock answerability config:
- `model.base`: Granite 4.2 (3b first, mirroring the proven 3b answerability run —
  memory `project_answerability_3b_run`).
- `adapter.target_modules`: keep `cross_stream` + q/o/MLP projections; **never**
  `k_proj`/`v_proj` (SR invariant #2).
- `adapter.last_context_token`: use the **single token `</tool_call>`** (id 100271
  on the 4.2 tokenizer). This is the token immediately before the supervised
  region, so it satisfies the single-token boundary validator with no code change.
  The collator `response_template` (`data.assistant_marker`) is set to the same
  `</tool_call>`, which masks through it and supervises the completion that follows.
  (The earlier plan proposed the multi-token `</function>\n</tool_call>\n`; the
  single-token `</tool_call>` is cleaner because `</tool_call>` tokenizes to one id
  and the existing single-token validator then just works.)
- `adapter.last_token`: `<|im_end|>` (end-of-completion marker).
- `data.enable_thinking`: **False**.

> **IMPLEMENTED (branch `experiments/sr-tool-adapter`).** The generator emits
> **pre-rendered `{"text"}` rows** (consumed by `load_jsonl_dataset`'s
> `allow_prerendered` passthrough, enabled only when `data.tool_formatting` is
> set), hand-assembling the completion so the supervised span carries
> **exactly one, terminal `<|im_end|>`** — the intermediate turn-closing
> `<|im_end|>` after `</tool_response>` is omitted (it is the EOS/stop token;
> supervising it would teach the adapter to halt before emitting the answer).
> Verified against the stock 4.2 template: our span is byte-identical to the
> template's except for the two omitted `<|im_end|>` seams (after `</tool_call>`
> and after `</tool_response>`).
>
> **`--thinking` flag (3 modes)** controls what the answer turn emits after
> `<|im_start|>assistant\n`:
> - `off` *(default)* — nothing: `<|im_start|>assistant\n"answerable"<|im_end|>`
> - `enabled` — open think tag: `<|im_start|>assistant\n<think>"answerable"<|im_end|>`
> - `disabled` — empty closed block: `<|im_start|>assistant\n<think></think>"answerable"<|im_end|>`
>
> The collator cuts at `</tool_call>`, so the think prefix is part of the
> **supervised** span (the adapter learns to emit it). `eval_tool_adapter.py` is
> thinking-mode agnostic: it strips a closed or dangling `<think>` block before
> reading the answer.
>
> **Data source.** `make_tool_adapter_data.py` accepts either the raw
> `answerability_evaluation.jsonl` shape (explicit `ground_truth` field) **or**
> the already-rendered `unified_compare/data_v4` shape (no `ground_truth` — the
> verdict is the final assistant turn, derived via `_verdict_from_messages`).
> Artifacts:
> - `scripts/make_tool_adapter_data.py` — answerability rows → tool-adapter convos
>   (`--thinking {off,enabled,disabled}`).
> - `src/shadow_residual/config/sr-tool-adapter-answerability-g42.yaml` — config
>   (`assistant_marker` / `last_context_token` = `</tool_call>`).
> - `scripts/eval_tool_adapter.py` — structural + verdict-accuracy + consistency eval.
> Boundary verified on real data: token before supervision = `</tool_call>`;
> supervised span = verdict `<tool_response>` + answer, ending at the single
> terminal `<|im_end|>`.

---

## Tool formatting modes

`data.tool_formatting` selects how much of the chat template the adapter owns.
Setting it opts in to tool training and fixes the three boundary markers
(`train.py:_TOOL_FORMATTING_MARKERS`, overriding any YAML values); the mode is
recorded into `adapter_config.json`.

| | `autoregressive_tool_formatting` | `external_tool_formatting` |
|---|---|---|
| adapter takes over after | base's `</tool_call>` | env-written `<tool_response>\n` |
| adapter emits | `\n<|im_start|>user\n<tool_response>\n{dict}\n</tool_response>\n<|im_start|>assistant\n[think]answer<|im_end|>` | `{dict}<|im_end|>` |
| env writes afterwards | nothing (one generate covers 3 turns) | `\n</tool_response>\n<|im_end|>\n`, then the base answers |
| `assistant_marker` | `</tool_call>` | `<tool_response>\n` |
| `last_context_token` | `</tool_call>` | `\n` |
| `last_token` | `<|im_end|>` | `<|im_end|>` |
| `--thinking` | applies (answer-turn prefix) | n/a (generator rejects non-`off`) |
| transcript vs. stock template | two omitted `<|im_end|>` seams | **byte-identical** after the env suffix |

### External mode, token by token

```
[M] ...</function>\n</tool_call>          <- base
[M] \n<|im_end|>\n<|im_start|>user\n      <- env (stock template glue)
[M] <tool_response>\n                     <- env; last context token = "\n"
    >>> MASK BOUNDARY <<<
[S] {"answerable": true}                  <- adapter
[S] <|im_end|>                            <- adapter's EOS stop; env discards it
    \n</tool_response>\n<|im_end|>\n       <- env appends; base resumes
```

The adapter is trained to end on `<|im_end|>` rather than `\n</tool_response>`
because stock `generate()` stops only on EOS. The env treats EOS as "payload
done" and writes the closing tag itself. (`</tool_response>` is a single special
token, id 100273, so an env that prefers it can use it as a stop string.)

The generator builds the external context by rendering the conversation plus a
sentinel `tool` message through the stock template and cutting before the
sentinel. The context is therefore exactly the template's own tool-turn opener,
and it extends the autoregressive context (identical through `</tool_call>`).
`tests/training/test_tool_formatting.py` asserts template byte-identity, that the
collator supervises exactly `{dict}<|im_end|>`, and the boundary validator for
both modes.

```bash
python scripts/make_tool_adapter_data.py --input <answerability.jsonl> --base-model <4.2> \
  --out-train train.jsonl --out-val val.jsonl --tool-formatting external_tool_formatting
sr-train --config src/shadow_residual/config/sr-tool-adapter-answerability-g42.yaml \
  --tool-formatting external_tool_formatting --train-data train.jsonl --val-data val.jsonl ...
python scripts/eval_tool_adapter.py --base-model <4.2> --adapter <ckpt> --val val.jsonl
```

`eval_tool_adapter.py` reads the mode from each row's `metadata.tool_formatting`.
In external mode it cuts the prompt at `<tool_response>\n`, requires the whole
generation (up to EOS) to parse as a dict with a boolean `answerable`, and reports
`structural_rate`, `verdict_accuracy`, a confusion table, and `eos_stop_rate`.

### First run result (2026-10-19, Granite 4.2-3b, local MPS)

`runs/tool_adapter_g42_3b` — 120 balanced rows (60 ans / 60 un), 1 epoch, 15
optimizer steps, ~39 min train on MPS. Base
`ibm-research/granite-4.2-3b-prerelease-r260622a`.

- Train: `train_loss≈0.70` (avg), final-step `loss=0.281`,
  `mean_token_accuracy=0.946`. Held-out teacher-forced: `eval_loss=0.258`,
  `eval_mean_token_accuracy=0.942`. No NaN, no overfit.
- Free-generation eval (`scripts/eval_tool_adapter.py`, 60 val rows):
  - **`structural_rate=1.0`** — every generation produced a valid, parseable
    `<tool_response>` with a boolean `answerable`. **The core goal: the adapter
    reliably generates the tool_response + answer at inference under stock
    `generate()`.** This is exactly what the single-`<|im_end|>` fix enables (a
    2-step smoke adapter trained on the stock-template form instead emitted
    `\n<|im_end|>` and halted — see the design section).
  - `consistency_rate=0.933` — verdict polarity matches the answer polarity
    (answerable→answer, unanswerable→refusal). The verdict↔answer coupling was
    learned well.
  - `verdict_accuracy=0.517` — ~chance; the adapter is biased toward one class.
    Expected for this bootstrap: 120 rows, 1 epoch, and the answerability set has
    **no gold answers**, so completions use synthetic templated rationales/answers
    — the decision signal is thin. The *pipeline* is validated end-to-end;
    accuracy needs real gold-answer data + scale (full 3191-row set, multi-epoch,
    GPU) to be meaningful.

**Takeaway:** structure/plumbing is proven on real Granite 4.2; the next lever is
data quality (gold answers, more rows) and compute, not the training mechanics.

---

## Open questions to resolve before training

1. **Multiple adapters, one model.** Each SR adapter is a separate PEFT adapter.
   Serving multiple tool-adapters means either (a) multiple adapters loaded and the
   switch selects by `adapter.set_adapter(name)` keyed on the called function name,
   or (b) one multi-task adapter. Out of scope for *this* bootstrap, but the data
   format should tag which tool each row trains (`metadata.tool_name`) so we can
   split per-adapter later.
2. **Does the base ever legitimately answer without a tool?** The system prompt
   says ground every answer. If the base sometimes answers directly (no
   tool_call), those turns must stay masked — the adapter only owns post-tool-call
   spans. The one-row-per-adapter-call construction handles this automatically.
3. **tool_response fidelity.** The adapter is generating JSON that must parse. Add
   a post-generation validator in eval that asserts the `<tool_response>` body is
   valid JSON with the expected keys (`answerable`, `rationale`) before scoring the
   answer. Malformed structural scaffolding is a train-time failure mode to watch.
4. **Verdict↔answer consistency.** Because the adapter emits *both* the verdict and
   the answer, train data must keep them consistent (answerable→answer,
   unanswerable→refusal). Consider an eval check that the answer polarity matches
   the emitted verdict.
5. **Off-spec transcript coherence (from the single-`<|im_end|>` design).** The
   emitted stream drops the `<|im_end|>` that would normally close the
   `<tool_response>` (user) turn. Verify the base re-ingests this as context
   without boundary confusion when the conversation continues (the follow-up turn
   in the example). If it degrades, the fallback is the "switch owns the decode
   loop" variant (fully-stock ChatML + custom stop criterion), which requires
   runtime support that is currently out of scope.

---

## Next steps (proposed order)

1. **Confirm the boundary approach (A)** and extend the config schema for a
   multi-token `response_template` if needed.
2. **Build a data generator** that takes answerability-style rows (question +
   documents + gold verdict + gold answer) and emits the agentic conversation shape
   above, one row per adapter call, with `tools` including the `answerability`
   function spec.
3. **Dry-run the collator** on a handful of rows; assert the supervised span equals
   `adapter_span.txt` byte-for-byte (minus the specific text) — i.e. starts at
   `<|im_start|>user` (right after `</function>\n</tool_call>\n`), ends at the
   answer's `<|im_end|>`, and contains **exactly one** `<|im_end|>`.
4. **Train SR r32 on 4.2-3b**, thinking-off, `--skip-post-train-generate` under
   FSDP, then eval as a separate single-GPU process.
5. **Add structural + consistency eval checks** (open questions 3 & 4).
6. Scale to 8b and additional adapters once the 3b answerability-as-tool run is
   validated.


---

## Worked example — the real inference trace (Granite 4.2, answerability-as-tool)

A single-tool flow: the retrieved documents are already in context (RAG-injected as a `tool` message), the model is given one tool — `answerability` — decides to call it, and the SR adapter then produces the tool_response verdict **and** the user-facing answer. Everything below is rendered with the real 4.2 ChatML template (`trim_blocks=True, lstrip_blocks=True`), `enable_thinking=False`.

### Messages that produce this

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "answerability",
        "description": "Decide whether the user's question is answerable from the retrieved documents currently in context. ...",
        "parameters": {
          "type": "object",
          "properties": {
            "question": {
              "type": "string"
            }
          },
          "required": [
            "question"
          ]
        }
      }
    }
  ],
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful enterprise assistant. Answer only from the retrieved documents. If a question cannot be answered from them, say so."
    },
    {
      "role": "user",
      "content": "What is our company's parental leave policy for new fathers?"
    },
    {
      "role": "tool",
      "content": "[{\"source\": \"hr_handbook\", \"document_id\": \"hr-14\", \"content\": \"Parental leave: ... 16 weeks ... regardless of gender or caregiver role.\"}]"
    }
  ]
}
```

### STEP 0 — `answerability` defined as a tool

Passed via `apply_chat_template(tools=[...])`; the template renders it into the system turn inside a `<tools>` block:

```
<tools>
{"name": "answerability", "description": "Decide whether the user's question is answerable from the retrieved documents currently in context. Returns a verdict and a short rationale. Call this before answering any grounded/RAG question so the assistant can refuse unanswerable questions instead of hallucinating.", "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "The user's question to assess for answerability against the in-context documents."}}, "required": ["question"]}}
</tools>
```

### STEP 1 — the formatted system prompt (the `<|im_start|>system` turn)

The tool def is embedded in the system turn, followed by the fixed tool-calling instructions:

```
<|im_start|>system
You are a helpful enterprise assistant. Answer only from the retrieved documents. If a question cannot be answered from them, say so.

# Tools

You have access to the following functions:

<tools>
{"name": "answerability", "description": "Decide whether the user's question is answerable from the retrieved documents currently in context. Returns a verdict and a short rationale. Call this before answering any grounded/RAG question so the assistant can refuse unanswerable questions instead of hallucinating.", "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "The user's question to assess for answerability against the in-context documents."}}, "required": ["question"]}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
```

### STEP 1 (cont.) — the user query turn

```
<|im_start|>user
What is our company's parental leave policy for new fathers?<|im_end|>
```

### STEP 1 (cont.) — the retrieved documents (RAG-injected as a tool_response → user turn)

```
<|im_start|>user
<tool_response>
[{"source": "hr_handbook", "document_id": "hr-14", "content": "Parental leave: All employees who become parents are eligible for 16 weeks of paid parental leave, which may be taken any time within the first 12 months after the birth or adoption of a child. This applies regardless of gender or caregiver role."}]
</tool_response>
<|im_end|>
```

### STEP 1 (cont.) — the generation prompt (base decodes from here)

```
<|im_start|>assistant
<think></think>
```

### STEP 2 — the base invokes `answerability` (its generated tool_call)

The base emits the tool call and would stop at `<|im_end|>`:

```

<tool_call>
<function=answerability>
<parameter=question>
What is our company's parental leave policy for new fathers?
</parameter>
</function>
</tool_call>
<|im_end|>
   <-- trailing <|im_end|> = base's natural stop
```

### STEP 3 — the switch cuts at `</tool_call>` (before `<|im_end|>`)

Last context token = `</tool_call>`. The switch parses the call (`name="answerability"`), recognizes it as an SR adapter, swaps the adapter in, keeps the KV cache, and resumes decoding. What it keeps from the base:

```

<tool_call>
<function=answerability>
<parameter=question>
What is our company's parental leave policy for new fathers?
</parameter>
</function>
</tool_call>
>>> CUT HERE (the <|im_end|> is NOT sampled) <<<
```

### STEP 4 — the adapter generates the tool_response, then stops at the assistant opener

The adapter emits the verdict `<tool_response>` turn **and the opener of the next
assistant turn**, then **stops at `<|im_start|>assistant\n`**. It does **not**
write the answer — control hands back to the base (or another adapter) to generate
the assistant answer. Which entity generates that answer is **out of scope** here.
Single-`<|im_end|>` seam: none after `</tool_call>` (the cut) and none after
`</tool_response>` before the opener.

```
<|im_start|>user
<tool_response>
{"answerable": true, "rationale": "The HR handbook states 16 weeks of paid parental leave is available to all parents regardless of gender, which covers new fathers."}
</tool_response>
<|im_start|>assistant
   <-- ADAPTER STOPS HERE (right after the assistant opener)
```

### STEP 5 — the base (or another adapter) writes the answer (out of scope)

Resuming from the handoff point (cursor right after `<|im_start|>assistant\n`), the
base or a downstream adapter generates the user-facing answer and its terminal
`<|im_end|>`. This turn is **not** the answerability adapter's output:

```
<think></think>Yes. New fathers are eligible for 16 weeks of paid parental leave, taken any time within the first 12 months after the birth or adoption. The policy applies to all parents regardless of gender.<|im_end|>
```

### Complete decoded transcript (prompt + base-kept + adapter)

```
<|im_start|>system
You are a helpful enterprise assistant. Answer only from the retrieved documents. If a question cannot be answered from them, say so.

# Tools

You have access to the following functions:

<tools>
{"name": "answerability", "description": "Decide whether the user's question is answerable from the retrieved documents currently in context. Returns a verdict and a short rationale. Call this before answering any grounded/RAG question so the assistant can refuse unanswerable questions instead of hallucinating.", "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "The user's question to assess for answerability against the in-context documents."}}, "required": ["question"]}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
<|im_start|>user
What is our company's parental leave policy for new fathers?<|im_end|>
<|im_start|>user
<tool_response>
[{"source": "hr_handbook", "document_id": "hr-14", "content": "Parental leave: All employees who become parents are eligible for 16 weeks of paid parental leave, which may be taken any time within the first 12 months after the birth or adoption of a child. This applies regardless of gender or caregiver role."}]
</tool_response>
<|im_end|>
<|im_start|>assistant
<think></think>
<tool_call>
<function=answerability>
<parameter=question>
What is our company's parental leave policy for new fathers?
</parameter>
</function>
</tool_call>
<|im_start|>user
<tool_response>
{"answerable": true, "rationale": "The HR handbook states 16 weeks of paid parental leave is available to all parents regardless of gender, which covers new fathers."}
</tool_response>
<|im_start|>assistant
<think></think>Yes. New fathers are eligible for 16 weeks of paid parental leave, taken any time within the first 12 months after the birth or adoption. The policy applies to all parents regardless of gender.<|im_end|>
```

> **Two deliberate seams** (both from the single-`<|im_end|>` design): there is no `<|im_end|>` after the base's `</tool_call>` (the cut point), and none after the adapter's `</tool_response>` before `<|im_start|>assistant` (the dropped intermediate stop). The adapter emits exactly one `<|im_end|>`, at the very end — its genuine stop under stock `generate()`.

---

## How the generation is parsed back into messages (answers open-question 5)

One `generate()` call produces a stream spanning **three** logical turns
(assistant tool_call → tool_response → final assistant answer), because the
single terminal `<|im_end|>` is the only stop. How an inference engine turns that
back into discrete messages depends entirely on the serving stack. Verified facts
(Granite 4.2 tokenizer): `<|im_start|>` (id 100256) and `<|im_end|>` (id 100257)
are both **single special tokens**; the accumulated generation holds 2 embedded
`<|im_start|>` + 1 terminal `<|im_end|>`.

**Scenario A — stock chat-completions server (the default, and the trap).** The
server treats the whole generation as **one assistant message**, runs the Granite
tool-call parser to lift `<tool_call>…</tool_call>` into `tool_calls[]`, and leaves
the rest as `content`. The `<tool_response>` and the final answer are **stranded
inside `content`** as raw text. With the OpenAI-default `skip_special_tokens=True`,
the `<|im_start|>` markers are **deleted on decode**, so the role boundaries vanish
entirely (`…</tool_call>\nuser\n<tool_response>…\nassistant\n"answerable"`). The
three messages are **not** recovered.

**Scenario B — turn-aware re-segmentation (opt-in).** If the switch runtime decodes
with `skip_special_tokens=False` and splits on `<|im_start|>ROLE`, it recovers all
three messages cleanly: `assistant`(tool_call) → `user`(tool_response) →
`assistant`(answer). This only works because the role markers survive, and it is
**not** what a stock endpoint does — the runtime must deliberately re-split.

**Scenario C — residual seam ambiguities even when B succeeds.** (1) The
tool_response comes back under role **`user`** (how 4.2 renders tool output), not
role `tool`; a downstream re-render could mislabel it. (2) The dropped intermediate
`<|im_end|>` means a **strict** parser that requires every `<|im_start|>` to be
`<|im_end|>`-closed before the next opens will see malformed turns.

**Bottom line.** The 3-message split is only recoverable if the switch runtime
(a) keeps special tokens on decode and (b) deliberately re-segments on
`<|im_start|>`. A naive OpenAI server collapses everything into one assistant
message with the tool_response + answer stranded in `content` and markers
stripped. The switch runtime — not the model — owns this responsibility.
