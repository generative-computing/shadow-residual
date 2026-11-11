# SR adapters as tools: `data.tool_formatting`

An SR adapter can be exposed to the base model as a **tool**. The base decides
when to call it (it emits a `<tool_call>`), the serving environment routes the
next tokens to the adapter, and the adapter produces the tool's
`<tool_response>` payload. **As a tool, the adapter never writes the assistant
answer.** After the tool turn, the base (or another adapter) resumes and answers.
Tool training also renders half of the source rows as plain (vanilla SR) chat
rows, so the same adapter answers directly when it is invoked after the plain
generation prompt (see [Vanilla rows](#vanilla-rows-datavanilla_frac)).

`data.tool_formatting` selects which of two handoff contracts the adapter is
trained for. It is opt-in. Unset (the default), training is standard chat SFT
with no tool logic.

| | `external_tool_formatting` (bare `--tool-formatting`) | `autoregressive_tool_formatting` |
|---|---|---|
| who writes the tool-turn scaffolding | the serving environment | the adapter |
| adapter takes over after | env-written `<tool_response>\n` | the base's `</tool_call>` |
| adapter emits | `{payload}<\|im_end\|>` | `\n<\|im_start\|>user\n<tool_response>\n{payload}\n</tool_response>\n<\|im_end\|>` |
| adapter stops on | `<\|im_end\|>` (EOS), which the env discards | `<\|im_end\|>` (EOS), which closes the tool turn |
| transcript the base re-reads | byte-identical to the stock template | stock, except the base turn's `<\|im_end\|>` after `</tool_call>` is missing |
| `assistant_marker` / `last_context_token` / `last_token` | `<tool_response>\n` / `\n` / `<\|im_end\|>` | `</tool_call>` / `</tool_call>` / `<\|im_end\|>` |

**Granite 4.2 only.** Both modes rely on Granite 4.2's ChatML tool-turn
rendering (for example `ibm-granite/granite-4.2-3b`). Any other base is
rejected at startup. See [Model support](#model-support).

---

## 1. Data: one row shape for both modes

A tool row is a stock chat row. Its `messages` end with the **trigger** (an
assistant turn carrying exactly one `tool_call`) and, for training, the **gold
tool response** (a `tool` message whose content is the payload the adapter must
produce). The row's `tools` must define the called tool. Optional RAG
`documents` are injected the same way as for every Granite 4.2 row: as a
`tool` message after the last user turn.

```json
{"messages": [
   {"role": "system", "content": "You are a helpful enterprise assistant. ..."},
   {"role": "user", "content": "How long is parental leave?"},
   {"role": "assistant", "content": "", "tool_calls": [
      {"type": "function", "function": {"name": "answerability",
                                        "arguments": {"question": "How long is parental leave?"}}}]},
   {"role": "tool", "content": "{\"answerable\": true}"}
 ],
 "tools": [{"type": "function", "function": {"name": "answerability", "...": "..."}}],
 "documents": [{"document_id": "hr-14", "content": "16 weeks paid leave."}],
 "ground_truth": "answerable"}
```

The rows don't depend on the mode. The trainer renders each row for whichever
mode it runs in, so one data file trains either mode. Eval and generation rows
are the same row without the trailing `tool` message; gold, if any, goes in
`ground_truth`. A trailing gold `tool` message is also accepted at generation
time and is dropped from the prompt.

### Train from source rows: `data.tool_task`

Set `data.tool_task` (or `--tool-task`) and the trainer converts each task
**source** row to a tool row on load, then renders it for the mode. You don't
need a separate data-prep step. Rows the converter can't use are dropped and
counted in the log (`converted N answerability source rows to tool rows (M
dropped as unusable)`). Post-training generation and `sr-generate` convert eval
source rows the same way: the gold tool message is dropped, and `ground_truth`
is kept for the scorer.

```bash
# one command; one source file, validation split held out in the trainer
sr-train --config src/shadow_residual/config/sr-tool-g42.yaml \
  --tool-formatting external_tool_formatting --tool-task answerability \
  --train-data src.jsonl --val-frac 0.1
```

`--val-frac F` (`data.val_frac`) holds out a seeded (`runtime.seed`) fraction of
the train rows instead of reading `data.val_path`. Set exactly one of the two.
`tool_task` requires `tool_formatting`. If a row already defines the task's
tool, it has already been converted, so loading it with `tool_task` set raises
an error. For pre-converted files, unset `tool_task`.

Answerability source rows may carry their documents as inline `tool` turns (the
adapter-bench `g42-alora-4.1-split` shape) instead of a `documents` field. The
converter moves them into `documents` as `{"document_id": <index>, "content":
...}`, the same records the benchmark's own renderer builds.

### Vanilla rows: `data.vanilla_frac`

Tool training from source rows (`tool_task`) renders a seeded (`runtime.seed`)
`round(F * n)` of the usable source rows as **vanilla SR** chat rows; the rest
train as tool rows for the mode. `F` is `data.vanilla_frac` (`--vanilla-frac F`),
**default 0.5**. One adapter then
serves both ways: as a tool, and by last-context-token activation after the
plain generation prompt. `--vanilla-frac 0` trains tool rows only; pre-converted
tool rows (no `tool_task`) always train as tool rows only. Each source row is used once, in one
of the two formats.

A vanilla row is the source row as a plain chat row (documents after the last
user turn, thinking off), and it supervises the source row's own final
assistant turn. The response formats differ. For answerability the vanilla gold
is the quoted label and the tool gold is the dict:

```
vanilla  ...<|im_start|>assistant\n<think></think>   "unanswerable"<|im_end|>\n
tool     ...<|im_start|>user\n<tool_response>\n       {"answerable": false}<|im_end|>
```

| row kind | marker (masked through) | last context token |
|---|---|---|
| vanilla | `<\|im_start\|>assistant\n<think></think>` | `</think>` |
| tool, external | `<tool_response>\n` | `\n` |
| tool, autoregressive | `</tool_call>` | `</tool_call>` |

TRL passes only token ids to the collator, so a row's kind is not available
there. The collator gets both markers and cuts after the one whose last
occurrence **ends last**. That is always the row's own marker: a vanilla row's
documents `<tool_response>\n` comes before its final assistant opener, and a
tool row's trigger-turn assistant opener comes before its tool marker. The
loader checks this on every row with the real tokenization and names the line
of a row that would be cut elsewhere. The dataset gets a `kind` column, and the
boundary validator runs per kind, each against its own last context token.
`adapter_config.json` records `vanilla_frac` and `vanilla_last_context_token`.

Notes:
- The split is exact per file, before a `--val-frac` split; train and val then
  each hold about F vanilla rows.
- Rows the tool converter drops (answerability `conversational`) are dropped for
  both kinds. A vanilla row keeps the source label as is (answerability
  `"partial"` stays `"partial"`); a row with only `ground_truth` gets its gold
  turn built from it.
- In tool mode (with or without vanilla rows), rows of at least
  `data.max_length` tokens are dropped and counted in the log. The trainer would
  truncate them from the end, which cuts off the supervised span and leaves an
  earlier marker (the documents, the trigger turn) as the cut point.
- Post-training generation and `sr-generate` render the tool prompt. Evaluate the
  vanilla side with the plain generation prompt (for example the adapter
  benchmark's own HF+PEFT runner).

### What each task's tool row holds

| `tool_task` | tool call arguments | payload |
|---|---|---|
| `answerability` | `{"question": <last user turn>}` | `{"answerable": true\|false}` (`partial` counts as answerable; `conversational` rows are dropped) |
| `guardian_core` | `{"criteria", "target"}`, lifted out of the `<guardian>` instruction turn | `{"score": "yes"\|"no"}` |
| `requirement_check` | `{"requirements"}`, lifted out of the `<requirements>` instruction turn | `{"score": "yes"\|"no"}` |
| `query_clarification` | `{"query": <last user turn>}` | `{"clarification": "CLEAR" \| <question>}` |

To add a task, write a converter in `tool_data/` that returns the row shape
above and register it in `tool_data/convert.py`. The formatting layer is
task-agnostic.

---

## 2. `external_tool_formatting`

The environment owns the chat template. The adapter only produces the payload.

```
[ctx]  ...<|im_start|>assistant\n<think></think>\n<tool_call>...</tool_call>   <- base
[ctx]  \n<|im_end|>\n<|im_start|>user\n                                       <- env (stock template glue)
[ctx]  <tool_response>\n                                                      <- env; last context token = "\n"
       >>> mask boundary: supervision starts here <<<
[sup]  {"answerable": true}                                                   <- adapter
[sup]  <|im_end|>                                                             <- adapter stop (EOS); env discards it
       \n</tool_response>\n<|im_end|>\n                                       <- env appends; base resumes
```

Serving loop:

1. The base generates until it closes its `</tool_call>`, which names the adapter's tool.
2. The env appends `\n<|im_end|>\n<|im_start|>user\n<tool_response>\n` and
   switches to the adapter.
3. The adapter generates until `<|im_end|>`. Everything before it is the payload.
4. The env drops the EOS and appends `\n</tool_response>\n<|im_end|>\n`, then
   hands back to the base. The base opens the next assistant turn and writes
   the answer.

**Why the span ends in `<|im_end|>` and not `\n</tool_response>`.** Under stock
`generate()` the only stop is EOS. Training the adapter to emit the payload
followed by `<|im_end|>` gives an engine-native stop. The env treats that EOS
as "payload done" and writes the closing tag itself.

**Properties.**
- Once the env suffix is appended, the transcript is byte-identical to
  `apply_chat_template([..., assistant(tool_calls), tool(payload)])`. There
  are no off-template seams.
- The supervised span is only `{payload}<|im_end|>`, so the adapter spends its
  capacity on the decision and none on scaffolding.
- `--thinking` does not apply, because there is no assistant turn.

## 3. `autoregressive_tool_formatting`

The adapter takes over straight after the base's `</tool_call>` and writes the
whole tool-response turn itself, with no env glue.

```
[ctx]  ...<|im_start|>assistant\n<think></think>\n<tool_call>...</tool_call>   <- base; last context token = </tool_call>
       >>> mask boundary: supervision starts here <<<
[sup]  \n<|im_start|>user\n<tool_response>\n                                  <- adapter (no <|im_end|> before it)
[sup]  {"answerable": true}                                                   <- adapter
[sup]  \n</tool_response>\n<|im_end|>                                         <- adapter; <|im_end|> = stop (EOS)
       \n<|im_start|>assistant\n...                                           <- base resumes and answers
```

Serving loop:

1. The base generates until `</tool_call>`.
2. The env switches to the adapter without writing anything.
3. The adapter generates until `<|im_end|>`, which closes the tool turn. The
   payload is the text between `<tool_response>\n` and `\n</tool_response>`
   (`tool_data.extract_tool_payload`).
4. The env hands back to the base, which opens the next assistant turn.

**The single-EOS seam.** The stock template closes the base's tool-call turn
with `\n<|im_end|>` before the tool turn. That `<|im_end|>` is omitted, because
`<|im_end|>` is the stop token: an adapter trained to emit it right after
`</tool_call>` halts immediately and produces nothing. An earlier attempt
trained on the fully stock stream did exactly that. The adapter therefore emits
exactly one `<|im_end|>`, the one that closes its own turn. This is the only
deviation from the stock template.

**Properties.**
- A single `generate()` call covers the tool turn, and no env glue is needed.
- The adapter must learn the scaffolding as well as the payload. In the
  answerability experiments (Granite 4.2-3b) this cost essentially nothing in
  accuracy: 0.95 autoregressive vs 0.954 external. That autoregressive run used
  an earlier span that also emitted an answer-label turn; the shipped span
  stops at the tool turn's `<|im_end|>`.
- The base re-reads one non-stock seam (`</tool_call>\n<|im_start|>user`).

---

## 4. How training uses the mode

`apply_tool_formatting` (in `train.py`) sets the three boundary markers from
the mode (`tool_data.TOOL_FORMATTING_MARKERS`):

- `data.assistant_marker` is the collator's response template. Everything up
  to and including its **last** occurrence is masked.
- `adapter.last_context_token` is the token right before the supervised
  region. `validate_last_context_token_boundary` checks this on real rows
  before training starts.
- `adapter.last_token` (`<|im_end|>`) is the end-of-completion marker.

**Do not set those three fields when `tool_formatting` is set.** The mode is
the single source of truth for the boundary, so setting any of them is a
startup error that names the conflicting field and the value the mode implies.

`load_jsonl_dataset(..., tool_formatting=mode)` renders each row with
`tool_data.render_tool_example`: the mode's prompt followed by the supervised
span. Rendering goes through the real chat template, and every row is checked
against the expected Granite 4.2 seams. A row that doesn't match (no trigger,
more than one tool call, an undefined tool, or a template that renders tool
turns differently) fails with its file and line number.
`data.enable_thinking` and `data.instruction_as_user_message` do not apply to
tool rows.

Because masking cuts at the **last** marker, the trigger must be the last
`</tool_call>` (autoregressive), and the env opener the last `<tool_response>\n`
(external), in the row. Retrieved documents and the system prompt's
tool-format instructions come earlier, so this holds for well-formed rows. A
payload that itself contains `<tool_response>\n` would break it.

The mode is recorded in the saved `adapter_config.json` as `tool_formatting`
(read it back with `generation_utils.read_tool_formatting_from_adapter`).

## 5. Generation and eval

`run_generation(..., tool_formatting=mode)`, which covers post-training,
FSDP and the `sr-generate` CLI:

- renders each eval row up to the mode's handoff point
  (`tool_data.render_tool_prompt`);
- decodes the generation with special tokens kept;
- writes the extracted tool-response payload as `generated_content`
  (`tool_data.extract_tool_payload`). In autoregressive mode the payload is
  unwrapped from the emitted turn. A malformed generation is kept verbatim, so
  it scores as invalid instead of vanishing.

`sr-generate` reads the mode from the adapter's `adapter_config.json`, falling
back to the config's `data.tool_formatting`.

## 6. Model support

`tool_data.check_tool_formatting_support` runs right after the tokenizer loads
(training and generation). It fails, with an error that names the model, unless
all of the following hold:

- the tokenizer is ChatML (`<|im_start|>` is in the vocabulary);
- `<|im_end|>`, `</tool_call>`, `<tool_response>` and `</tool_response>` are
  single tokens;
- a probe conversation renders with Granite 4.2's tool-turn seams
  (`</tool_call>\n<|im_end|>\n<|im_start|>user\n<tool_response>\n{payload}\n</tool_response>\n<|im_end|>\n`).

The check probes the template's behaviour, not the model name, so any Granite
4.2 checkpoint (3b, 8b, local paths) passes and Granite 4.1 is rejected.

## 7. Quick start

One command, straight from answerability source rows (the reference config sets
`tool_task: answerability`):

```bash
sr-train --config src/shadow_residual/config/sr-tool-g42.yaml \
  --base-model ibm-granite/granite-4.2-3b --data-root /path/to/src \
  --eval-data /path/to/src/val.jsonl --run-dir runs/tool              # external (config default)
#        --tool-formatting autoregressive_tool_formatting               # or the other mode
#        --tool-task guardian_core                                      # or another task
#        --train-data src.jsonl --val-frac 0.1                          # one source file
#        --debug-collator 0.05                                          # print training examples
```

`--data-root` holds the source `train.jsonl` / `val.jsonl`; `--run-dir` gets
`checkpoints/`. These flags fill the config's `${DATA_ROOT}` / `${RUN_DIR}` /
`${BASE_MODEL}` / `${EVAL_DATA}` placeholders (the environment variables of the
same names still work; the flag wins). `--debug-collator
[PROB]` always prints the first collated batch. It then prints about PROB of
later batches (default 0.001). Each print shows the decoded row, the last
context token, and the supervised text.

The reference config is
[`src/shadow_residual/config/sr-tool-g42.yaml`](../src/shadow_residual/config/sr-tool-g42.yaml).
