# SPDX-License-Identifier: Apache-2.0
"""Unified training-config schema.

A single Pydantic model for adapter training. The schema does not name
specific architectures (LoRA vs. shadow-residual) — those are implementation
details. The adapter is always active (no gated/aLoRA activation); the two
optional token fields shape the *data/labels*, not the model forward:

- `adapter.last_context_token` — the final context/prompt token. When set,
  train.py validates that the supervised region (`labels != -100`) starts
  immediately after it (validate + error).
- `adapter.last_token` — end-of-completion marker. When set, train.py appends
  it to any training row whose tokens don't already end with it.

Defaults follow the canonical synthesis in docs/TRAINING_CONFIG_DELTA.html §8.

Usage:
    from shadow_residual.config import load_training_config
    cfg = load_training_config("path/to/config.yaml")
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# --- Enums --------------------------------------------------------------------


class SchedulerType(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class IntervalStrategy(str, Enum):
    NO = "no"
    EPOCH = "epoch"
    STEPS = "steps"


class Bias(str, Enum):
    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"


class ToolFormatting(str, Enum):
    """How an SR adapter invoked as a tool relates to the chat template.

    Granite 4.2 (ChatML) only. In both modes the adapter produces the tool
    response and stops; it never writes the assistant answer (the base, or
    another adapter, resumes after the tool turn).

    - EXTERNAL (bare ``--tool-formatting``): the serving environment owns the
      template. It closes the base's tool-call turn, writes
      ``<|im_start|>user\\n<tool_response>\\n`` and hands over; the adapter emits
      only the tool-response payload and stops on EOS (``<|im_end|>``). The
      environment then appends ``\\n</tool_response>\\n<|im_end|>\\n``.
    - AUTOREGRESSIVE: the adapter takes over right after the base's
      ``</tool_call>`` and emits the whole tool-response turn,
      ``\\n<|im_start|>user\\n<tool_response>\\n{payload}\\n</tool_response>\\n<|im_end|>``,
      stopping on that ``<|im_end|>``.

    Each mode fixes the boundary markers; see ``docs/tool_formatting.md``.
    """
    AUTOREGRESSIVE = "autoregressive_tool_formatting"
    EXTERNAL = "external_tool_formatting"


class ToolTask(str, Enum):
    """Task source format converted to tool rows on load (``data.tool_task``).

    Mirrors ``shadow_residual.training.tool_data.convert.TASKS``.
    """
    ANSWERABILITY = "answerability"
    GUARDIAN_CORE = "guardian_core"
    REQUIREMENT_CHECK = "requirement_check"
    QUERY_CLARIFICATION = "query_clarification"


# --- Sub-models ---------------------------------------------------------------


class _Strict(BaseModel):
    """Base for all sub-models — forbids unknown keys so typos fail loudly."""
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ModelConfig(_Strict):
    base: str = Field(
        ...,
        description=(
            "HF model ID or local path: an SR checkpoint (e.g. AlonMal/granite-4.1-3b-sr, "
            "made with sr-convert) or an upstream Granite model (e.g. "
            "ibm-granite/granite-4.1-3b, converted in memory)"
        ),
    )
    attn_implementation: str | None = None  # e.g. "sdpa", "flash_attention_2"
    # Dtype the SR+PEFT model is materialized in. "bf16" (default) loads the
    # whole model — base AND trainable LoRA adapters — in bfloat16. "fp32"
    # loads everything in float32; with accelerate mixed_precision: bf16 this
    # gives bf16 *compute* via AMP while keeping fp32 master weights, fp32
    # gradients, and fp32 optimizer state. fp32 master weights are the
    # PEFT-recommended recipe for adapter training stability (PEFT keeps
    # adapters in fp32 by design); the bf16 default force-casts adapters down
    # to bf16 to satisfy FSDP's uniform-dtype flatten, which loses adapter
    # precision and destabilizes training at 8B/30B scale.
    param_dtype: Literal["bf16", "fp32"] = "bf16"


class DataConfig(_Strict):
    train_path: str
    # Exactly one of val_path / val_frac. val_frac holds out a seeded
    # (runtime.seed) fraction of the train rows as the validation set.
    val_path: str | None = None
    val_frac: float | None = Field(None, gt=0.0, lt=1.0)
    max_length: PositiveInt = 8192
    # Optional response-template string for label masking. Used only when the
    # tokenizer's chat template lacks `{% generation %}` markers (in which case
    # we fall back from TRL's `assistant_only_loss=True` to a per-batch
    # response-only collator). Default None = "let train.py pick a sensible
    # value for the model family".
    assistant_marker: str | None = None
    # Forwarded to `apply_chat_template(enable_thinking=...)` when rendering
    # each training row. Toggles the model's thinking/reasoning trace for
    # templates that support it (e.g. Granite). Default False preserves prior
    # behavior; templates that ignore the kwarg treat it as a no-op. Overridable
    # from the CLI via `--thinking` (CLI wins).
    enable_thinking: bool = False
    # ChatML (Granite 4.2) document placement. When RAG `documents` are
    # injected as a tool (search-response) message, this controls where it lands:
    #   - False (default): after the LAST user turn (plain agentic RAG flow).
    #   - True: before the FIRST assistant turn — for judge/guardian data where a
    #     later user turn carries a judging *instruction* and the documents are
    #     grounding for the first assistant answer being evaluated.
    # No-op for non-ChatML (Granite 4.1) rendering and for rows without documents.
    # Overridable from the CLI via `--instruction-as-user-message` (CLI wins).
    instruction_as_user_message: bool = False
    # Opt-in SR-adapter-as-tool training (Granite 4.2 only). None (default) =
    # standard chat SFT with no tool logic. When set, rows end with the trigger
    # tool call + gold tool response and are rendered for the mode, and the mode
    # sets assistant_marker / last_context_token / last_token itself (setting
    # any of them too is an error). Overridable from the CLI via
    # `--tool-formatting [MODE]` (CLI wins; a bare flag selects external).
    tool_formatting: ToolFormatting | None = None
    # Requires tool_formatting. When set, train/val (and generation) rows are
    # task SOURCE rows, converted to tool rows on load by the task's converter
    # (shadow_residual.training.tool_data); unusable rows are dropped. None =
    # rows are already tool rows. Overridable from the CLI via `--tool-task`.
    tool_task: ToolTask | None = None
    # Tool training renders this seeded (runtime.seed) fraction of the source
    # rows as plain (vanilla SR) chat rows — the adapter answers after
    # "<|im_start|>assistant\n<think></think>" — and the rest as tool rows.
    # Needs tool_task. None (unset) = 0.5 with tool_task, else 0; 0 = tool rows
    # only. The split is exact before a val_frac split. Overridable from the
    # CLI via `--vanilla-frac F`.
    vanilla_frac: float | None = Field(None, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _one_val_source(self):
        if (self.val_path is None) == (self.val_frac is None):
            raise ValueError("set exactly one of data.val_path / data.val_frac")
        return self


class AdapterConfig(_Strict):
    # `target_modules` carries BOTH the module-name selection and the LoRA
    # rank in a single field. Two shapes:
    #
    #   - scalar int (e.g. `32`)         → uniform rank on the SR-safe target
    #                                      set (q/o + MLP + cross_stream; NOT
    #                                      "all-linear" — that would wrap the
    #                                      forbidden K/V projections)
    #   - dict[str, int]                 → per-module names AND ranks
    #
    # The dict form is concise (no separate `rank` field, no risk of the
    # names-and-ranks lists drifting out of sync).
    target_modules: PositiveInt | dict[str, PositiveInt] = 32
    # LoRA scaling is `alpha / r`, so per-module RANK alone can't hold the
    # effective scale fixed while rank varies. Two shapes, mirroring
    # `target_modules`:
    #   - scalar int (or None → 2 * max rank) → one `lora_alpha` for every module
    #   - dict[str, int]                      → per-module alpha (PEFT's
    #                                           `alpha_pattern`); requires the
    #                                           dict form of `target_modules` with
    #                                           EXACTLY the same key set.
    #
    # Dict values may be 0 (the scalar form may not): a per-module alpha of 0
    # zeroes that module's LoRA scale (`alpha/r == 0`), so its delta is identically
    # 0 on the forward AND receives no gradient — the module is pinned at its init.
    # (Used e.g. to run a `cross_stream` tap present-but-silent.)
    alpha: PositiveInt | dict[str, NonNegativeInt] | None = None
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    bias: Bias = Bias.NONE
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    # The adapter is always active. These two optional single-token markers
    # shape the training data/labels (enforced in train.py) and are recorded
    # into the saved adapter_config.json:
    #   - last_context_token: the final context/prompt token. When set, the
    #     supervised region (labels != -100) must start immediately after it;
    #     train.py validates this and errors on mismatch.
    #   - last_token: end-of-completion marker. When set, train.py appends it
    #     to any row whose tokens don't already end with it.
    # Each must encode to exactly ONE token id (validated in train.py where the
    # tokenizer is available).
    last_context_token: str | None = None
    last_token: str | None = None
    # MoE bases (granitemoe) only: when true (the default), the
    # decoder routes once on the base stream and reuses that expert partition for
    # the adapter stream; set false to route each stream independently. No-op on
    # dense bases. Recorded into the saved adapter_config.json and read back by
    # the serving path (training.generation_utils.read_share_moe_routing_from_adapter)
    # so the adapter self-describes its routing mode.
    share_moe_routing: bool = True

    @field_validator("target_modules")
    @classmethod
    def _validate_target_modules(cls, v):
        if isinstance(v, dict) and not v:
            raise ValueError("target_modules dict cannot be empty")
        return v

    @model_validator(mode="after")
    def _default_alpha(self):
        # Default alpha to 2 * (max) rank.
        if self.alpha is None:
            r = self.target_modules
            max_rank = r if isinstance(r, int) else max(r.values())
            self.alpha = 2 * max_rank
        return self

    @model_validator(mode="after")
    def _validate_alpha_keys(self):
        # A dict `alpha` is only meaningful alongside a dict `target_modules`, and
        # it must cover EVERY targeted module. Rationale: PEFT resolves a single
        # `target_name_key` from chain(rank_pattern, alpha_pattern) and looks up
        # BOTH dicts with it, so a partial alpha_pattern mis-resolves silently.
        # Beyond that, an omitted module falling back to the scalar default is
        # exactly the uncontrolled scale change a parameter-aligned ablation is
        # trying to eliminate — so require the key sets to match exactly.
        if not isinstance(self.alpha, dict):
            return self
        if not isinstance(self.target_modules, dict):
            raise ValueError(
                "a per-module `alpha` dict requires the dict form of "
                "`target_modules` (per-module alpha has no meaning without "
                "per-module names/ranks)"
            )
        alpha_keys = set(self.alpha)
        target_keys = set(self.target_modules)
        if alpha_keys != target_keys:
            missing = sorted(target_keys - alpha_keys)
            extra = sorted(alpha_keys - target_keys)
            raise ValueError(
                "`alpha` dict keys must match the `target_modules` keys exactly; "
                f"missing={missing} unexpected={extra}"
            )
        return self


class OptimizerConfig(_Strict):
    name: str = "paged_adamw_8bit"
    learning_rate: PositiveFloat = 2.0e-4
    weight_decay: float = Field(0.01, ge=0.0)
    max_grad_norm: PositiveFloat = 1.0


class SchedulerConfig(_Strict):
    type: SchedulerType = SchedulerType.COSINE
    warmup_ratio: float = Field(0.05, ge=0.0, le=1.0)


class BatchConfig(_Strict):
    per_device_train: PositiveInt = 1
    gradient_accumulation: PositiveInt = 16
    per_device_eval: PositiveInt = 1
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] = Field(default_factory=lambda: {"use_reentrant": False})


class TrainerConfig(_Strict):
    """SFTTrainer-specific knobs that aren't part of the optimizer/batch story."""
    packing: bool = False
    debug_collator: bool = False
    # Fraction of collator batches the debug collator prints (the first batch
    # always prints). `--debug-collator PROB` sets it.
    debug_probability: float = Field(0.001, ge=0.0, le=1.0)


class ValidationConfig(_Strict):
    """In-loop validation behavior (formerly `eval:`)."""
    strategy: IntervalStrategy = IntervalStrategy.EPOCH
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: PositiveInt | None = 3
    stop_on_nan: bool = True


class SaveConfig(_Strict):
    strategy: IntervalStrategy = IntervalStrategy.EPOCH
    total_limit: PositiveInt | None = 3
    output_dir: str


class RuntimeConfig(_Strict):
    num_train_epochs: PositiveInt = 3
    seed: int = 42
    logging_steps: PositiveInt = 1
    dataloader_num_workers: int = Field(4, ge=0)
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard"])
    fsdp: bool | dict[str, Any] = False


class GenerationConfig(_Strict):
    """Post-training generation parameters.

    The block is optional at the YAML level (``cfg.generation`` may be
    ``None`` — see :class:`TrainingConfig`); when present, ``input_path``
    is the trigger field and is required. Used by the standalone
    ``generate`` entry point and by ``train.py`` for post-training
    evaluation.
    """
    input_path: str
    max_new_tokens: PositiveInt = 256
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    do_sample: bool = False
    batch_size: PositiveInt = 8
    output_filename: str = "predictions.jsonl"
    # When > 0, probabilistically log decoded prompt + generated text for
    # ~debug_probability of generation batches. Mirrors trainer.debug_collator
    # so the prediction path is as inspectable as training is. 0.0 = off.
    debug_probability: float = Field(0.0, ge=0.0, le=1.0)


# --- Root model ---------------------------------------------------------------


class TrainingConfig(BaseModel):
    """Root unified training config."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    model: ModelConfig
    data: DataConfig
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    save: SaveConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig | None = None


# --- Loader -------------------------------------------------------------------

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

# The reference configs' ${VAR} placeholders and the CLI flag that supplies each.
CONFIG_VARIABLE_FLAGS = {
    "BASE_MODEL": "--base-model",
    "DATA_ROOT": "--data-root",
    "EVAL_DATA": "--eval-data",
    "RUN_DIR": "--run-dir",
}


def _resolve_env_vars(value: Any, variables: Mapping[str, str] | None = None) -> Any:
    """Recursively resolve ${VAR} in any string, from ``variables`` first and
    then the environment. Raise on unset vars."""
    variables = variables or {}
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            name = match.group(1)
            if name in variables:
                return variables[name]
            if name not in os.environ:
                flag = CONFIG_VARIABLE_FLAGS.get(name)
                how = f"pass {flag} or set {name}" if flag else f"set {name}"
                raise KeyError(f"Environment variable ${{{name}}} is not set ({how})")
            return os.environ[name]
        return _VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v, variables) for v in value]
    return value


def load_training_config(
    path: str | Path,
    variables: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> TrainingConfig:
    """Load YAML, apply ``overrides``, resolve ${VAR}, validate against TrainingConfig.

    ``overrides`` maps dotted field paths (``"data.train_path"``) to values set
    before ${VAR} resolution, so a placeholder in an overridden field needs no
    value. A ``generation.*`` override applies only if the YAML has a
    ``generation`` block. ``variables`` supply ${VAR} values ahead of the
    environment (the CLI's ``--base-model`` / ``--data-root`` / ...).
    """
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    for dotted, val in (overrides or {}).items():
        section, field = dotted.split(".", 1)
        if section == "generation" and not isinstance(raw.get("generation"), dict):
            continue
        raw.setdefault(section, {})[field] = val
    resolved = _resolve_env_vars(raw, variables)
    return TrainingConfig.model_validate(resolved)
