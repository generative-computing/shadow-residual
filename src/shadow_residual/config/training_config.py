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
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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


#: The cross-stream layer TYPE axis — see `AdapterConfig.cross_stream_type`. Kept a
#: Literal rather than an Enum so YAML values stay plain strings both here and in the
#: saved adapter_config.json. Adding a type here also needs a row in
#: `cross_stream.CROSS_STREAM_TAP_TYPES` (the module dispatch).
CrossStreamType = Literal["lora", "linear", "monarch"]


# --- Sub-models ---------------------------------------------------------------


class _Strict(BaseModel):
    """Base for all sub-models — forbids unknown keys so typos fail loudly."""
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ModelConfig(_Strict):
    base: str = Field(..., description="HF model ID or local path, e.g. ibm-granite/granite-4.1-3b")
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
    val_path: str
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
    # Cross-stream (w-cross) layer TYPE — what kind of module sits at each tapped
    # cross-stream site. ORTHOGONAL to the wiring: the tap NAME is the wiring (see
    # cross_stream.CROSS_STREAM_TAPS, and note the taps themselves are derived from
    # target_modules — there is no separate tap field). Two shapes, mirroring
    # `target_modules` / `alpha`:
    #
    #   - scalar (e.g. `monarch`)  → that type at EVERY built cross-stream tap
    #   - dict[tap, type]          → per-tap, so one model can MIX types
    #                                (e.g. {cross_stream: monarch,
    #                                       cross_stream_post_attn: linear})
    #
    # The tap's numeric parameter always rides in `target_modules[tap]`, and its
    # meaning is type-dependent:
    #   - "lora" (default): the site is a frozen H×H linear made trainable by stock
    #     PEFT LoRA; `target_modules[tap]` is the LoRA rank R.
    #   - "linear": the site is a directly-trainable full H×H matrix (one matmul,
    #     NOT a low-rank bottleneck); `target_modules[tap]` is required by the schema
    #     but does NOT shape the (always H×H) matrix — provenance only.
    #   - "monarch": the site is a two-factor block-diagonal butterfly — FULL RANK
    #     at H·(b + H/b) params, where `target_modules[tap]` is the block count b.
    #     Minimized at b ≈ √H; at H=2560, b=40 gives 266,240 params/layer, which is
    #     exactly LoRA r=52 — the parameter-matched control. That count is a FLOOR,
    #     so parity with an r=32 arm is unreachable.
    #
    # "linear" and "monarch" are NOT LoRA targets (a 3-D block factor cannot be one,
    # and CLAUDE.md forbids _register_custom_module) — they are routed to PEFT's
    # `modules_to_save` (see config/adapters.to_peft_config), which requires the dict
    # form of `target_modules` since that slot carries the per-tap number. Recorded
    # into the saved adapter_config.json; the serving path recovers each non-lora
    # tap's type AND number from the saved tensor SHAPES (see
    # training.generation_utils.read_cross_stream_tap_types_from_adapter).
    cross_stream_type: CrossStreamType | dict[str, CrossStreamType] = "lora"
    # LoRA scaling is `alpha / r`, so per-module RANK alone is not enough to run a
    # parameter-aligned ablation: halving a module's rank doubles its effective
    # scale. Two shapes, mirroring `target_modules`:
    #
    #   - scalar int (or None → 2 * max rank) → one `lora_alpha` for every module
    #   - dict[str, int]                      → per-module alpha (PEFT's
    #                                           `alpha_pattern`); requires the
    #                                           dict form of `target_modules` with
    #                                           EXACTLY the same key set.
    alpha: PositiveInt | dict[str, PositiveInt] | None = None
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
    # MoE bases (Granite 5.0 / granitemoe) only: when true (the default), the
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

    def cross_stream_type_for(self, tap: str) -> str:
        """The layer TYPE of one cross-stream tap — scalar or per-tap form.

        Taps absent from a dict ``cross_stream_type`` are ``"lora"`` (the default),
        so a dict only has to name the taps that differ.
        """
        if isinstance(self.cross_stream_type, str):
            return self.cross_stream_type
        return self.cross_stream_type.get(tap, "lora")

    def cross_stream_taps(self) -> list[str]:
        """The ``cross_stream*`` keys of a dict ``target_modules``, in config order.

        Empty for the scalar form — which cannot express a non-lora type (see
        :meth:`_validate_cross_stream_type`), so the LoRA-only projection in
        ``config.adapters`` is correct there by construction.
        """
        if not isinstance(self.target_modules, dict):
            return []
        return [k for k in self.target_modules if k.startswith("cross_stream")]

    def cross_stream_tap_types_map(self) -> dict[str, dict[str, Any]]:
        """The per-tap ``{"type", "num"}`` map the SR model builder consumes.

        Joins this schema's two axes into the single shape
        :func:`shadow_residual.shadow_residual.cross_stream.build_cross_stream_tap`
        takes: the TYPE from ``cross_stream_type`` and the tap's numeric parameter
        from its ``target_modules`` entry (whose meaning is type-dependent — a LoRA
        rank for ``"lora"``, provenance for ``"linear"``, the block count ``b`` for
        ``"monarch"``). Empty for the scalar ``target_modules`` form, which cannot
        express a non-lora type — the builder then defaults every tap to ``"lora"``.
        """
        return {
            t: {"type": self.cross_stream_type_for(t), "num": self.target_modules[t]}
            for t in self.cross_stream_taps()
        }

    def non_lora_cross_stream_taps(self) -> list[str]:
        """Taps whose type is NOT ``"lora"`` — i.e. PEFT ``modules_to_save`` targets.

        The single computation behind every "this is not a LoRA target" decision:
        `alpha` must not cover them (there is no ``α/r`` off a non-LoRA layer), and
        ``to_peft_config`` must keep them out of ``r`` / ``target_modules`` /
        ``rank_pattern`` and route them to ``modules_to_save`` instead.
        """
        return [t for t in self.cross_stream_taps() if self.cross_stream_type_for(t) != "lora"]

    @model_validator(mode="after")
    def _validate_cross_stream_type(self):
        is_lora_only = self.cross_stream_type == "lora"
        if isinstance(self.cross_stream_type, dict) and not self.cross_stream_type:
            raise ValueError("cross_stream_type dict cannot be empty")
        # A non-lora type needs the DICT form of target_modules with a cross_stream*
        # entry: that slot carries the tap's numeric parameter (a provenance dim for
        # "linear", the block count b for "monarch"), and the tap set itself is derived
        # from target_modules — a type for a tap that is not built there has no effect.
        if not is_lora_only and not isinstance(self.target_modules, dict):
            raise ValueError(
                f"cross_stream_type={self.cross_stream_type!r} requires the dict form "
                "of target_modules with a 'cross_stream' entry "
                "(e.g. {cross_stream: 32, q_proj: 32, ...})."
            )
        if not is_lora_only and not self.cross_stream_taps():
            raise ValueError(
                f"cross_stream_type={self.cross_stream_type!r} requires at least one "
                "cross_stream* entry in target_modules (the tap set is derived from "
                "target_modules, and that entry carries the tap's numeric parameter)."
            )
        # A dict may only name taps that actually exist, or the type would be silently
        # ignored. (No tap-NAME registry check here — cross_stream.py raises on any
        # unregistered cross_stream* name in target_modules, which this is a subset of.)
        if isinstance(self.cross_stream_type, dict):
            known = set(self.cross_stream_taps())
            unknown = sorted(set(self.cross_stream_type) - known)
            if unknown:
                raise ValueError(
                    "a per-tap `cross_stream_type` dict may only name cross-stream "
                    "taps present in `target_modules` (that is where the tap set and "
                    f"each tap's numeric parameter come from); unexpected={unknown} "
                    f"known={sorted(known)}"
                )
        return self

    @model_validator(mode="after")
    def _default_alpha(self):
        # Default alpha to 2 * (max) LoRA rank. alpha is a LoRA-only concept, so a
        # non-lora tap's target_modules value is not a rank (it is a provenance dim or
        # a Monarch block count) and must not inflate alpha — exclude every such tap.
        if self.alpha is None:
            r = self.target_modules
            if isinstance(r, int):
                max_rank = r
            else:
                non_lora = set(self.non_lora_cross_stream_taps())
                ranks = {
                    name: rank for name, rank in r.items() if name not in non_lora
                }
                # Fall back to the full dict when EVERY entry was a non-lora tap (a
                # degenerate but valid "no LoRA targets" config), so alpha still has
                # a number to key off.
                max_rank = max((ranks or dict(r)).values())
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
        # A non-lora cross-stream tap is not a LoRA target (it rides in
        # modules_to_save, not rank_pattern/alpha_pattern), so alpha must NOT carry an
        # entry for it — exclude every such tap from the required LoRA-target key set.
        # (For the "lora" type the taps ARE LoRA targets and must be covered like any
        # other module.)
        target_keys = target_keys - set(self.non_lora_cross_stream_taps())
        if alpha_keys != target_keys:
            missing = sorted(target_keys - alpha_keys)
            extra = sorted(alpha_keys - target_keys)
            raise ValueError(
                "`alpha` dict keys must match the LoRA-target `target_modules` "
                f"keys exactly; missing={missing} unexpected={extra}"
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


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR} in any string. Raise on unset vars."""
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise KeyError(f"Environment variable ${{{name}}} is not set")
            return os.environ[name]
        return _VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load YAML, resolve ${VAR}, validate against TrainingConfig."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    resolved = _resolve_env_vars(raw)
    return TrainingConfig.model_validate(resolved)
