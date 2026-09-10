# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual modeling classes (HF backend).

Inherits from :class:`GraniteMoeHybridPreTrainedModel` directly. The SR
model builds on top of the upstream HF parent with no intermediate
routing or adapter-switching layers.

Two streams still live inside the decoder layers (``h_base`` frozen,
``h_adapt`` trainable, with per-layer cross-stream injection); see
``decoder_hf.py`` and ``attention_hf.py``. After the last layer the
adapter stream is normed and projected directly — no final-step merge.

Usage::

    from shadow_residual.shadow_residual import (
        ShadowResidualForCausalLM,
    )
    model = ShadowResidualForCausalLM(config)
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import transformers
from packaging.version import parse as _parse_version
from transformers import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridPreTrainedModel,
    GraniteMoeHybridRMSNorm,
    GraniteMoeHybridRotaryEmbedding,
)

# transformers 5.9.0 renamed `input_embeds` -> `inputs_embeds` and dropped the
# unused `cache_position` kwarg in `create_causal_mask`.
_TRANSFORMERS_GE_5_9 = _parse_version(transformers.__version__) >= _parse_version("5.9.0")

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig

from .cross_stream import CrossStream, CrossStreamLinear
from .decoder_hf import ShadowResidualDecoderLayer


def _is_shadow_residual_enabled(config: ShadowResidualConfig) -> bool:
    return bool(getattr(config, "shadow_residual", False))


def _any_cross_stream_wrapped(layers) -> bool:
    """Has any decoder layer's cross-stream site been wrapped by PEFT?

    A bare (unwrapped) cross-stream site is a no-op (returns zeros): both the
    ``"lora"``-type :class:`CrossStream` (frozen zero-init linear) and the
    ``"linear"``-type :class:`CrossStreamLinear` (zero-init H×H matrix). When
    an adapter is attached, PEFT replaces the tap attribute with a wrapper — a
    stock ``lora.Linear`` (lora type, via ``target_modules``) or a
    ``ModulesToSaveWrapper`` (linear type, via ``modules_to_save``). Neither
    wrapper is a :class:`CrossStream` / :class:`CrossStreamLinear` instance (each
    holds the original module underneath).

    A layer may carry several tapped sites (see
    :data:`cross_stream.CROSS_STREAM_TAPS`); any one being wrapped is enough. This
    check decides whether the dual-stream forward runs. Bare sites everywhere →
    bare single-stream (unadapted base); any wrapped site → dual-stream forward.
    Crucially, an unwrapped ``CrossStreamLinear`` on a bare base must NOT be
    mistaken for an active adapter, or a no-adapter model would wrongly take the
    dual-stream path — so both bare types are excluded here.
    """
    for layer in layers:
        for tap_name in getattr(layer, "cross_stream_tap_names", ()):
            cs = getattr(layer, tap_name, None)
            if cs is not None and not isinstance(cs, (CrossStream, CrossStreamLinear)):
                return True
    return False


def _adapters_globally_disabled(modules) -> bool:
    """True iff a top-level ``disable_adapter()`` is currently in effect.

    When the user wraps a forward in ``with peft_model.disable_adapter():`` every
    LoRA layer's ``disable_adapters`` flag is set; the SR model then takes the
    bare single-stream path (the unadapted base) instead of dual-stream. On a bare
    (unwrapped) model there are no LoRA layers, so this returns False (and the
    absence of a wrapped cross_stream selects the bare path anyway).
    """
    from peft.tuners.lora.layer import LoraLayer

    for m in modules:
        if isinstance(m, LoraLayer):
            return bool(getattr(m, "disable_adapters", False))
    return False


class ShadowResidualPreTrainedModel(GraniteMoeHybridPreTrainedModel):
    """``PreTrainedModel`` base class for shadow-residual models."""

    config_class = ShadowResidualConfig
    base_model_prefix = "model"
    _no_split_modules = ["ShadowResidualDecoderLayer"]
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # The "lora"-type cross-stream site is a frozen zero-init nn.Linear. The
        # parent's _init_weights treats it as a generic linear and fills it with a
        # normal distribution — which would make base_layer(h_base) nonzero and
        # pollute the adapter stream (breaking the frozen-base invariant and the
        # cross-stream "pure B·A" semantics). Re-zero it after the parent runs so
        # post_init() leaves it as an exact no-op. Kept frozen.
        #
        if isinstance(module, CrossStream):
            nn.init.zeros_(module.weight)
            module.weight.requires_grad_(False)
        # The "linear"-type site (CrossStreamLinear) stays trainable, but the
        # parent's _init_weights re-fills its inner `proj` nn.Linear with a normal
        # distribution, clobbering the zero-init done in CrossStreamLinear.__init__.
        # Re-zero the H×H weight so the injection is again exactly zero at
        # post_init() (adapter stream starts base-identical); leave it trainable
        # (no requires_grad flip — unlike the frozen CrossStream above).
        elif isinstance(module, CrossStreamLinear):
            nn.init.zeros_(module.proj.weight)


class ShadowResidualModel(ShadowResidualPreTrainedModel):
    """Shadow-residual decoder model.

    Constructs ``embed_tokens`` / ``layers`` / ``norm`` / ``rotary_emb``
    directly on top of :class:`GraniteMoeHybridPreTrainedModel`. There
    is no adapter routing or switching layer — SR uses standard PEFT
    LoRA with a frozen base stream and cross-stream injection.
    """

    def __init__(self, config: ShadowResidualConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx,
        )
        self.embedding_multiplier = config.embedding_multiplier

        self.layers = nn.ModuleList(
            ShadowResidualDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )

        self.norm = GraniteMoeHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.position_embedding_type = getattr(config, "position_embedding_type", "rope")
        if self.position_embedding_type == "rope":
            self.rotary_emb = GraniteMoeHybridRotaryEmbedding(config=config)
        else:
            self.rotary_emb = None

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_length,
                device=device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        embed_dtype = self.embed_tokens.weight.dtype
        mask_shape_proxy = inputs_embeds if inputs_embeds is not None else torch.empty(
            batch_size, seq_length, 1, device=device, dtype=embed_dtype,
        )
        mask_kwargs = {
            "config": self.config,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if _TRANSFORMERS_GE_5_9:
            mask_kwargs["inputs_embeds"] = mask_shape_proxy
        else:
            mask_kwargs["input_embeds"] = mask_shape_proxy
            mask_kwargs["cache_position"] = cache_position
        causal_mask = create_causal_mask(**mask_kwargs)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds * self.embedding_multiplier

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(
                inputs_embeds, position_ids=position_ids,
            )

        # ---- Decoder loop: dual-stream (adapter) vs. bare single-stream ----
        # An SR adapter is "active" iff a cross_stream site has been PEFT-wrapped
        # AND adapters are not globally disabled (top-level disable_adapter()).
        #   * active   → shared-base-K/V dual-stream over a stacked [2,B,S,H]
        #                tensor. Under training with gradient checkpointing the
        #                GradientCheckpointingLayer.__call__ checkpoints each layer
        #                (the dual-stream forward is pure, so recompute is exact);
        #                otherwise the same forward runs directly.
        #   * inactive → bare single-stream = the unadapted base model (a plain
        #                causal LM). Also the path a top-level disable_adapter()
        #                collapses to.
        adapter_active = _any_cross_stream_wrapped(self.layers) and not _adapters_globally_disabled(self.modules())

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if adapter_active:
            hs = torch.stack([inputs_embeds, inputs_embeds.clone()], dim=0)  # [2,B,S,H]
            checkpointing = self.gradient_checkpointing and self.training
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hs[0],)
                if checkpointing:
                    decoder_layer.gradient_checkpointing = True
                    hs = decoder_layer(
                        hs,
                        attention_mask=causal_mask,
                        past_key_values=None,   # training: no cache
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                else:
                    decoder_layer.gradient_checkpointing = False
                    hs = decoder_layer(
                        hs,
                        attention_mask=causal_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
            # Only norm(h_adapt) reaches the LM head; the base→adapter coupling
            # already happened per-layer via cross_stream(h_base).
            hidden_states = self.norm(hs[1])
        else:
            # Bare single-stream: no adapter → exactly the unadapted base model.
            h = inputs_embeds
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (h,)
                h = decoder_layer.forward_bare(
                    h,
                    attention_mask=causal_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            hidden_states = self.norm(h)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ShadowResidualForCausalLM(ShadowResidualPreTrainedModel, GenerationMixin):
    """Causal-LM head for the shadow-residual model."""

    config_class = ShadowResidualConfig

    def __init__(self, config: ShadowResidualConfig):
        super().__init__(config)
        self.model = ShadowResidualModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight tying is config-driven so the same model works for tied bases
        # (granite-4.1, tie_word_embeddings=True — lm_head aliases embed_tokens)
        # and untied bases (granite-4.2, tie_word_embeddings=False — lm_head is a
        # separately-trained matrix present in the checkpoint). Set as an
        # INSTANCE attribute (not a class literal) so it reflects this config:
        # HF reads _tied_weights_keys in tie_weights()/from_pretrained's
        # missing-key logic. Empty dict => nothing is aliased and post_init's
        # tie_weights() leaves the distinct lm_head untouched. This runs on every
        # rank (incl. meta under FSDP), so the parameter-sharing structure is
        # identical across ranks before the FSDP wrap — see the factory's
        # tie_weights() call for the FSDP-symmetry rationale.
        if getattr(config, "tie_word_embeddings", True):
            self._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        else:
            self._tied_weights_keys = {}
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits / self.config.logits_scaling

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ShadowResidualPreTrainedModel",
    "ShadowResidualModel",
    "ShadowResidualForCausalLM",
    "_is_shadow_residual_enabled",
]
