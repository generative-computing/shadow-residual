# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual modeling classes (HF backend).

Inherits from :class:`GraniteMoeHybridPreTrainedModel` directly. The
historical inheritance chain went through ``GraniteSwitchModel`` to pick
up SwitchedLoRA / SingleSwitch scaffolding, but the SR + PEFT training
path never used any of that — it carried the merge gate
``torch.where(adapter_indices > 0, h_adapt, h_base)`` purely as
GraniteSwitch routing baggage, and the gate severed autograd from the
LoRA delta when no control tokens were in the data. Phase 1 of the
refactor dropped the gate; this is Phase 2, severing the GraniteSwitch
inheritance entirely and building the SR model directly on top of the
upstream HF parent.

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

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig

from .cross_stream import CrossStream
from .decoder_hf import ShadowResidualDecoderLayer


def _is_shadow_residual_enabled(config: GraniteSwitchConfig) -> bool:
    return bool(getattr(config, "shadow_residual", False))


def _any_cross_stream_wrapped(layers) -> bool:
    """Has any decoder layer's ``cross_stream`` site been wrapped by PEFT?

    The bare :class:`CrossStream` site is a no-op (returns zeros). When
    PEFT installs a :class:`CrossStreamLora` (because the user listed
    ``"cross_stream"`` in ``target_modules``), it replaces the
    ``cross_stream`` attribute with the wrapper. The wrapper is not a
    :class:`CrossStream` instance — its ``base_layer`` is.

    This check decides whether the dual-stream forward needs to run. A
    bare ``CrossStream`` everywhere → single-stream early-exit; any
    wrapped site anywhere → dual-stream forward.
    """
    for layer in layers:
        cs = getattr(layer, "cross_stream", None)
        if cs is None:
            continue
        if not isinstance(cs, CrossStream):
            return True
    return False


class ShadowResidualPreTrainedModel(GraniteMoeHybridPreTrainedModel):
    """``PreTrainedModel`` base class for shadow-residual models.

    We keep ``GraniteSwitchConfig`` as the config class for now (its
    extra fields are harmless when unused, and existing checkpoints
    round-trip). A dedicated minimal config can come later.
    """

    config_class = GraniteSwitchConfig
    base_model_prefix = "model"
    _no_split_modules = ["ShadowResidualDecoderLayer"]
    _is_stateful = True


class ShadowResidualModel(ShadowResidualPreTrainedModel):
    """Shadow-residual decoder model.

    Constructs ``embed_tokens`` / ``layers`` / ``norm`` / ``rotary_emb``
    directly on top of :class:`GraniteMoeHybridPreTrainedModel`. There
    is no :class:`SingleSwitch`, no ``adapter_token_ids`` buffer, and
    no SwitchedLoRA wrappers — those were GraniteSwitch scaffolding the
    SR + PEFT path never used.
    """

    def __init__(self, config: GraniteSwitchConfig):
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

        # ---- Decoder loop: adapter-only early-exit vs. dual-stream ----
        # Two ways to enter dual-stream mode:
        #   1. A CrossStream site has been wrapped by PEFT — the standard
        #      SR path with per-layer base→adapter injection.
        #   2. ``shared_base_kv=True`` is set on the config — "weak SR":
        #      both streams flow in parallel, K/V is computed once from
        #      ``normed_base``, but no cross_stream wrapping → bare
        #      CrossStream returns zeros so the per-layer merge is a
        #      no-op. Useful for ablating the W_cross contribution
        #      while keeping the frozen-base K/V topology.
        # Otherwise we take the adapter-only single-stream early-exit
        # (one Q/O/MLP pass per layer with the LoRA delta firing).
        cross_stream_active = _any_cross_stream_wrapped(self.layers)
        shared_base_kv = bool(getattr(self.config, "shared_base_kv", False))

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if cross_stream_active or shared_base_kv:
            # When shared_base_kv is on, the adapter stream attends to
            # K/V derived from normed_base (single cache: past_key_values).
            # The disjoint per-stream base cache is unused in that mode.

            # Disjoint base K/V cache — frozen-base invariant requires
            # the base stream's K/V to be adapter-free, so it can't share
            # storage with the adapter cache. Bind its lifetime to the
            # adapter cache so it persists across generate() decode steps:
            # transformers' generate loop threads the same past_key_values
            # object through every step, so we keep the same base cache as
            # long as the same adapter cache object keeps coming back; a
            # fresh adapter cache (new generate() call, or training step)
            # triggers a fresh base cache.
            if use_cache and not shared_base_kv:
                if (
                    getattr(self, "_pkv_base_owner", None) is not past_key_values
                    or getattr(self, "_pkv_base", None) is None
                ):
                    self._pkv_base = DynamicCache(config=self.config)
                    self._pkv_base_owner = past_key_values
                past_key_values_base = self._pkv_base
            else:
                past_key_values_base = None

            h_base = inputs_embeds
            h_adapt = inputs_embeds.clone()
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (h_base,)

                # Manually checkpoint the layer under training (NOT via a
                # GradientCheckpointingLayer __call__, which would trigger PEFT's
                # fragile hook/guard machinery). The layer's own forward re-runs on
                # recompute and re-enters offset_recovery_enabled(), so the gated
                # variants recover alora_offsets from their cache on recompute.
                if self.gradient_checkpointing and self.training:
                    def _layer_call(hb, ha, layer=decoder_layer, _kw=kwargs):
                        return layer(
                            hb, ha,
                            cross_stream_active=True,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            past_key_values_base=past_key_values_base,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **_kw,
                        )
                    layer_outputs = self._gradient_checkpointing_func(
                        _layer_call, h_base, h_adapt,
                    )
                else:
                    layer_outputs = decoder_layer(
                        h_base, h_adapt,
                        cross_stream_active=True,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        past_key_values_base=past_key_values_base,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

                h_base, h_adapt = layer_outputs[0]

                if output_attentions and len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)

            # The base→adapter coupling already happened per-layer via
            # cross_stream(h_base) injection inside each decoder layer
            # (see decoder_hf.py). After the last layer we norm + project
            # the adapter stream directly. The previous final-step gate
            # `torch.where(adapter_indices > 0, h_adapt, h_base)` was
            # vestigial GraniteSwitch routing — see Phase 1 of the SR
            # refactor.
            hidden_states = self.norm(h_adapt)
        else:
            # Adapter-only early-exit: no h_base maintained.
            h_adapt = inputs_embeds
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (h_adapt,)

                # Manual checkpoint (see dual-stream branch). The layer forward
                # re-enters offset_recovery_enabled() on recompute so the gated
                # variants recover alora_offsets from cache.
                if self.gradient_checkpointing and self.training:
                    def _layer_call(ha, layer=decoder_layer, _kw=kwargs):
                        return layer(
                            None, ha,
                            cross_stream_active=False,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **_kw,
                        )
                    layer_outputs = self._gradient_checkpointing_func(
                        _layer_call, h_adapt,
                    )
                else:
                    layer_outputs = decoder_layer(
                        None, h_adapt,
                        cross_stream_active=False,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

                _, h_adapt = layer_outputs[0]

                if output_attentions and len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(h_adapt)

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

    config_class = GraniteSwitchConfig

    def __init__(self, config: GraniteSwitchConfig):
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
