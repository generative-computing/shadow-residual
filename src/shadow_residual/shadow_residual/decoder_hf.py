# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual decoder layer for Granite Switch (HF backend).

Maintains two parallel hidden state streams (base and adapter):

- Attention: see :class:`ShadowResidualAttention`. Q and O are stream-
  gated (``"base"`` context skips LoRA delta); K/V is single-shared and
  picks up its LoRA delta (documented exception).
- MLP: ``mlp.gate_proj`` / ``mlp.up_proj`` / ``mlp.down_proj`` are
  :class:`_StreamGatedLinear` instances nested under ``self.mlp``,
  called once per stream inside the appropriate :func:`stream_context`.
  Base call → no delta; adapter call → delta fires. The ``mlp.``
  namespace ensures PEFT saves adapter keys with the same structure as
  upstream HF Granite (``layers.{i}.mlp.gate_proj.lora_A.weight``).

Frozen-base invariant
---------------------
With cross-stream active, the base stream's Q/O/MLP contributions are
bit-identical to the unadapted base model. K/V is the single documented
exception.

Cross-stream injection (= the merge)
------------------------------------
At the end of each layer, the cross-stream site couples base → adapter:
``h_adapt += W_cross(h_base)``.  ``W_cross`` is a no-op
:class:`CrossStream` site until PEFT installs a :class:`CrossStreamLora`
wrapper.  This *is* the per-layer base→adapter merge — it's the reason
to maintain a frozen-base reference at all.

Adapter-only early-exit
-----------------------
When the model-level gate determines no :class:`CrossStream` site has
been wrapped, the SR forward runs the **adapter path only** — no base
stream is computed. This is the "LoRA-on-SR-architecture without
cross-stream" mode and reduces to plain LoRA on Granite Switch.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridRMSNorm,
)

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig

from ._stream_context import offset_recovery_enabled, stream_context
from ._stream_gated_linear import _StreamGatedLinear
from .attention_hf import ShadowResidualAttention
from .cross_stream import CrossStream


class ShadowResidualMLP(nn.Module):
    """Stream-gated SwiGLU MLP for Shadow Residual.

    Mirrors :class:`GraniteMLP` structure (gate/up/down projections + act_fn)
    but uses :class:`_StreamGatedLinear` so that the LoRA delta is suppressed
    in the base stream context and active in the adapter stream context.
    """

    def __init__(self, config: GraniteSwitchConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.shared_intermediate_size
        self.gate_proj = _StreamGatedLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = _StreamGatedLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = _StreamGatedLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class ShadowResidualDecoderLayer(nn.Module):
    """Decoder layer with shadow residual streams (base + adapter).

    NOT a :class:`GradientCheckpointingLayer` — deliberately. Making it one
    triggers PEFT's checkpoint-aware aLoRA hook machinery in
    ``_enable_peft_forward_hooks``, whose ``register_full_backward_hook`` cleanup
    does not fire for this layer's TUPLE output, leaving stale handles that raise
    "Multiple invocations of PEFT forward hooks" on the trainer's mid-training
    eval pass (a no-backward forward). Instead the model loop checkpoints these
    layers manually (``ShadowResidualModel.forward``), and the gated LoRA variants
    recover ``alora_offsets`` across the recompute boundary via their own cache
    (scoped by ``offset_recovery_enabled()`` set inside ``forward`` below, so it is
    active on the recompute pass too). That avoids PEFT's hook/guard entirely.
    """

    def __init__(self, config: GraniteSwitchConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier
        self.layer_type = "attention"

        self.self_attn = ShadowResidualAttention(config, layer_idx)

        self.mlp = ShadowResidualMLP(config)

        self.input_layernorm = GraniteMoeHybridRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GraniteMoeHybridRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

        # Cross-stream site — the per-layer base→adapter merge point.
        # Bare CrossStream is a no-op (returns zeros); PEFT installs a
        # :class:`CrossStreamLora` wrapper here at training time when
        # ``"cross_stream"`` is in ``target_modules``. Without an adapter,
        # contributes zero to ``h_adapt``.
        self.cross_stream = CrossStream(config.hidden_size)

    def forward(
        self,
        h_base: Optional[torch.Tensor],
        h_adapt: torch.Tensor,
        *,
        cross_stream_active: bool,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        past_key_values_base: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple:
        """Forward with shadow-residual streams.

        Args:
            h_base: base hidden state ``[B, S, H]``, or ``None`` in
                adapter-only early-exit mode.
            h_adapt: adapter hidden state ``[B, S, H]``. In adapter-only
                mode this is the only stream.
            cross_stream_active: if ``False``, runs the adapter path
                only and returns ``((None, h_adapt), …)``.

        Returns:
            ``((h_base, h_adapt), [attn_weights], [present_kv])``.
            ``h_base`` is ``None`` in the adapter-only early-exit path.
        """
        del position_ids, kwargs  # not used by this layer

        # Enable aLoRA-offset recovery for the duration of this layer's forward.
        # Set HERE (inside the layer forward), not in the model loop, so it is
        # also active when gradient checkpointing re-runs this forward during the
        # backward recompute — an outer (model-level) context would be gone by
        # then, exactly like the stream_context tags below. This lets the gated
        # LoRA variants restore the aLoRA offsets that PEFT's hook fails to
        # re-inject on recompute (see _stream_context.py / the variants).
        with offset_recovery_enabled():
            return self._forward_impl(
                h_base, h_adapt,
                cross_stream_active=cross_stream_active,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                past_key_values_base=past_key_values_base,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

    def _forward_impl(
        self,
        h_base: Optional[torch.Tensor],
        h_adapt: torch.Tensor,
        *,
        cross_stream_active: bool,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        past_key_values_base: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple:
        """Body of :meth:`forward`, run under the offset-recovery context."""
        if not cross_stream_active:
            # ---- Adapter-only early-exit ----
            # Single Q/K/V/O pass under "adapter" context (LoRA delta
            # fires). MLP under "adapter" context.
            normed = self.input_layernorm(h_adapt)
            _, o_adapt, present_key_values = self.self_attn(
                normed_base=normed,
                normed_adapt=None,
                cross_stream_active=False,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            h_adapt = h_adapt + o_adapt * self.residual_multiplier

            normed = self.post_attention_layernorm(h_adapt)
            with stream_context("adapter"):
                mlp_adapt = self.mlp(normed)
            h_adapt = h_adapt + mlp_adapt * self.residual_multiplier

            outputs = ((None, h_adapt),)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (present_key_values,)
            return outputs

        # ---- Dual-stream forward (cross-stream active) ----
        bsz = h_base.shape[0]

        # Attention block. Per-stream layernorm is computed jointly to
        # match the original SR fused-batch pattern; the two halves are
        # then handed to the attention module with explicit context
        # gating handled inside ShadowResidualAttention.
        normed = self.input_layernorm(torch.cat([h_base, h_adapt], dim=0))
        normed_base, normed_adapt = normed[:bsz], normed[bsz:]

        o_base, o_adapt, present_key_values = self.self_attn(
            normed_base=normed_base,
            normed_adapt=normed_adapt,
            cross_stream_active=True,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_key_values_base=past_key_values_base,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        h_base = h_base + o_base * self.residual_multiplier
        h_adapt = h_adapt + o_adapt * self.residual_multiplier

        # MLP block. Per-stream layernorm fused; per-stream MLP under
        # the matching stream context.
        normed = self.post_attention_layernorm(torch.cat([h_base, h_adapt], dim=0))
        normed_base, normed_adapt = normed[:bsz], normed[bsz:]

        with stream_context("base"):
            mlp_base = self.mlp(normed_base)
        with stream_context("adapter"):
            mlp_adapt = self.mlp(normed_adapt)

        h_base = h_base + mlp_base * self.residual_multiplier
        h_adapt = h_adapt + mlp_adapt * self.residual_multiplier

        # End-of-layer cross-stream injection: base → adapter.
        # This IS the per-layer merge — the reason the frozen-base
        # reference exists at all.
        h_adapt = h_adapt + self.cross_stream(h_base)

        outputs = ((h_base, h_adapt),)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (present_key_values,)
        return outputs
