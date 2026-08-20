# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual decoder layer (HF backend) — shared-base-K/V only.

Two forward paths:

Dual-stream (adapter present)
-----------------------------
Maintains two hidden-state streams (base + adapter). The base stream runs on the
frozen base weights via ``base_layer`` (``_base_only`` / ``mlp.forward_base_only``)
— a pure call with no adapter-state mutation — so it is bit-identical to the
unadapted base model. The adapter stream runs the wrapped projections so the LoRA
delta fires (always active — plain LoRA, on every position). K/V is computed once
from the base stream (shared, one cache). At the end of each layer the cross-stream
site couples base → adapter: ``h_adapt += W_cross(h_base)`` — the per-layer
base→adapter merge.

Because the forward is pure (no contextvar / ``disable_adapter`` toggling) and the
layer is a :class:`GradientCheckpointingLayer` with a single stacked
``[2, B, S, H]`` tensor in/out, HF gradient checkpointing drives it via ``__call__``
and recompute is exact — no manual checkpointing, no custom recompute-stable
variant. The same path serves training (checkpointed) and dual-stream inference.

Bare single-stream (no adapter)
-------------------------------
When no PEFT adapter is attached, :meth:`forward_bare` runs one plain
Q/K/V/O/MLP pass — the projections are ordinary ``nn.Linear``, so the output is
exactly the unadapted base model. Selected by the model-level "no adapter active"
branch (see modeling_hf.py).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridRMSNorm,
)

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig

from ._stream_gated_linear import _StreamGatedLinear, _base_only
from .attention_hf import ShadowResidualAttention
from .cross_stream import CrossStream


class ShadowResidualMLP(nn.Module):
    """SwiGLU MLP for Shadow Residual.

    Mirrors :class:`GraniteMLP` (gate/up/down projections + act_fn) but uses
    :class:`_StreamGatedLinear` so PEFT wraps them with stock LoRA. :meth:`forward`
    runs the wrapped projections (adapter stream, delta fires);
    :meth:`forward_base_only` runs the raw base weights (base stream, no delta).
    """

    def __init__(self, config: ShadowResidualConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.shared_intermediate_size
        self.gate_proj = _StreamGatedLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = _StreamGatedLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = _StreamGatedLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def forward_base_only(self, x: torch.Tensor) -> torch.Tensor:
        """MLP on the frozen base weights, LoRA delta bypassed entirely.

        Calls each projection's ``base_layer`` directly — a pure function of ``x``
        with no adapter-state mutation, so gradient-checkpoint recompute reproduces
        it exactly. Falls back to the module itself on a bare (unwrapped) model.
        """
        g = _base_only(self.gate_proj, x)
        u = _base_only(self.up_proj, x)
        return _base_only(self.down_proj, self.act_fn(g) * u)


class ShadowResidualDecoderLayer(GradientCheckpointingLayer):
    """Decoder layer with shadow residual streams (base + adapter), shared K/V.

    * :meth:`forward` — the dual-stream path. Takes and returns a single stacked
      tensor ``[2, B, S, H]`` (base + adapter) and gates the base stream with pure
      ``base_layer`` calls. As a :class:`GradientCheckpointingLayer` with a
      single-tensor input/output and a pure forward, HF gradient checkpointing
      drives it via ``__call__`` and recompute is exact. Serves both checkpointed
      training and dual-stream inference.

    * :meth:`forward_bare` — single-stream forward for a bare (no-adapter) model;
      produces exactly the unadapted base logits.
    """

    def __init__(self, config: ShadowResidualConfig, layer_idx: int):
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

        # Cross-stream site — the per-layer base→adapter merge point. A frozen
        # zero-init nn.Linear (no-op) until PEFT wraps it with stock LoRA when
        # ``"cross_stream"`` is in target_modules. On a bare model it contributes
        # zero to h_adapt.
        self.cross_stream = CrossStream(config.hidden_size)

    def forward(
        self,
        hs: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Dual-stream forward over a stacked ``[2, B, S, H]`` tensor.

        ``hs[0]`` is the base stream, ``hs[1]`` the adapter stream. Returns a single
        stacked ``[2, B, S, H]`` tensor — single tensor in/out is what makes it GCL-
        and checkpoint-friendly. K/V is computed once from the base stream (shared);
        the KV cache is mutated in place inside attention and not returned (the model
        loop keeps its own reference).
        """
        del kwargs
        h_base, h_adapt = hs[0], hs[1]
        bsz = h_base.shape[0]

        normed = self.input_layernorm(torch.cat([h_base, h_adapt], dim=0))
        normed_base, normed_adapt = normed[:bsz], normed[bsz:]

        o_base, o_adapt, _ = self.self_attn(
            normed_base=normed_base,
            normed_adapt=normed_adapt,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            cache_position=cache_position,
        )
        h_base = h_base + o_base * self.residual_multiplier
        h_adapt = h_adapt + o_adapt * self.residual_multiplier

        normed = self.post_attention_layernorm(torch.cat([h_base, h_adapt], dim=0))
        normed_base, normed_adapt = normed[:bsz], normed[bsz:]

        mlp_base = self.mlp.forward_base_only(normed_base)
        mlp_adapt = self.mlp(normed_adapt)
        h_base = h_base + mlp_base * self.residual_multiplier
        h_adapt = h_adapt + mlp_adapt * self.residual_multiplier

        # End-of-layer cross-stream injection: base → adapter (the per-layer merge).
        h_adapt = h_adapt + self.cross_stream(h_base)

        return torch.stack([h_base, h_adapt], dim=0)

    def forward_bare(
        self,
        h: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Single-stream forward for a bare (no-adapter) model.

        One plain Q/K/V/O/MLP pass on the base weights — exactly the unadapted base
        model. No cross-stream contribution (bare CrossStream returns zero). Returns
        the updated hidden state ``[B, S, H]``.
        """
        normed = self.input_layernorm(h)
        o, _ = self.self_attn.forward_bare(
            normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        h = h + o * self.residual_multiplier
        normed = self.post_attention_layernorm(h)
        h = h + self.mlp(normed) * self.residual_multiplier
        return h
