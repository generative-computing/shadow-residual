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
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeParallelExperts,
    GraniteMoeTopKGating,
)
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridRMSNorm,
)

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig

from ._stream_gated_linear import _StreamGatedLinear, _base_only
from .attention_hf import ShadowResidualAttention
from .cross_stream import CrossStream


class ShadowResidualMLP(nn.Module):
    """Per-layer FFN for Shadow Residual — dense SwiGLU **or** frozen sparse MoE.

    Two modes, selected once at construction by ``config.num_local_experts``:

    * **Dense** (``num_local_experts == 0``, Granite 4.x): a SwiGLU MLP mirroring
      :class:`GraniteMLP` (gate/up/down projections + act_fn) built from
      :class:`_StreamGatedLinear` so PEFT wraps them with stock LoRA. :meth:`forward`
      runs the wrapped projections (adapter stream, LoRA delta fires);
      :meth:`forward_base_only` runs the raw base weights (base stream, no delta).

    * **MoE** (``num_local_experts > 0``, Granite 5.0 / ``granitemoe``): a frozen
      sparse expert bank mirroring :class:`GraniteMoeMoE` — a top-k router plus two
      :class:`GraniteMoeParallelExperts`. The experts are **not** LoRA targets in
      the attention-only iteration, so they are plain (non-``_StreamGatedLinear``)
      frozen modules with no ``base_layer`` indirection. :meth:`forward` and
      :meth:`forward_base_only` therefore run the **same** frozen MoE computation on
      their own input — the base stream routes on ``normed_base`` (exactly what the
      stock model sees, so it stays bit-identical) and the adapter stream routes
      independently on ``normed_adapt``. Both are pure functions of their input, so
      gradient-checkpoint recompute is exact (routing is deterministic given the
      input; the only cost is the router's data-dependent ``expert_size.tolist()``
      sync, which also makes MoE mode a ``torch.compile`` graph-break — SR does not
      compile the decoder, so this is a documented limitation, not a correctness
      issue).

    Routing is split into :meth:`_route` (the top-k decision) and
    :meth:`_apply_experts` (the frozen expert bank over a *given* partition) so the
    decoder **shares** the base stream's routing with the adapter stream by
    default (``config.share_moe_routing``, default True), or routes each stream
    independently when set False. Shared routing constrains the adapter to the
    base's expert selection and halves the per-layer router work; independent
    routing lets the adapter steer *which* experts fire. Either way the base
    stream routes on ``normed_base`` — so the frozen-base invariant holds in both
    modes. See :class:`ShadowResidualDecoderLayer`.
    """

    def __init__(self, config: ShadowResidualConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_local_experts = int(getattr(config, "num_local_experts", 0) or 0)
        self.is_moe = self.num_local_experts > 0
        self.act_fn = ACT2FN[config.hidden_act]

        if self.is_moe:
            # Frozen sparse expert bank (mirrors GraniteMoeMoE). `intermediate_size`
            # is the per-expert FFN width; `input_linear` fuses gate⊕up at 2*width
            # (SwiGLU chunk happens in forward), `output_linear` maps back to hidden.
            self.intermediate_size = config.intermediate_size
            self.num_experts_per_tok = int(config.num_experts_per_tok)
            self.router = GraniteMoeTopKGating(
                input_size=self.hidden_size,
                num_experts=self.num_local_experts,
                top_k=self.num_experts_per_tok,
            )
            self.input_linear = GraniteMoeParallelExperts(
                self.num_local_experts, self.hidden_size, self.intermediate_size * 2,
            )
            self.output_linear = GraniteMoeParallelExperts(
                self.num_local_experts, self.intermediate_size, self.hidden_size,
            )
        else:
            self.intermediate_size = config.shared_intermediate_size
            self.gate_proj = _StreamGatedLinear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = _StreamGatedLinear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = _StreamGatedLinear(self.intermediate_size, self.hidden_size, bias=False)

    def _route(self, x: torch.Tensor):
        """Top-k routing decision for a ``[B, S, H]`` (or already-flattened) input.

        Returns the three fields the expert computation consumes:
        ``(batch_index, batch_gates, expert_size)``. Split out from
        :meth:`_apply_experts` so a caller can route **once** on one stream and
        apply the resulting partition to another (shared-routing mode — see
        :class:`ShadowResidualDecoderLayer`). Deterministic given ``x``.
        """
        x = x.reshape(-1, x.shape[-1])
        _, batch_index, batch_gates, expert_size, _ = self.router(x)
        return batch_index, batch_gates, expert_size

    def _apply_experts(
        self, x_flat: torch.Tensor, batch_index, batch_gates, expert_size,
        *, bsz: int, length: int,
    ) -> torch.Tensor:
        """Run the frozen expert bank over a **given** routing partition.

        ``x_flat`` is the ``[B*S, H]`` token matrix to gather from; the routing
        tuple may have been computed on ``x_flat`` itself (independent routing)
        or on the *other* stream's hidden state (shared routing). Mirrors the
        expert body of :meth:`GraniteMoeMoE.forward`: gather ``x_flat[batch_index]``
        → fused gate⊕up (``input_linear``) → SwiGLU → ``output_linear`` → scale by
        the router gates → scatter back. Frozen and delta-free.
        """
        emb_size = x_flat.shape[-1]
        expert_inputs = x_flat[batch_index]
        hidden_states = self.input_linear(expert_inputs, expert_size)
        chunked = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act_fn(chunked[0]) * chunked[1]
        expert_outputs = self.output_linear(hidden_states, expert_size)
        expert_outputs = expert_outputs * batch_gates[:, None]

        zeros = torch.zeros(
            (bsz * length, emb_size), dtype=expert_outputs.dtype, device=expert_outputs.device,
        )
        out = zeros.index_add(0, batch_index, expert_outputs)
        return out.view(bsz, length, emb_size)

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen sparse-MoE forward — mirrors :meth:`GraniteMoeMoE.forward`.

        Self-routing: routes on ``x`` and applies that partition to ``x``. Used by
        the bare/base-only paths and by the adapter stream under independent
        routing (``share_moe_routing=False``). Shared routing (the default) does
        NOT call this for the adapter stream — the decoder calls :meth:`_route`
        once on the base stream and :meth:`_apply_experts` on each stream directly.
        """
        bsz, length, _ = x.shape
        routing = self._route(x)
        return self._apply_experts(
            x.reshape(-1, x.shape[-1]), *routing, bsz=bsz, length=length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_moe:
            return self._moe(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def forward_base_only(self, x: torch.Tensor) -> torch.Tensor:
        """MLP on the frozen base weights, LoRA delta bypassed entirely.

        Dense mode: calls each projection's ``base_layer`` directly — a pure
        function of ``x`` with no adapter-state mutation, so gradient-checkpoint
        recompute reproduces it exactly. Falls back to the module itself on a bare
        (unwrapped) model.

        MoE mode: the expert bank is frozen and carries no delta, so this is
        identical to :meth:`forward` — a plain frozen MoE pass on ``x``.
        """
        if self.is_moe:
            return self._moe(x)
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
        # MoE-only: route once on the base stream and reuse that partition for the
        # adapter stream (vs. routing each stream independently). ON by default;
        # only meaningful when the MLP is the sparse expert bank. train.py records
        # it into adapter_config.json and the serving path reads it back, so a
        # loaded adapter's routing mode matches how it was trained.
        # Fallback False is the "attribute genuinely absent" guard, NOT the
        # feature default — build_sr_config always sets share_moe_routing (default
        # True). Only a config constructed outside build_sr_config would fall back.
        self.share_moe_routing = bool(getattr(config, "share_moe_routing", False))

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

        if self.mlp.is_moe and self.share_moe_routing:
            # Route once on the base stream; run both streams' experts through
            # that partition. The base stream still routes on normed_base (so it
            # stays bit-identical to stock); the adapter stream reuses the base's
            # expert selection rather than routing on normed_adapt.
            length, emb = normed_base.shape[1], normed_base.shape[2]
            routing = self.mlp._route(normed_base)
            mlp_base = self.mlp._apply_experts(
                normed_base.reshape(-1, emb), *routing, bsz=bsz, length=length,
            )
            mlp_adapt = self.mlp._apply_experts(
                normed_adapt.reshape(-1, emb), *routing, bsz=bsz, length=length,
            )
        else:
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
