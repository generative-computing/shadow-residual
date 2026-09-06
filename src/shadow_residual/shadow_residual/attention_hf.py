# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual attention (HF backend) — shared-base-K/V only.

Two forward paths:

Shared base K/V dual-stream (the adapter path)
----------------------------------------------
K and V are computed exactly once, from ``normed_base`` (base weights, no LoRA
delta). The adapter stream skips the K/V projections entirely — it only computes
Q (with optional Q LoRA on ``normed_adapt``). Both Q_base and Q_adapt attend
against the same shared K/V; only one KV cache is used. Per-attention-head this
is ``softmax((x_adapt · W_Q)(x_base · W_K)ᵀ / √d)(x_base · W_V)`` for the adapter
stream. K/V LoRA is structurally meaningless here (no place for a K/V delta to
land) and is rejected at build time. The base stream's Q/K/V/O all remain
bit-identical to the unadapted base model.

The base stream is gated by calling each projection's ``base_layer`` directly
(``_base_only``) — a pure, adapter-state-free call. The adapter stream calls the
wrapped module so the LoRA delta fires (always active — plain LoRA, no gating).
This is checkpoint-pure, so the same :meth:`forward` serves both training (under
gradient checkpointing) and inference.

:meth:`forward` is a thin composition of :meth:`forward_base` (which owns the K/V
computation and the single cache update) and :meth:`forward_adapt` (Q only,
against the base's shared K/V). The halves are exposed separately so the decoder
can run the base stream's whole sublayer chain *ahead* of the adapter's when a
cross-stream tap needs it — see ``decoder_hf``.

Bare single-stream (no adapter attached)
----------------------------------------
When the model runs without a PEFT adapter, :meth:`forward_bare` computes one
plain Q/K/V/O pass — the projections are ordinary ``nn.Linear`` (nothing to gate),
so the output is exactly the unadapted base model. Used only by the model-level
"no adapter active" branch (see modeling_hf.py).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig

from ._stream_gated_linear import _StreamGatedLinear, _base_only


class ShadowResidualAttention(nn.Module):
    """Attention with shadow residual streams (base + adapter), shared K/V.

    Two attention calls (one per stream's Q against the shared, base-only K/V).
    The base contribution is bit-identical to the unadapted base model.
    """

    def __init__(self, config: ShadowResidualConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config,
            "projection_head_dim",
            self.hidden_size // self.num_heads,
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.attention_multiplier
        self.is_causal = True  # Required by HF attention backends.
        self.attention_dropout = config.attention_dropout

        # Optional QK-norm (Qwen3) — gated by config.qk_norm.
        self.qk_norm = getattr(config, "qk_norm", False)
        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size

        # Stream-gated linears — PEFT wraps q_proj / o_proj (and MLP) with stock
        # LoRA when listed in target_modules; k_proj / v_proj are never adapted
        # (shared base-only K/V — K/V LoRA is rejected at build time).
        self.q_proj = _StreamGatedLinear(
            self.hidden_size, q_size, bias=config.attention_bias,
        )
        self.k_proj = _StreamGatedLinear(
            self.hidden_size, kv_size, bias=config.attention_bias,
        )
        self.v_proj = _StreamGatedLinear(
            self.hidden_size, kv_size, bias=config.attention_bias,
        )
        self.o_proj = _StreamGatedLinear(
            q_size, self.hidden_size, bias=config.attention_bias,
        )

    def _run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """One attention call against an already-cached K/V."""
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_out, _ = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
        )
        # attn_out: [B, S, num_heads, head_dim] → [B, S, num_heads * head_dim]
        bsz, q_len = attn_out.shape[0], attn_out.shape[1]
        return attn_out.reshape(bsz, q_len, self.num_heads * self.head_dim)

    def _shape_qkv(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Reshape [B,S,H*D] → [B,H,S,D] and apply rope. Returns (q, k, v, cos, sin)."""
        bsz, q_len, _ = q.shape
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        if k is not None:
            k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        if v is not None:
            v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            if k is not None:
                k = self.k_norm(k)

        cos, sin = position_embeddings if position_embeddings is not None else (None, None)
        if position_embeddings is not None:
            q_t = q.transpose(1, 2)
            if k is not None:
                k_t = k.transpose(1, 2)
                q_t, k_t = apply_rotary_pos_emb(q_t, k_t, cos, sin)
                k = k_t
            else:
                # Apply rope to q only — pass q as both args, discard the second.
                q_t, _ = apply_rotary_pos_emb(q_t, q_t, cos, sin)
            q = q_t
        else:
            q = q.transpose(1, 2)
            if k is not None:
                k = k.transpose(1, 2)
        if v is not None:
            v = v.transpose(1, 2)
        return q, k, v, cos, sin

    def forward_base(
        self,
        normed_base: torch.Tensor,
        *,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Base-stream half of the dual-stream attention.

        Owns the *only* K/V computation and the *only* cache update in the layer,
        and returns the post-cache ``(k, v)`` so :meth:`forward_adapt` can attend
        against exactly the same shared tensors. Every call goes through
        ``base_layer`` (``_base_only``), so this is pure and bit-identical to the
        unadapted base model.

        Returns ``(o_base, k_shaped, v_shaped)``.
        """
        q_base = _base_only(self.q_proj, normed_base)
        k = _base_only(self.k_proj, normed_base)
        v = _base_only(self.v_proj, normed_base)

        q_base, k_shaped, v_shaped, cos, sin = self._shape_qkv(
            q_base, k, v, position_embeddings,
        )

        if use_cache and past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_shaped, v_shaped = past_key_values.update(
                k_shaped, v_shaped, self.layer_idx, cache_kwargs,
            )

        attn_base = self._run_attention(q_base, k_shaped, v_shaped, attention_mask)
        return _base_only(self.o_proj, attn_base), k_shaped, v_shaped

    def forward_adapt(
        self,
        normed_adapt: torch.Tensor,
        k_shaped: torch.Tensor,
        v_shaped: Optional[torch.Tensor],
        *,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Adapter-stream half — Q only, against the base stream's shared K/V.

        ``k_shaped`` / ``v_shaped`` come from :meth:`forward_base` (post-cache).
        The adapter skips the K/V projections entirely and touches no cache, so
        calling this after the base half is what makes the shared-K/V topology
        work in either execution order.
        """
        q_adapt = self.q_proj(normed_adapt)
        # Q-only RoPE for the adapter stream (same positions as base).
        q_adapt, _, _, _, _ = self._shape_qkv(q_adapt, None, None, position_embeddings)
        attn_adapt = self._run_attention(q_adapt, k_shaped, v_shaped, attention_mask)
        return self.o_proj(attn_adapt)

    def forward(
        self,
        normed_base: torch.Tensor,
        normed_adapt: torch.Tensor,
        *,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Cache]]:
        """Shared-base-K/V dual-stream attention — both halves, base first.

        K/V are computed once from ``normed_base`` via ``base_layer`` (no delta).
        Q_base likewise via ``base_layer``; Q_adapt via the wrapped q_proj so the
        LoRA delta fires. Both Qs attend against the shared K/V; a single KV cache
        (``past_key_values``) is used. The base-stream calls are pure (no adapter
        state mutation), so this is checkpoint-safe and serves both training and
        inference.

        Returns ``(o_base, o_adapt, present_kv)``.
        """
        o_base, k_shaped, v_shaped = self.forward_base(
            normed_base,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        o_adapt = self.forward_adapt(
            normed_adapt,
            k_shaped,
            v_shaped,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        return o_base, o_adapt, past_key_values if use_cache else None

    def forward_bare(
        self,
        normed: torch.Tensor,
        *,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        """Single-stream attention for a bare (no-adapter) model.

        One plain Q/K/V/O pass. On a bare model the projections are ordinary
        ``nn.Linear`` (or LoRA-disabled), so this is exactly the unadapted base
        model's attention. Returns ``(o, present_kv)``.
        """
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        q, k_shaped, v_shaped, cos, sin = self._shape_qkv(q, k, v, position_embeddings)
        if use_cache and past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_shaped, v_shaped = past_key_values.update(
                k_shaped, v_shaped, self.layer_idx, cache_kwargs,
            )
        attn = self._run_attention(q, k_shaped, v_shaped, attention_mask)
        o = self.o_proj(attn)
        return o, past_key_values if use_cache else None
