# SPDX-License-Identifier: Apache-2.0
"""Shadow-residual attention for Granite Switch (HF backend).

Maintains two parallel hidden state streams (base and adapter).  The
projections (``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``) are
:class:`_StreamGatedLinear` instances; PEFT replaces each one the user
lists in ``target_modules`` with a :class:`ShadowResidualLora` wrapper.
The wrapper checks a per-call contextvar (``current_stream()``):

- ``"base"`` → wrapper returns ``base_layer(x)`` only (delta gated off).
- ``"adapter"`` (default) → wrapper applies the LoRA delta normally
  (with ALORA gating if configured).

Three forward modes (selected at runtime):

Disjoint K/V (default — ``config.shared_base_kv=False``)
--------------------------------------------------------
In dual-stream mode every wrapped projection is called twice — once
under ``stream_context("base")`` (LoRA delta gated off — base stream
is bit-identical to the unadapted base model) and once under
``stream_context("adapter")`` (delta fires). This applies to **all**
projections including ``k_proj`` and ``v_proj``: each stream computes
its own K and V from its own normed input and writes into its own KV
cache slot. There is no shared K/V tensor and no shared cache; the
``past_key_values_base`` cache holds ``W_kv · normed_base`` (frozen,
adapter-free), and ``past_key_values`` holds whatever the adapter
computes from ``normed_adapt`` — including any K/V LoRA delta the user
configured. The two attention calls each read their own K/V.

Shared base K/V (``config.shared_base_kv=True``)
------------------------------------------------
K and V are computed exactly once, from ``normed_base``, under
``stream_context("base")``. The adapter stream skips the K/V
projections entirely — it only computes Q (with optional Q LoRA on
``normed_adapt``). Both Q_base and Q_adapt then attend against the
same shared K/V; only one KV cache is used (the adapter cache;
``past_key_values_base`` is unused). Per-attention-head this is
``softmax((x_adapt · W_Q)(x_base · W_K)ᵀ / √d)(x_base · W_V)`` for the
adapter stream. K/V LoRA is structurally meaningless in this mode and
must not be configured (no place for a K/V adapter delta to land).
The base stream's Q, K, V, O, MLP all remain bit-identical to the
unadapted base model; the adapter stream attends to the same K/V the
frozen base would have written.

Single-stream early-exit (adapter-only)
---------------------------------------
When the model-level gate determines no :class:`CrossStream` site has
been wrapped (i.e. ``"cross_stream"`` is not in ``target_modules``), the
SR forward runs the **adapter path only** — base compute is skipped
entirely.  This is the "LoRA on the SR architecture without
cross-stream" path: equivalent to plain LoRA on Granite Switch. Only
one cache is used in this mode. The ``shared_base_kv`` flag is
irrelevant here — there is no second stream to share with.
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

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig

from ._stream_context import stream_context
from ._stream_gated_linear import _StreamGatedLinear


class ShadowResidualAttention(nn.Module):
    """Attention with shadow residual streams (base + adapter).

    Two attention calls (one per stream's Q against the shared K/V),
    with stream-gated wrappers around Q/O so the base contribution is
    identical to the unadapted base model.
    """

    def __init__(self, config: GraniteSwitchConfig, layer_idx: int):
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

        # Shared-base-K/V mode: K/V is computed once from normed_base and
        # both streams' Qs attend against the same K/V (single cache).
        # See module docstring for full description.
        self.shared_base_kv = bool(getattr(config, "shared_base_kv", False))

        # Optional QK-norm (Qwen3) — gated by config.qk_norm.
        self.qk_norm = getattr(config, "qk_norm", False)
        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size

        # Stream-gated linears — PEFT will dispatch these to
        # ShadowResidualLora when the user lists their attribute names
        # in target_modules.
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

    def forward(
        self,
        normed_base: torch.Tensor,
        normed_adapt: Optional[torch.Tensor],
        *,
        cross_stream_active: bool,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        past_key_values_base: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """Forward with shadow-residual streams.

        Args:
            normed_base: pre-normed base stream ``[B, S, H]``.
            normed_adapt: pre-normed adapter stream ``[B, S, H]``, or
                ``None`` in adapter-only early-exit mode (then
                ``cross_stream_active`` must be ``False``).
            cross_stream_active: when ``False`` only the adapter stream
                is computed (single Q/K/V/O call inside an ``"adapter"``
                context); the returned ``o_base`` is ``None``.
                When ``True`` both streams are computed with disjoint
                K/V projections and disjoint caches.
            past_key_values: KV cache for the **adapter** stream (and
                the only cache used in early-exit mode).
            past_key_values_base: KV cache for the **base** stream
                (dual-stream only). Holds frozen, adapter-free K/V.

        Returns:
            ``(o_base, o_adapt, present_kv)``. ``o_base`` is ``None`` in
            the adapter-only early-exit path. ``present_kv`` is the
            adapter cache (the base cache is updated in-place).
        """
        if not cross_stream_active:
            # ---- Adapter-only early-exit ----
            # Single Q/K/V/O pass under "adapter" context. K/V LoRA, if
            # configured, fires here. There is no base stream to protect.
            with stream_context("adapter"):
                q_adapt = self.q_proj(normed_base)
                k = self.k_proj(normed_base)
                v = self.v_proj(normed_base)
                q_adapt, k_shaped, v_shaped, cos, sin = self._shape_qkv(
                    q_adapt, k, v, position_embeddings,
                )
                if use_cache and past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    k_shaped, v_shaped = past_key_values.update(
                        k_shaped, v_shaped, self.layer_idx, cache_kwargs,
                    )
                attn_adapt = self._run_attention(q_adapt, k_shaped, v_shaped, attention_mask)
                o_adapt = self.o_proj(attn_adapt)
            return None, o_adapt, past_key_values if use_cache else None

        if self.shared_base_kv:
            # ---- Shared base K/V forward ----
            # K and V are computed once from normed_base under "base"
            # context (LoRA on K/V is forbidden in this mode — see
            # config_helpers). Q_base from normed_base ("base" context),
            # Q_adapt from normed_adapt ("adapter" context, Q LoRA delta
            # fires). Both Qs attend against the same K/V; only the
            # adapter cache (past_key_values) is used — past_key_values_base
            # is intentionally ignored.
            with stream_context("base"):
                q_base = self.q_proj(normed_base)
                k = self.k_proj(normed_base)
                v = self.v_proj(normed_base)
            with stream_context("adapter"):
                q_adapt = self.q_proj(normed_adapt)

            q_base, k_shaped, v_shaped, cos, sin = self._shape_qkv(
                q_base, k, v, position_embeddings,
            )
            # Q-only RoPE for the adapter stream (same positions as base).
            q_adapt, _, _, _, _ = self._shape_qkv(
                q_adapt, None, None, position_embeddings,
            )

            if use_cache and past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k_shaped, v_shaped = past_key_values.update(
                    k_shaped, v_shaped, self.layer_idx, cache_kwargs,
                )

            attn_base = self._run_attention(q_base, k_shaped, v_shaped, attention_mask)
            attn_adapt = self._run_attention(q_adapt, k_shaped, v_shaped, attention_mask)

            with stream_context("base"):
                o_base = self.o_proj(attn_base)
            with stream_context("adapter"):
                o_adapt = self.o_proj(attn_adapt)

            return o_base, o_adapt, past_key_values if use_cache else None

        # ---- Dual-stream forward (disjoint K/V) ----
        # Each stream computes its own Q/K/V from its own normed input.
        # Base call: LoRA delta gated off (frozen-base invariant — base
        # K/V is bit-identical to the unadapted base model's K/V on
        # normed_base). Adapter call: delta fires on normed_adapt.
        # Two parallel KV caches: past_key_values (adapter) holds the
        # adapter K/V; past_key_values_base holds the frozen base K/V.
        cache_kwargs = None
        with stream_context("base"):
            q_base = self.q_proj(normed_base)
            k_base = self.k_proj(normed_base)
            v_base = self.v_proj(normed_base)
            q_base, k_base_shaped, v_base_shaped, cos, sin = self._shape_qkv(
                q_base, k_base, v_base, position_embeddings,
            )
            if use_cache and past_key_values_base is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k_base_shaped, v_base_shaped = past_key_values_base.update(
                    k_base_shaped, v_base_shaped, self.layer_idx, cache_kwargs,
                )

        with stream_context("adapter"):
            q_adapt = self.q_proj(normed_adapt)
            k_adapt = self.k_proj(normed_adapt)
            v_adapt = self.v_proj(normed_adapt)
            q_adapt, k_adapt_shaped, v_adapt_shaped, _, _ = self._shape_qkv(
                q_adapt, k_adapt, v_adapt, position_embeddings,
            )
            if use_cache and past_key_values is not None:
                if cache_kwargs is None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k_adapt_shaped, v_adapt_shaped = past_key_values.update(
                    k_adapt_shaped, v_adapt_shaped, self.layer_idx, cache_kwargs,
                )

        # Two attention calls — each Q against its own K/V.
        attn_base = self._run_attention(q_base, k_base_shaped, v_base_shaped, attention_mask)
        attn_adapt = self._run_attention(q_adapt, k_adapt_shaped, v_adapt_shaped, attention_mask)

        # O projections — base under "base" context (frozen-base
        # invariant), adapter under "adapter" context.
        with stream_context("base"):
            o_base = self.o_proj(attn_base)
        with stream_context("adapter"):
            o_adapt = self.o_proj(attn_adapt)

        return o_base, o_adapt, past_key_values if use_cache else None
