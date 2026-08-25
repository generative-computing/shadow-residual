# SPDX-License-Identifier: Apache-2.0
"""Fused→unfused weight transfer for shadow-residual.

The upstream :class:`GraniteMoeHybridForCausalLM` keeps attention projections
fused as ``self_attn.qkv_proj`` and the MLP gate/up fused as
``shared_mlp.input_linear``. The shadow-residual decoder uses unfused
``q_proj`` / ``k_proj`` / ``v_proj`` and ``gate_proj`` / ``up_proj`` so
that PEFT can install LoRA on each one independently.

This module slices fused-source tensors into the unfused-destination
tensors. Everything else transfers by name.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _qkv_slice_sizes(config) -> tuple[int, int, int]:
    head_dim = getattr(
        config,
        "projection_head_dim",
        config.hidden_size // config.num_attention_heads,
    )
    q_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim
    return q_size, kv_size, kv_size


def _unbind_param(model, full_name: str) -> None:
    """Detach a parameter or buffer from its parent module so refcount→0.

    Walks ``full_name`` (e.g. ``model.layers.0.self_attn.qkv_proj.weight``)
    down to the leaf module, then sets the leaf attribute to ``None``.
    Tolerant of names that no longer exist (already unbound).
    """
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return
        parent = getattr(parent, p)
    leaf = parts[-1]
    if not hasattr(parent, leaf):
        return
    # nn.Module.__setattr__ rejects setting a registered Parameter to a non-
    # Parameter; use _parameters / _buffers dict assignment to bypass.
    if hasattr(parent, "_parameters") and leaf in parent._parameters:
        parent._parameters[leaf] = None
    elif hasattr(parent, "_buffers") and leaf in parent._buffers:
        parent._buffers[leaf] = None
    else:
        try:
            setattr(parent, leaf, None)
        except (TypeError, AttributeError):
            pass


def transfer_base_weights(src_model, dst_model, *, drain_src: bool = False) -> None:
    """Copy base parameters from ``src_model`` into ``dst_model``.

    Slices fused source tensors (``qkv_proj``, ``shared_mlp.input_linear``)
    into the per-projection unfused destination tensors. Strips any
    ``base_layer`` indirection that PEFT or SwitchedLoRA wrappers leave on
    source state-dict keys. LoRA tensors on the source side are skipped —
    SR carries no built-in LoRA.

    The destination is mutated in place.

    Args:
        drain_src: when True, each source tensor is unbound from
            ``src_model`` immediately after it has been copied (or
            sliced and copied) into the destination. This drops peak
            CPU RAM during construction from ``|src| + |dst|`` to
            roughly ``|dst| + one_tensor``, which is the difference
            between OOM and not for 30B-scale bases at 4 ranks × 400Gi.
            After this call ``src_model`` is unusable — its parameters
            are gone — so the caller must ``del`` it. Defaults to
            False to preserve the historical "src is read-only"
            contract that tests rely on.
    """
    config = dst_model.config
    src_state = src_model.state_dict()
    # The destination may itself be peft-wrapped — in that case its state_dict
    # carries `.base_layer.weight` keys for every wrapped projection. The
    # source-side key paths we compute below (`...q_proj.weight`,
    # `...gate_proj.weight`) are the unwrapped names, so we additionally
    # register a stripped alias for each peft-wrapped destination key. The
    # underlying tensors are shared (state_dict returns the live Parameters),
    # so an in-place ``copy_`` through either alias mutates the same storage.
    dst_state = dict(dst_model.state_dict())
    for dst_key in list(dst_state.keys()):
        if ".base_layer.weight" in dst_key:
            stripped = dst_key.replace(".base_layer.weight", ".weight")
            dst_state.setdefault(stripped, dst_state[dst_key])
        elif ".base_layer.bias" in dst_key:
            stripped = dst_key.replace(".base_layer.bias", ".bias")
            dst_state.setdefault(stripped, dst_state[dst_key])
    dst_keys = set(dst_state.keys())

    q_size, k_size, v_size = _qkv_slice_sizes(config)

    n_copied = 0
    n_split = 0
    n_skipped = 0

    # Snapshot keys so we can pop entries out of src_state as we go (drains
    # the dict's strong refs, complementing the per-tensor _unbind_param
    # under drain_src=True).
    src_items = list(src_state.items())
    src_state.clear()

    def _handle_one(src_key, src_val):
        nonlocal n_copied, n_split, n_skipped

        if "lora_A" in src_key or "lora_B" in src_key:
            n_skipped += 1
            return

        # Fused QKV → q_proj / k_proj / v_proj
        if src_key.endswith("self_attn.qkv_proj.weight") or src_key.endswith(
            "self_attn.qkv_proj.base_layer.weight"
        ):
            prefix = src_key[: src_key.index("self_attn.qkv_proj")] + "self_attn."
            q_w = src_val[:q_size, :]
            k_w = src_val[q_size : q_size + k_size, :]
            v_w = src_val[q_size + k_size : q_size + k_size + v_size, :]
            for name, tensor in [
                ("q_proj.weight", q_w),
                ("k_proj.weight", k_w),
                ("v_proj.weight", v_w),
            ]:
                dst_key = prefix + name
                if dst_key in dst_keys and dst_state[dst_key].shape == tensor.shape:
                    with torch.no_grad():
                        dst_state[dst_key].copy_(tensor)
                    n_split += 1
                else:
                    n_skipped += 1
            return

        if src_key.endswith("self_attn.qkv_proj.bias") or src_key.endswith(
            "self_attn.qkv_proj.base_layer.bias"
        ):
            prefix = src_key[: src_key.index("self_attn.qkv_proj")] + "self_attn."
            q_b = src_val[:q_size]
            k_b = src_val[q_size : q_size + k_size]
            v_b = src_val[q_size + k_size : q_size + k_size + v_size]
            for name, tensor in [
                ("q_proj.bias", q_b),
                ("k_proj.bias", k_b),
                ("v_proj.bias", v_b),
            ]:
                dst_key = prefix + name
                if dst_key in dst_keys and dst_state[dst_key].shape == tensor.shape:
                    with torch.no_grad():
                        dst_state[dst_key].copy_(tensor)
                    n_split += 1
                else:
                    n_skipped += 1
            return

        # shared_mlp.input_linear (gate ⨁ up) → mlp.gate_proj / mlp.up_proj
        if src_key.endswith("shared_mlp.input_linear.weight") or src_key.endswith(
            "shared_mlp.input_linear.base_layer.weight"
        ):
            layer_prefix = src_key[: src_key.index("shared_mlp.")]
            inter = src_val.shape[0] // 2
            gate_w = src_val[:inter, :]
            up_w = src_val[inter:, :]
            for name, tensor in [("mlp.gate_proj.weight", gate_w), ("mlp.up_proj.weight", up_w)]:
                dst_key = layer_prefix + name
                if dst_key in dst_keys and dst_state[dst_key].shape == tensor.shape:
                    with torch.no_grad():
                        dst_state[dst_key].copy_(tensor)
                    n_split += 1
                else:
                    n_skipped += 1
            return

        # shared_mlp.output_linear → mlp.down_proj
        if src_key.endswith("shared_mlp.output_linear.weight") or src_key.endswith(
            "shared_mlp.output_linear.base_layer.weight"
        ):
            layer_prefix = src_key[: src_key.index("shared_mlp.")]
            dst_key = layer_prefix + "mlp.down_proj.weight"
            if dst_key in dst_keys and dst_state[dst_key].shape == src_val.shape:
                with torch.no_grad():
                    dst_state[dst_key].copy_(src_val)
                n_copied += 1
            else:
                n_skipped += 1
            return

        # Sparse-MoE expert bank (Granite 5.0 / granitemoe). Upstream stores the
        # expert weights under `block_sparse_moe.*`; the SR MLP nests the mirrored
        # frozen bank under `.mlp.` (router + input_linear + output_linear, see
        # decoder_hf.ShadowResidualMLP MoE mode). Straight copy_ — no reslice:
        #   * input_linear.weight  : 3-D [n_experts, 2*inter, H] (gate⊕up fused;
        #                            SR's SwiGLU chunk consumes it identically)
        #   * output_linear.weight : 3-D [n_experts, H, inter]
        #   * router.layer.weight  : 2-D [n_experts, H]
        # (The experts are NOT LoRA targets in the attention-only iteration, so no
        # .base_layer indirection is expected on the destination — but we accept it
        # for symmetry with the other branches in case that changes.)
        for moe_tail, sr_name in (
            ("block_sparse_moe.input_linear.weight", "mlp.input_linear.weight"),
            ("block_sparse_moe.output_linear.weight", "mlp.output_linear.weight"),
            ("block_sparse_moe.router.layer.weight", "mlp.router.layer.weight"),
        ):
            if src_key.endswith(moe_tail) or src_key.endswith(
                moe_tail.replace(".weight", ".base_layer.weight")
            ):
                marker = "block_sparse_moe."
                layer_prefix = src_key[: src_key.index(marker)]
                dst_key = layer_prefix + sr_name
                if dst_key in dst_keys and dst_state[dst_key].shape == src_val.shape:
                    with torch.no_grad():
                        dst_state[dst_key].copy_(src_val)
                    n_copied += 1
                else:
                    n_skipped += 1
                return

        # Already-unfused MLP on the upstream — keys live nested under .mlp.
        # (granite, llama, qwen, … all do this). The SR decoder layer also
        # nests gate_proj / up_proj / down_proj under self.mlp, so the key
        # structure matches directly (layers.{i}.mlp.gate_proj.weight on both
        # sides). This branch handles the base_layer indirection that PEFT
        # adds when the destination is already PEFT-wrapped.
        for mlp_proj in ("gate_proj", "up_proj", "down_proj"):
            for tail in (
                f"mlp.{mlp_proj}.weight",
                f"mlp.{mlp_proj}.base_layer.weight",
                f"mlp.{mlp_proj}.bias",
                f"mlp.{mlp_proj}.base_layer.bias",
            ):
                if not src_key.endswith(tail):
                    continue
                layer_prefix = src_key[: src_key.index("mlp.")]
                dst_suffix = "weight" if tail.endswith("weight") else "bias"
                dst_key = f"{layer_prefix}mlp.{mlp_proj}.{dst_suffix}"
                if dst_key in dst_keys and dst_state[dst_key].shape == src_val.shape:
                    with torch.no_grad():
                        dst_state[dst_key].copy_(src_val)
                    n_copied += 1
                else:
                    n_skipped += 1
                return

        # Strip ``base_layer`` indirection for everything else.
        candidate = src_key.replace(".base_layer.weight", ".weight").replace(
            ".base_layer.bias", ".bias"
        )
        if candidate not in dst_keys:
            n_skipped += 1
            return
        dst_val = dst_state[candidate]
        if dst_val.shape != src_val.shape:
            n_skipped += 1
            return
        with torch.no_grad():
            dst_val.copy_(src_val)
        n_copied += 1

    for src_key, src_val in src_items:
        _handle_one(src_key, src_val)
        if drain_src:
            _unbind_param(src_model, src_key)
        del src_val

    logger.info(
        "Shadow-residual weight transfer: copied %d, split %d, skipped %d.",
        n_copied, n_split, n_skipped,
    )


__all__ = ["transfer_base_weights"]
