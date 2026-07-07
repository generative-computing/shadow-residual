# SPDX-License-Identifier: Apache-2.0
"""Pre-invocation invariant test for the SR + aLoRA path (HF backend).

When ``LoraConfig.alora_invocation_tokens`` is set, the adapter must be
inactive on positions strictly before the start of the *last* invocation
match in ``input_ids``: every wrapped projection collapses to ``Wx``
(LoRA delta gated off via ``ALoraLinearVariant``), and the cross-stream
contribution to ``h_adapt`` is zero (``CrossStreamLora`` zero-masks the
delta). Therefore on those pre-invocation positions:

1. The model's logits must equal the logits of the *same model* with
   ``lora_B = 0`` (a base-equivalent control, since with B=0 every LoRA
   delta is identically zero on every position).
2. The adapter-side K/V cache (``past_key_values``) entries must equal
   the base-side K/V cache (``past_key_values_base``) entries, because
   adapter K/V are computed as ``Wx`` from ``h_adapt`` which itself
   equals ``h_base`` pre-invocation.

Both invariants are by-construction consequences of the `stream_context`
+ `ALoraLinearVariant` + `CrossStreamLora` masking, but the failure
modes (someone moves a `stream_context`, peft changes the variant
dispatch, K/V LoRA gets retargeted) are silent — this test pins them.

CPU-only, tiny config, no HF download.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig as GraniteSwitchConfig
from shadow_residual.shadow_residual import (
    ShadowResidualForCausalLM,
)
from shadow_residual.shadow_residual._stream_gated_linear import (
    _StreamGatedLinear,
)
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
)
from shadow_residual.shadow_residual.cross_stream import CrossStream
from shadow_residual.peft_shadow_residual import CrossStreamLora
from shadow_residual.peft_shadow_residual.stream_gated_lora import (
    ShadowResidualLora,
)


# Position of the (single-token) invocation in the prompt below.
# input_ids = [1, 2, 3, T_INV, 7, 8] → invocation starts at position 3.
INVOCATION_TOKEN_ID = 250
INPUT_IDS = torch.tensor([[1, 2, 3, INVOCATION_TOKEN_ID, 7, 8]], dtype=torch.long)
P_INVOCATION = 3  # first post-invocation position


def _build_tiny_sr_config() -> GraniteSwitchConfig:
    cfg = GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        num_adapters=0,
        max_lora_rank=4,
        switch_head_dim=16,
    )
    set_shadow_residual(cfg, enabled=True)
    return cfg


def _build_sr_peft(
    *,
    cfg: GraniteSwitchConfig,
    seed: int,
    alora_invocation_tokens: list[int] | None,
    perturb_lora_b: bool,
) -> torch.nn.Module:
    """Build an SR + PEFT model with deterministic init.

    Two knobs:
    - ``alora_invocation_tokens``: ``None`` → unconditional LoRA; list →
      gated aLoRA (peft attaches ``ALoraLinearVariant`` automatically).
    - ``perturb_lora_b``: when False, ``lora_B`` stays at peft's zero
      init → adapter is a no-op everywhere → output equals base model.
      That's the "base reference" we compare against.
    """
    torch.manual_seed(seed)
    sr = ShadowResidualForCausalLM(cfg)

    lora_kwargs = dict(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "cross_stream"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",  # required for ALoraLinearVariant dispatch
    )
    if alora_invocation_tokens is not None:
        lora_kwargs["alora_invocation_tokens"] = alora_invocation_tokens
    lora_cfg = LoraConfig(**lora_kwargs)
    lora_cfg._register_custom_module(
        {CrossStream: CrossStreamLora, _StreamGatedLinear: ShadowResidualLora}
    )
    peft_model = get_peft_model(sr, lora_cfg)

    if perturb_lora_b:
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_B" in name:
                    p.data.normal_(mean=0.0, std=0.05)

    peft_model.eval()
    return peft_model


def _get_inner_sr_model(peft_model) -> torch.nn.Module:
    """Walk down to the ``ShadowResidualModel`` that owns ``_pkv_base``."""
    # peft: PeftModelForCausalLM.base_model = LoraModel
    #         LoraModel.model = ShadowResidualForCausalLM
    #           .model = ShadowResidualModel  ← we want this
    return peft_model.base_model.model.model


def test_pre_invocation_logits_match_base_dual_stream():
    """Pre-invocation logits with aLoRA gating == base-equivalent logits.

    Two models built from the same seed and config:
    - ``model_alora``: aLoRA gating on, ``lora_B`` perturbed → adapter
      fires only post-invocation.
    - ``model_base``: no gating, ``lora_B = 0`` → adapter is a no-op
      everywhere → output is the base-equivalent reference.

    For positions ``[0, P_INVOCATION)`` the two must agree to fp
    tolerance. For positions ``[P_INVOCATION:]`` they must differ on at
    least one element — otherwise the gating is vacuous (e.g. the LoRA
    delta isn't reaching the model at all) and the pre-invocation
    agreement is meaningless.
    """
    cfg = _build_tiny_sr_config()

    model_alora = _build_sr_peft(
        cfg=cfg, seed=0,
        alora_invocation_tokens=[INVOCATION_TOKEN_ID],
        perturb_lora_b=True,
    )
    model_base = _build_sr_peft(
        cfg=cfg, seed=0,
        alora_invocation_tokens=None,
        perturb_lora_b=False,  # lora_B stays zero → base-equivalent
    )

    with torch.no_grad():
        out_alora = model_alora(input_ids=INPUT_IDS)
        out_base = model_base(input_ids=INPUT_IDS)

    pre = slice(0, P_INVOCATION)
    post = slice(P_INVOCATION, None)

    # Pre-invocation: aLoRA must collapse to base-equivalent.
    torch.testing.assert_close(
        out_alora.logits[:, pre, :], out_base.logits[:, pre, :],
        atol=1e-5, rtol=1e-4,
        msg="pre-invocation logits diverge from base-equivalent — the aLoRA "
            "gate isn't fully zeroing the LoRA / cross-stream delta on "
            "positions < P_INVOCATION",
    )

    # Post-invocation: adapter must actually be doing something. If this
    # passes vacuously the pre-invocation check above is meaningless.
    assert not torch.allclose(
        out_alora.logits[:, post, :], out_base.logits[:, post, :],
        atol=1e-5, rtol=1e-4,
    ), (
        "post-invocation logits match the no-LoRA baseline — the LoRA "
        "delta isn't reaching the model, so the pre-invocation agreement "
        "is vacuous"
    )


def test_pre_invocation_kv_cache_matches_base_dual_stream():
    """Adapter K/V cache values pre-invocation == base K/V cache values.

    With aLoRA gating on, every K/V projection on the adapter side
    collapses to ``W_kv · normed_adapt`` for ``pos < P_INVOCATION``.
    Inductively (h_adapt[pos<p] == h_base[pos<p] across layers, see the
    decoder hf docstring), normed_adapt == normed_base on those
    positions, so adapter K/V == base K/V there.

    We compare ``past_key_values`` (adapter cache) to
    ``past_key_values_base`` (base cache) layer by layer. Sanity:
    they must DIFFER on post-invocation positions, otherwise we're
    comparing two empty caches or identical streams.
    """
    cfg = _build_tiny_sr_config()
    model_alora = _build_sr_peft(
        cfg=cfg, seed=0,
        alora_invocation_tokens=[INVOCATION_TOKEN_ID],
        perturb_lora_b=True,
    )

    with torch.no_grad():
        model_alora(input_ids=INPUT_IDS, use_cache=True)

    inner = _get_inner_sr_model(model_alora)
    cache_adapt = inner._pkv_base_owner  # adapter cache
    cache_base = inner._pkv_base  # base cache

    assert cache_adapt is not None and cache_base is not None, (
        "expected both adapter (`_pkv_base_owner`) and base (`_pkv_base`) "
        "caches to exist after a use_cache=True forward"
    )

    n_layers = cfg.num_hidden_layers
    pre = slice(0, P_INVOCATION)
    post = slice(P_INVOCATION, None)

    saw_post_difference = False
    for layer_idx in range(n_layers):
        k_adapt = cache_adapt.layers[layer_idx].keys
        v_adapt = cache_adapt.layers[layer_idx].values
        k_base = cache_base.layers[layer_idx].keys
        v_base = cache_base.layers[layer_idx].values

        # Pre-invocation: adapter K/V must equal base K/V exactly enough
        # that fp noise from the parallel compute path is the only
        # explanation for any drift.
        torch.testing.assert_close(
            k_adapt[:, :, pre, :], k_base[:, :, pre, :],
            atol=1e-5, rtol=1e-4,
            msg=f"layer {layer_idx}: adapter K diverges from base K on "
                "pre-invocation positions — adapter K/V cache is being "
                "polluted by the LoRA delta despite aLoRA gating",
        )
        torch.testing.assert_close(
            v_adapt[:, :, pre, :], v_base[:, :, pre, :],
            atol=1e-5, rtol=1e-4,
            msg=f"layer {layer_idx}: adapter V diverges from base V on "
                "pre-invocation positions",
        )

        # Sanity: post-invocation, K/V must diverge somewhere across
        # layers (the delta is firing, so K/V LoRA + cross-stream-driven
        # h_adapt drift should break the equality on at least one layer).
        if not torch.allclose(
            k_adapt[:, :, post, :], k_base[:, :, post, :],
            atol=1e-5, rtol=1e-4,
        ):
            saw_post_difference = True

    assert saw_post_difference, (
        "adapter K and base K agree on every layer at every "
        "post-invocation position — the adapter is a no-op, so the "
        "pre-invocation equality above is vacuous"
    )
