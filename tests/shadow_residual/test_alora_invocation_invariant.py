# SPDX-License-Identifier: Apache-2.0
"""Always-active adapter invariant for the SR path (HF backend).

aLoRA gating was removed: the adapter is plain LoRA and active on EVERY
position. This test pins that behavior — with a nonzero LoRA delta the model's
logits must differ from the base-equivalent (``lora_B = 0``) control on *all*
positions, including the earliest ones. (Under the old aLoRA gating, early
positions matched the base; that is no longer the case, and this test guards
against a gating regression sneaking back in.)

CPU-only, tiny config, no HF download.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.shadow_residual import (
    ShadowResidualForCausalLM,
)
from shadow_residual.shadow_residual.config_helpers import (
    set_shadow_residual,
)


INPUT_IDS = torch.tensor([[1, 2, 3, 7, 8, 9]], dtype=torch.long)


def _build_tiny_sr_config() -> ShadowResidualConfig:
    cfg = ShadowResidualConfig(
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


def _build_sr_peft(*, cfg: ShadowResidualConfig, seed: int, perturb_lora_b: bool):
    """Build an SR + PEFT model with deterministic init.

    ``perturb_lora_b``: when False, ``lora_B`` stays at peft's zero init → the
    adapter is a no-op everywhere → output equals the base model (the reference).
    """
    torch.manual_seed(seed)
    sr = ShadowResidualForCausalLM(cfg)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        # Shared-KV only: no k_proj/v_proj (K/V LoRA forbidden). Q/O/cross_stream.
        target_modules=["q_proj", "o_proj", "cross_stream"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(sr, lora_cfg)

    if perturb_lora_b:
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_B" in name:
                    p.data.normal_(mean=0.0, std=0.05)

    peft_model.eval()
    return peft_model


def test_adapter_active_on_all_positions():
    """With a nonzero LoRA delta, logits differ from the base on every position.

    No gating → the adapter fires from position 0. Compare an SR+LoRA model with
    perturbed ``lora_B`` against the same model with ``lora_B = 0`` (base
    reference). They must differ on *every* position (in particular position 0,
    which the old aLoRA gate would have left base-identical).
    """
    cfg = _build_tiny_sr_config()

    model_adapt = _build_sr_peft(cfg=cfg, seed=0, perturb_lora_b=True)
    model_base = _build_sr_peft(cfg=cfg, seed=0, perturb_lora_b=False)

    with torch.no_grad():
        out_adapt = model_adapt(input_ids=INPUT_IDS)
        out_base = model_base(input_ids=INPUT_IDS)

    seq_len = INPUT_IDS.shape[1]
    for pos in range(seq_len):
        assert not torch.allclose(
            out_adapt.logits[:, pos, :], out_base.logits[:, pos, :],
            atol=1e-5, rtol=1e-4,
        ), (
            f"position {pos}: adapter logits match the no-LoRA baseline — the "
            f"adapter is not active here, i.e. gating regressed back in"
        )
