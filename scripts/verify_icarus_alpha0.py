# SPDX-License-Identifier: Apache-2.0
"""CPU verification of the SR-vs-Icarus mechanism before burning GPU hours.

Icarus == dual-stream SR with the cross_stream LoRA injection pinned OFF via a
per-module alpha of 0. This script builds a tiny SR+PEFT model two ways and
asserts the exact numerics the comparison relies on:

  SR     (cross_stream alpha > 0): injection fires, cross_stream LoRA trains.
  Icarus (cross_stream alpha = 0): injection is identically 0 AND cross_stream
                                   LoRA receives no gradient (pinned at init).

If any assertion here fails, the 12-cell comparison would be measuring the wrong
thing — so this is the go/no-go gate for launching.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.shadow_residual.config_helpers import set_shadow_residual
from shadow_residual.shadow_residual.cross_stream import CrossStream


def tiny_sr_config() -> ShadowResidualConfig:
    cfg = ShadowResidualConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_adapters=0,
        max_lora_rank=16,
        switch_head_dim=16,
    )
    set_shadow_residual(cfg, enabled=True)
    return cfg


def build(alpha_cross: int, r: int = 16):
    """Build tiny SR+PEFT with q/o alpha=64 and cross_stream alpha=alpha_cross."""
    torch.manual_seed(0)
    model = ShadowResidualForCausalLM(tiny_sr_config())
    lc = LoraConfig(
        r=r,
        lora_alpha=min(64, alpha_cross) if alpha_cross == 0 else 64,
        target_modules=["q_proj", "o_proj", "cross_stream"],
        rank_pattern={"q_proj": r, "o_proj": r, "cross_stream": r},
        alpha_pattern={"q_proj": 64, "o_proj": 64, "cross_stream": alpha_cross},
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lc)
    # Perturb cross_stream lora_B off its zero init so a live injection is
    # actually nonzero (otherwise SR and Icarus would both read 0 at init and
    # the forward check couldn't tell them apart).
    with torch.no_grad():
        for m in peft_model.modules():
            if isinstance(m, CrossStream):
                # m is the base_layer; its parent LoRA wrapper holds lora_B.
                pass
        for name, p in peft_model.named_parameters():
            if "cross_stream" in name and "lora_B" in name:
                p.normal_(std=0.1)
    return peft_model


def cross_stream_scalings(peft_model):
    """Return the resolved LoRA scaling (alpha/r) for each wrapped cross_stream."""
    out = []
    for name, m in peft_model.named_modules():
        if name.endswith("cross_stream") and hasattr(m, "scaling"):
            out.append((name, dict(m.scaling)))
    return out


def main() -> int:
    print("=== Building Icarus (cross_stream alpha=0) ===")
    icarus = build(alpha_cross=0)
    sc = cross_stream_scalings(icarus)
    assert sc, "no wrapped cross_stream found — target_modules wiring broke"
    for name, scaling in sc:
        val = scaling["default"]
        print(f"  {name}: scaling={val}")
        assert val == 0.0, f"Icarus cross_stream scaling must be 0, got {val}"

    print("=== Building SR (cross_stream alpha=64) ===")
    sr = build(alpha_cross=64)
    for name, scaling in cross_stream_scalings(sr):
        val = scaling["default"]
        print(f"  {name}: scaling={val}")
        assert val > 0.0, f"SR cross_stream scaling must be >0, got {val}"

    # Forward: same input, same seed → the ONLY numerical difference between the
    # two models is the cross_stream injection. Logits must differ for SR but the
    # cross_stream contribution must be exactly 0 for Icarus.
    ids = torch.randint(0, 300, (2, 8))
    with torch.no_grad():
        lo_icarus = icarus(input_ids=ids).logits
        lo_sr = sr(input_ids=ids).logits
    max_abs_diff = (lo_sr - lo_icarus).abs().max().item()
    print(f"=== Forward: max|logit_SR - logit_Icarus| = {max_abs_diff:.6g} ===")
    assert max_abs_diff > 1e-4, (
        "SR and Icarus produced identical logits — the live cross_stream "
        "injection had no effect. Either alpha didn't resolve or the forward "
        "doesn't use cross_stream."
    )

    # Gradient check: cross_stream lora_A/lora_B must get NO gradient in Icarus
    # (pinned at init) and a nonzero gradient in SR.
    print("=== Gradient check ===")
    for tag, model in (("Icarus", build(alpha_cross=0)), ("SR", build(alpha_cross=64))):
        model.zero_grad()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        grads = {}
        for name, p in model.named_parameters():
            if "cross_stream" in name and ("lora_A" in name or "lora_B" in name):
                g = None if p.grad is None else p.grad.abs().sum().item()
                grads[name.split(".lora")[-1]] = g
        gsum = sum(v for v in grads.values() if v)
        print(f"  {tag}: cross_stream lora grad abs-sum = {gsum}")
        if tag == "Icarus":
            assert gsum == 0.0, f"Icarus cross_stream received gradient ({gsum}) — not pinned!"
        else:
            assert gsum > 0.0, f"SR cross_stream received NO gradient ({gsum}) — not training!"

    print("\nALL CHECKS PASSED — SR vs Icarus mechanism verified on CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
