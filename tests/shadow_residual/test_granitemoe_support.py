# SPDX-License-Identifier: Apache-2.0
"""Granite 5.0 (``granitemoe`` sparse-MoE) support for shadow-residual.

Iteration 1 = **attention-only** adaptation: LoRA on Q/O + ``cross_stream``, the
sparse expert bank runs **frozen** so the base stream stays bit-identical to the
stock ``GraniteMoeForCausalLM``. These CPU tests cover:

- Frozen-base MoE parity: SR ``forward_bare`` and the dual-stream base stream
  reproduce stock ``GraniteMoeForCausalLM`` logits (weight transfer + the MoE MLP
  mirror are correct).
- Experts/router are frozen after a stock-PEFT attach; only attention LoRA +
  cross-stream are trainable.
- The single ``ShadowResidualMLP`` class builds the MoE bank on a granitemoe
  config and the dense SwiGLU on a dense config (per-layer mode, no new class).
- Gradient-checkpoint recompute is exact in MoE mode.
"""

import pytest
import torch

from peft import LoraConfig, get_peft_model

transformers = pytest.importorskip("transformers")
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM  # noqa: E402

from shadow_residual.shadow_residual.build import build_sr_base  # noqa: E402
from shadow_residual.shadow_residual.decoder_hf import ShadowResidualMLP  # noqa: E402
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────


def _tiny_moe_config():
    return GraniteMoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,   # per-expert FFN width
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )


@pytest.fixture
def tiny_moe_base(tmp_path):
    """A saved tiny stock ``GraniteMoeForCausalLM`` plus its logits on a fixed
    input, so tests can build an SR base from it and compare."""
    torch.manual_seed(0)
    cfg = _tiny_moe_config()
    stock = GraniteMoeForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        stock_logits = stock(input_ids=ids).logits
    path = tmp_path / "tiny_granitemoe"
    stock.save_pretrained(path)
    cfg.save_pretrained(path)
    return {"path": str(path), "ids": ids, "stock_logits": stock_logits}


def _attach_attention_lora(sr_model, *, perturb=True):
    """Attention-only + cross_stream LoRA (the iteration-1 target set)."""
    lora = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "o_proj", "cross_stream"],
        lora_dropout=0.0,
    )
    pm = get_peft_model(sr_model, lora)
    if perturb:
        with torch.no_grad():
            for name, p in pm.named_parameters():
                if "lora_B" in name:
                    p.data.normal_(mean=0.0, std=0.02)
    return pm


# ── Config / construction ─────────────────────────────────────────


class TestMoEConfigAndConstruction:
    def test_moe_config_carries_expert_fields(self):
        """A granitemoe-derived SR config keeps num_local_experts / top-k
        (they are no longer pinned to 0) and passes the hybrid parent's strict
        Mamba validator despite the granitemoe base carrying no mamba dims."""
        cfg = ShadowResidualConfig(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=48,
            num_local_experts=4, num_experts_per_tok=2, vocab_size=128,
        )
        assert cfg.num_local_experts == 4
        assert cfg.num_experts_per_tok == 2

    def test_mlp_is_moe_on_moe_config_dense_otherwise(self):
        """One ShadowResidualMLP class: MoE bank when num_local_experts>0,
        dense SwiGLU otherwise."""
        moe_cfg = ShadowResidualConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=48,
            num_local_experts=4, num_experts_per_tok=2, vocab_size=128,
        )
        dense_cfg = ShadowResidualConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=48,
            num_local_experts=0, vocab_size=128,
        )
        moe_mlp = ShadowResidualMLP(moe_cfg)
        dense_mlp = ShadowResidualMLP(dense_cfg)
        assert moe_mlp.is_moe is True
        assert hasattr(moe_mlp, "router") and hasattr(moe_mlp, "input_linear")
        assert not hasattr(moe_mlp, "gate_proj")
        assert dense_mlp.is_moe is False
        assert hasattr(dense_mlp, "gate_proj")
        assert not hasattr(dense_mlp, "router")


# ── Frozen-base parity ────────────────────────────────────────────


class TestFrozenBaseParity:
    def test_bare_forward_matches_stock(self, tiny_moe_base):
        """SR bare (no adapter) forward == stock GraniteMoeForCausalLM logits.
        Exercises the MoE weight-transfer branches + the MoE MLP mirror."""
        sr = build_sr_base(tiny_moe_base["path"]).eval()
        assert sr.model.layers[0].mlp.is_moe is True
        with torch.no_grad():
            sr_logits = sr(input_ids=tiny_moe_base["ids"]).logits
        torch.testing.assert_close(
            sr_logits, tiny_moe_base["stock_logits"], atol=1e-4, rtol=1e-4,
        )

    def test_base_stream_matches_stock_under_disabled_adapter(self, tiny_moe_base):
        """With a (perturbed) adapter attached but disabled, the SR base stream
        reproduces stock logits — the frozen-base invariant on a MoE base."""
        sr = build_sr_base(tiny_moe_base["path"])
        pm = _attach_attention_lora(sr, perturb=True).eval()
        with torch.no_grad():
            with pm.disable_adapter():
                disabled = pm(input_ids=tiny_moe_base["ids"]).logits
        torch.testing.assert_close(
            disabled, tiny_moe_base["stock_logits"], atol=1e-4, rtol=1e-4,
        )


# ── Freezing / trainability ───────────────────────────────────────


class TestExpertsFrozen:
    def test_experts_and_router_frozen_after_peft_attach(self, tiny_moe_base):
        sr = build_sr_base(tiny_moe_base["path"])
        pm = _attach_attention_lora(sr, perturb=False)
        trainable_experts = [
            n for n, p in pm.named_parameters()
            if p.requires_grad and (
                "input_linear" in n or "output_linear" in n or "router" in n
            )
        ]
        assert trainable_experts == [], trainable_experts

    def test_only_attention_and_cross_stream_trainable(self, tiny_moe_base):
        sr = build_sr_base(tiny_moe_base["path"])
        pm = _attach_attention_lora(sr, perturb=False)
        leaves = {
            n.split(".lora")[0].split(".")[-1]
            for n, p in pm.named_parameters() if p.requires_grad
        }
        assert leaves == {"q_proj", "o_proj", "cross_stream"}, leaves


# ── Dual-stream / gradient checkpointing ──────────────────────────


class TestDualStreamAndCheckpointing:
    def test_dual_stream_active_and_finite(self, tiny_moe_base):
        """A perturbed adapter drives the dual-stream forward; output is finite
        and differs from the frozen base (the adapter actually fires)."""
        sr = build_sr_base(tiny_moe_base["path"])
        pm = _attach_attention_lora(sr, perturb=True).eval()
        with torch.no_grad():
            dual = pm(input_ids=tiny_moe_base["ids"]).logits
        assert torch.isfinite(dual).all()
        assert (dual - tiny_moe_base["stock_logits"]).abs().max() > 1e-4

    def test_gradient_checkpoint_recompute_is_exact(self, tiny_moe_base):
        """MoE routing is deterministic given the input, so a checkpointed
        dual-stream forward must produce the same loss/grads as a non-checkpointed
        one."""
        ids = tiny_moe_base["ids"]
        labels = ids.clone()

        def run(gc: bool):
            sr = build_sr_base(tiny_moe_base["path"])
            pm = _attach_attention_lora(sr, perturb=True)
            pm.train()
            if gc:
                pm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            out = pm(input_ids=ids, labels=labels)
            out.loss.backward()
            grads = {
                n: p.grad.detach().clone()
                for n, p in pm.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            return out.loss.detach().clone(), grads

        torch.manual_seed(1)
        loss_gc, grads_gc = run(True)
        torch.manual_seed(1)
        loss_no, grads_no = run(False)

        torch.testing.assert_close(loss_gc, loss_no, atol=1e-5, rtol=1e-4)
        assert grads_gc.keys() == grads_no.keys()
        for k in grads_gc:
            torch.testing.assert_close(grads_gc[k], grads_no[k], atol=1e-4, rtol=1e-3)


# ── Shared MoE routing (config.share_moe_routing) ─────────────────


class TestSharedMoeRouting:
    """Shared routing: route once on the base stream and reuse that expert
    partition for the adapter stream. ON by default; set share_moe_routing=False
    to route each stream independently. The base stream still routes on itself,
    so the frozen-base invariant holds in both modes."""

    def test_default_is_shared_routing(self, tiny_moe_base):
        """Default build enables shared routing on every layer."""
        sr = build_sr_base(tiny_moe_base["path"])
        assert all(layer.share_moe_routing is True for layer in sr.model.layers)

    def test_can_opt_out_to_independent_routing(self, tiny_moe_base):
        """share_moe_routing=False flows to every layer (independent routing)."""
        sr = build_sr_base(tiny_moe_base["path"], share_moe_routing=False)
        assert all(layer.share_moe_routing is False for layer in sr.model.layers)

    def test_flag_flows_to_every_layer(self, tiny_moe_base):
        sr = build_sr_base(tiny_moe_base["path"], share_moe_routing=True)
        assert all(layer.share_moe_routing is True for layer in sr.model.layers)

    def test_bare_forward_matches_stock_with_sharing(self, tiny_moe_base):
        """Sharing routes the (only) stream on itself in bare mode, so bare
        forward still reproduces stock logits."""
        sr = build_sr_base(tiny_moe_base["path"], share_moe_routing=True).eval()
        with torch.no_grad():
            sr_logits = sr(input_ids=tiny_moe_base["ids"]).logits
        torch.testing.assert_close(
            sr_logits, tiny_moe_base["stock_logits"], atol=1e-4, rtol=1e-4,
        )

    def test_base_stream_matches_stock_with_sharing(self, tiny_moe_base):
        """Frozen-base invariant under sharing: the base stream routes on
        normed_base regardless, so a disabled adapter still matches stock."""
        sr = build_sr_base(tiny_moe_base["path"], share_moe_routing=True)
        pm = _attach_attention_lora(sr, perturb=True).eval()
        with torch.no_grad():
            with pm.disable_adapter():
                disabled = pm(input_ids=tiny_moe_base["ids"]).logits
        torch.testing.assert_close(
            disabled, tiny_moe_base["stock_logits"], atol=1e-4, rtol=1e-4,
        )

    def test_shared_differs_from_independent_on_adapter(self, tiny_moe_base):
        """The flag actually changes the adapter stream: with the same perturbed
        adapter, shared vs. independent routing produce different dual-stream
        logits (the adapter stream is routed differently)."""
        ids = tiny_moe_base["ids"]

        def dual_logits(share: bool):
            sr = build_sr_base(tiny_moe_base["path"], share_moe_routing=share)
            pm = _attach_attention_lora(sr, perturb=True).eval()
            with torch.no_grad():
                return pm(input_ids=ids).logits

        torch.manual_seed(3)
        shared = dual_logits(True)
        torch.manual_seed(3)
        indep = dual_logits(False)
        assert torch.isfinite(shared).all()
        assert (shared - indep).abs().max() > 1e-4

    def test_gradient_checkpoint_recompute_is_exact_with_sharing(self, tiny_moe_base):
        """Routing on normed_base is deterministic, so a checkpointed shared-routing
        forward matches a non-checkpointed one in loss and grads."""
        ids = tiny_moe_base["ids"]
        labels = ids.clone()

        def run(gc: bool):
            sr = build_sr_base(tiny_moe_base["path"], share_moe_routing=True)
            pm = _attach_attention_lora(sr, perturb=True)
            pm.train()
            if gc:
                pm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            out = pm(input_ids=ids, labels=labels)
            out.loss.backward()
            grads = {
                n: p.grad.detach().clone()
                for n, p in pm.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            return out.loss.detach().clone(), grads

        torch.manual_seed(1)
        loss_gc, grads_gc = run(True)
        torch.manual_seed(1)
        loss_no, grads_no = run(False)

        torch.testing.assert_close(loss_gc, loss_no, atol=1e-5, rtol=1e-4)
        assert grads_gc.keys() == grads_no.keys()
        for k in grads_gc:
            torch.testing.assert_close(grads_gc[k], grads_no[k], atol=1e-4, rtol=1e-3)
