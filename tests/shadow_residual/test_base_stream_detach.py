"""Training must not backpropagate through the frozen base stream (perf only)."""
import pytest
import torch
from peft import LoraConfig, get_peft_model

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.config_helpers import set_shadow_residual
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig

TARGETS = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "cross_stream"]
LAYERS = 4


def _model():
    torch.manual_seed(0)
    cfg = ShadowResidualConfig(
        vocab_size=1000, hidden_size=256, intermediate_size=1024,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
    )
    cfg._attn_implementation = "sdpa"
    set_shadow_residual(config=cfg, enabled=True)
    model = get_peft_model(
        ShadowResidualForCausalLM(cfg),
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=TARGETS),
    )
    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    return model


def _batch():
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1000, (2, 64), generator=gen)
    labels = ids.clone()
    labels[:, :10] = -100
    return ids, labels


def _layers(model):
    return model.base_model.model.model.layers


def _k_proj_tracked(model, train, embeds_require_grad=False):
    """Forward only; whether autograd tracks the frozen k_proj input per layer."""
    ids, _ = _batch()
    tracked = []
    hooks = [layer.self_attn.k_proj.register_forward_pre_hook(
                 lambda module, args: tracked.append(args[0].requires_grad))
             for layer in _layers(model)]
    kwargs = {}
    if embeds_require_grad:
        emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        kwargs["inputs_embeds"] = emb
    else:
        kwargs["input_ids"] = ids
    try:
        model.train(train)
        model(use_cache=False, **kwargs)
    finally:
        for h in hooks:
            h.remove()
    return tracked


def _lora_grads(model, train, ckpt):
    ids, labels = _batch()
    model.train(train)
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    out = model(input_ids=ids, labels=labels, use_cache=False)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    return grads, out.loss.detach()


def test_training_forward_does_not_track_frozen_base_ops():
    m = _model()
    m.train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    assert _k_proj_tracked(m, train=True) == [False] * 4


@pytest.mark.parametrize("ckpt", [False, True])
def test_training_lora_grads_equal_eval_path(ckpt):
    # eval mode never detaches, so it is the stock path; dropout is 0, so forwards match
    g_train, loss_train = _lora_grads(_model(), train=True, ckpt=ckpt)
    g_eval, loss_eval = _lora_grads(_model(), train=False, ckpt=False)
    assert torch.equal(loss_train, loss_eval)
    assert all(torch.equal(g_train[n], g_eval[n]) for n in g_eval) and len(g_eval) == 48


def test_eval_mode_still_tracks_base_stream():
    m = _model()
    m.eval()
    assert _k_proj_tracked(m, train=False, embeds_require_grad=True) == [True] * 4
