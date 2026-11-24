# SPDX-License-Identifier: Apache-2.0
"""Training-free compression of trained LoRA A matrices (compress_a / compress_adapters / train_heal), CPU."""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.config_helpers import set_shadow_residual
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.training.compress_a import (
    kaiming_scale_a,
    load_adapter,
    quantize_rows,
    refit_b,
    relative_error,
    shared_row_space,
    structured_bank,
)
from shadow_residual.training.compress_adapters import compress_all, parse_variant, slug, write_outputs
from shadow_residual.training.compute_frozen_a import collect
from shadow_residual.training.frozen_a import A_FILE, _damped_cholesky, install_frozen_a

SITES = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "cross_stream"]
RANK = 4
G = torch.Generator().manual_seed(0)


def _cov(d=24, n=4000, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g) @ torch.randn(d, d, generator=g)
    return (x.T @ x / n).double()


def _factor(out=16, d=24, r=RANK, s=2.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(out, r, generator=g).double(), torch.randn(r, d, generator=g).double() / math.sqrt(d), s


def test_refit_b_is_exact_when_new_a_spans_old_a():
    b, a, s = _factor()
    mix = torch.randn(RANK, RANK, generator=G).double()
    a_new = mix @ a
    b_new = refit_b(b, a, s, a_new, cov=_cov())
    assert relative_error(b, a, s, b_new, a_new, 2.0, None) < 1e-10


def test_shared_row_space_full_rank_is_lossless_for_every_adapter():
    fs = [_factor(seed=i) for i in range(3)]
    cov = _cov()
    S = _damped_cholesky(cov, 1e-6)
    for c in (None, cov):
        a_new, ret = shared_row_space(fs, 3 * RANK, cov=c)
        assert ret == pytest.approx(1.0, abs=1e-12)
        for b, a, s in fs:
            b_new = refit_b(b, a, s, a_new, cov=c)
            assert relative_error(b, a, s, b_new, a_new, 2.0, S if c is not None else None) < 1e-10


def test_whitened_truncation_is_optimal_for_output_error():
    fac = [_factor(seed=5)]
    cov = _cov()
    S = _damped_cholesky(cov, 1e-6)
    b, a, s = fac[0]
    errs = {}
    for name, c in (("white", cov), ("frob", None)):
        a_new, ret = shared_row_space(fac, 2, cov=c)
        b_new = refit_b(b, a, s, a_new, cov=cov)  # best B for output error in both cases
        errs[name] = relative_error(b, a, s, b_new, a_new, 2.0, S)
        if c is not None:
            assert errs[name] == pytest.approx(1 - ret, abs=1e-8)
    assert errs["white"] <= errs["frob"] + 1e-12
    a_f, ret_f = shared_row_space(fac, 2, cov=None)
    assert relative_error(b, a, s, refit_b(b, a, s, a_f), a_f, 2.0, None) == pytest.approx(1 - ret_f, abs=1e-8)


def test_shared_row_space_rejects_rank_above_joint_dimension():
    with pytest.raises(ValueError):
        shared_row_space([_factor()], RANK + 1)


def test_astack_ignores_b():
    b, a, s = _factor()
    a1, _ = shared_row_space([(b, a, s)], 2, use_b=False)
    a2, _ = shared_row_space([(b * torch.linspace(1, 9, b.shape[1]).double(), a, s)], 2, use_b=False)
    assert torch.allclose(a1.abs(), a2.abs(), atol=1e-10)


def test_kaiming_scale_matches_stock_lora_feature_energy():
    cov = _cov()
    a_dir, _ = shared_row_space([_factor()], RANK, cov=cov)
    a = kaiming_scale_a(a_dir, cov)
    want = float(cov.diagonal().sum()) / (3 * cov.shape[0])
    assert torch.allclose(a @ cov @ a.T, want * torch.eye(RANK, dtype=torch.float64), atol=1e-5 * want)
    a_o, _ = shared_row_space([_factor()], RANK)
    assert torch.allclose(kaiming_scale_a(a_o, None).norm(dim=1), torch.full((RANK,), 1 / math.sqrt(3)).double())


def test_quantize_rows_levels_and_error_order():
    a = (torch.rand(8, 256, generator=G) * 2 - 1).double()
    errs = {}
    for mode, max_levels in (("int8", 255), ("int4", 15), ("int3", 7), ("int2", 3), ("sign", 2)):
        q = quantize_rows(a, mode)
        for row in q:
            assert row.unique().numel() <= max_levels
        errs[mode] = float((q - a).pow(2).sum() / a.pow(2).sum())
    assert errs["int8"] < errs["int4"] < errs["int3"] < errs["int2"] < errs["sign"] < 1
    assert quantize_rows(a, "sign").abs().unique().numel() <= 8  # one magnitude per row
    with pytest.raises(ValueError):
        quantize_rows(a, "int1")


@pytest.mark.parametrize("kind", ["uniform", "sign", "sparse", "gather"])
def test_structured_bank_shapes_energy_and_sharing(kind):
    shapes = {f"layers.{l}.{s}": (8, 4096 if s == "down_proj" else 1024) for l in range(3) for s in SITES}
    bank = structured_bank(shapes, kind, seed=3)
    assert {k: tuple(v.shape) for k, v in bank.items()} == shapes
    for l in range(3):
        assert torch.equal(bank[f"layers.{l}.gate_proj"], bank[f"layers.{l}.up_proj"])
    a = bank["layers.0.q_proj"]
    rows = torch.cat([v for v in bank.values() if v.shape[1] == 1024])
    assert rows.pow(2).sum(1).mean() == pytest.approx(1 / 3, rel=0.1)  # kaiming row energy
    if kind == "sign":
        assert a.abs().unique().numel() == 1
    if kind == "sparse":
        assert (a != 0).float().mean() == pytest.approx(1 / 32, rel=0.2)
    if kind == "gather":
        assert ((a != 0).sum(1) == 1).all() and (a != 0).any(0).sum() == 8
    assert not torch.equal(bank["layers.0.q_proj"], bank["layers.1.q_proj"])
    shared = structured_bank(shapes, kind, seed=3, cross_layer=True)
    for s in SITES:
        assert torch.equal(shared[f"layers.0.{s}"], shared[f"layers.2.{s}"])
    if kind == "gather":
        with pytest.raises(ValueError, match="dense"):
            structured_bank(shapes, kind, seed=3, whole_model=True)
        return
    whole = structured_bank(shapes, kind, seed=3, whole_model=True)
    assert {k: tuple(v.shape) for k, v in whole.items()} == shapes
    base = whole["layers.0.down_proj"]
    for k, v in whole.items():
        torch.testing.assert_close(v, base[:, : v.shape[1]] * (4096 / v.shape[1]) ** 0.5, rtol=0, atol=0)
    narrow = torch.cat([v for v in whole.values() if v.shape[1] == 1024])
    assert narrow.pow(2).sum(1).mean() == pytest.approx(1 / 3, rel=0.15)
    assert base.pow(2).sum(1).mean() == pytest.approx(1 / 3, rel=0.15)


def test_bank_cli_writes_a_bank_install_frozen_a_accepts(tiny_sr, tmp_path):
    from safetensors.torch import save_file

    from shadow_residual.training.compress_a import main as bank_main
    from shadow_residual.training.frozen_a import MANIFEST_FILE

    model = get_peft_model(copy.deepcopy(tiny_sr), LoraConfig(
        r=2, lora_alpha=4, target_modules=SITES, lora_dropout=0.0))
    ref = tmp_path / "ref"
    ref.mkdir()
    from shadow_residual.training.frozen_a import site_key

    save_file({site_key(n.split(".lora_A.")[0]): torch.zeros(RANK, p.shape[1])
               for n, p in model.named_parameters() if ".lora_A." in n}, str(ref / A_FILE))
    out = tmp_path / "bank"
    assert bank_main(["--shapes-from", str(ref), "--kind", "sign", "--rank", "2", "--cross-layer",
                      "--out", str(out)]) == 0
    bank = load_file(str(out / A_FILE))
    assert all(v.shape[0] == 2 for v in bank.values())
    assert torch.equal(bank["layers.0.q_proj"], bank["layers.1.q_proj"])
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    assert manifest["rank"] == 2
    assert manifest["tensors"] == 5  # cross-layer, gate/up shared: q, o, gate=up, down, cross
    assert install_frozen_a(model, out)["frozen"] == 2 * len(SITES)


def test_parse_variant():
    assert parse_variant("dw-w:32") == ("dw-w", "32")
    assert parse_variant("dwq:32-sign") == ("dwq", "32-sign")
    assert slug("q:int4") == "q-int4"
    for bad in ("dw", "nope:3", "dw:x", "dwq:32-int1", "dwq:x-sign"):
        with pytest.raises(ValueError):
            parse_variant(bad)


# ----------------------------------------------------------------------------- tiny SR end to end


@pytest.fixture
def tiny_sr():
    cfg = ShadowResidualConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_adapters=0, max_lora_rank=4,
        switch_head_dim=16,
    )
    set_shadow_residual(config=cfg, enabled=True)
    torch.manual_seed(0)
    return ShadowResidualForCausalLM(cfg).eval()


def _batches(n=6, seq=20):
    g = torch.Generator().manual_seed(2)
    for _ in range(n):
        ids = torch.randint(0, 300, (2, seq), generator=g)
        mask = torch.ones_like(ids)
        mask[1, seq - 5:] = 0
        yield ids, mask


def _train_like_adapter(tiny_sr, out, seed):
    model = get_peft_model(copy.deepcopy(tiny_sr), LoraConfig(
        r=RANK, lora_alpha=2 * RANK, target_modules=SITES, lora_dropout=0.0,
        rank_pattern={s: RANK for s in SITES}))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_B." in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    model.save_pretrained(str(out))
    return out


@pytest.fixture
def adapters(tiny_sr, tmp_path):
    return {n: _train_like_adapter(tiny_sr, tmp_path / n, seed=i) for i, n in enumerate(("t1", "t2"))}


def _logits(tiny_sr, adapter_dir, ids):
    m = PeftModel.from_pretrained(copy.deepcopy(tiny_sr), str(adapter_dir)).eval()
    with torch.no_grad():
        return m(input_ids=ids).logits


VARIANTS = ["astack:4", "dw:4", "dw-w:4", "dw-w:8", "dw-w-gu:8", "q:int4", "q:sign", "qr:int2", "tw:2",
            "q:int8", "qr:int8", "q:int2", "dwq:4-sign"]


@pytest.fixture
def compressed(tiny_sr, adapters, tmp_path):
    collector = collect(tiny_sr, _batches())
    loaded = {n: load_adapter(d)[1] for n, d in adapters.items()}
    out, banks, stats = compress_all(loaded, VARIANTS, collector.covariance)
    root = tmp_path / "out"
    write_outputs(root, adapters, out, banks, stats)
    return root, stats


def test_compress_all_writes_loadable_adapters_with_shared_a(tiny_sr, adapters, compressed):
    root, stats = compressed
    ids = torch.randint(0, 300, (2, 12), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        base = tiny_sr(input_ids=ids).logits
    for spec in VARIANTS:
        method, arg = parse_variant(spec)
        t1 = load_file(str(root / slug(spec) / "t1" / "adapter_model.safetensors"))
        t2 = load_file(str(root / slug(spec) / "t2" / "adapter_model.safetensors"))
        cfg = json.loads((root / slug(spec) / "t1" / "adapter_config.json").read_text())
        rank = int(arg.partition("-")[0]) if method not in ("q", "qr") else RANK
        assert cfg["r"] == rank and cfg["lora_alpha"] == 2 * rank and cfg["a_compression"]["variant"] == spec
        a_keys = [k for k in t1 if ".lora_A." in k]
        assert len(a_keys) == 2 * len(SITES)
        if method in ("astack", "dw", "dw-w", "dw-w-gu", "dwq"):
            assert all(torch.equal(t1[k], t2[k]) for k in a_keys), spec
            bank = load_file(str(root / slug(spec) / "bank" / A_FILE))
            assert set(bank) == {f"layers.{l}.{s}" for l in range(2) for s in SITES}
        else:
            assert not (root / slug(spec) / "bank").exists()
        gate = [k for k in a_keys if "gate_proj" in k][0]
        assert torch.equal(t1[gate], t1[gate.replace("gate_proj", "up_proj")]) == (method == "dw-w-gu"), spec
        # stock PEFT loads it on a bare SR base; disable_adapter is still the base
        m = PeftModel.from_pretrained(copy.deepcopy(tiny_sr), str(root / slug(spec) / "t1")).eval()
        with torch.no_grad():
            with m.disable_adapter():
                assert torch.allclose(m(input_ids=ids).logits, base, atol=1e-5)

    # full joint rank (2 adapters x r=4) is lossless: same logits as the originals
    for n in ("t1", "t2"):
        ref = _logits(tiny_sr, adapters[n], ids)
        assert not torch.allclose(ref, base, atol=1e-3)
        got = _logits(tiny_sr, root / "dw-w-8" / n, ids)
        assert torch.allclose(got, ref, atol=2e-3), (got - ref).abs().max()
    for site, v in stats["dw-w:8"]["per_site"].items():
        assert v["rel_err_white"] < 1e-3, site
    assert stats["dw-w:8"]["retained_mean"] == pytest.approx(1.0, abs=1e-6)
    # at half rank the whitened variant has the lower output error, the Frobenius one the lower weight error
    for site in SITES:
        w, f = stats["dw-w:4"]["per_site"][site], stats["dw:4"]["per_site"][site]
        assert w["rel_err_white"] <= f["rel_err_white"] + 1e-9, site
        assert f["rel_err_frob"] <= w["rel_err_frob"] + 1e-9, site


def test_dwq_shares_a_one_bit_a_and_bank_matches(compressed):
    root, stats = compressed
    bank = load_file(str(root / "dwq-4-sign" / "bank" / A_FILE))
    t1 = load_file(str(root / "dwq-4-sign" / "t1" / "adapter_model.safetensors"))
    for k, v in t1.items():
        if ".lora_A." in k:
            for row in v.float():
                assert row.abs().unique().numel() == 1
    from shadow_residual.training.frozen_a import saved_a_mismatches
    assert saved_a_mismatches(root / "dwq-4-sign" / "t1", root / "dwq-4-sign" / "bank") == []
    assert len(bank) == 2 * len(SITES)
    for site in SITES:  # refit after quantising beats the unrefit 1-bit per-adapter A
        assert stats["dwq:4-sign"]["per_site"][site]["rel_err_white"] < 1.0


def test_quantised_refit_is_never_worse_and_int8_is_near_lossless(compressed):
    _, stats = compressed
    for site in SITES:
        for mode in ("int8", "int2"):
            q, qr = stats[f"q:{mode}"]["per_site"][site], stats[f"qr:{mode}"]["per_site"][site]
            assert qr["rel_err_white"] <= q["rel_err_white"] + 1e-9, (site, mode)
        assert stats["qr:int8"]["per_site"][site]["rel_err_white"] < 1e-3, site


def test_source_scaling_other_than_two_is_rescaled(tiny_sr, tmp_path):
    model = get_peft_model(copy.deepcopy(tiny_sr), LoraConfig(
        r=RANK, lora_alpha=RANK, target_modules=SITES, lora_dropout=0.0))  # s = 1
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_B." in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    src = tmp_path / "s1"
    model.save_pretrained(str(src))
    loaded = {"x": load_adapter(src)[1]}
    assert all(d["s"] == 1.0 for d in loaded["x"].values())
    out, banks, stats = compress_all(loaded, ["dw:4", "q:int8"])
    write_outputs(tmp_path / "o", {"x": src}, out, banks, stats)
    ids = torch.randint(0, 300, (2, 12), generator=torch.Generator().manual_seed(5))
    ref = _logits(tiny_sr, src, ids)
    for v in ("dw-4", "q-int8"):
        assert torch.allclose(_logits(tiny_sr, tmp_path / "o" / v / "x", ids), ref, atol=3e-3), v


def test_heal_rejects_b_from_a_different_bank(compressed):
    from shadow_residual.training import train_heal

    root, _ = compressed
    with pytest.raises(ValueError, match="not compressed onto bank"):
        train_heal._patch_factory(str(root / "dw-4" / "bank"), str(root / "dw-w-4" / "t1"))


def test_cli_main_writes_variants_and_rejects_whitened_without_calibration(adapters, tmp_path):
    from shadow_residual.training.compress_adapters import main

    args = ["--base-model", "unused", "--adapters"] + [f"{n}={d}" for n, d in adapters.items()]
    assert main(args + ["--variants", "dw:4", "q:int4", "--out-root", str(tmp_path / "o")]) == 0
    for v in ("dw-4", "q-int4"):
        for n in adapters:
            assert (tmp_path / "o" / v / n / "adapter_model.safetensors").exists()
            assert (tmp_path / "o" / v / n / "adapter_config.json").exists()
    assert (tmp_path / "o" / "dw-4" / "bank" / A_FILE).exists()
    assert set(json.loads((tmp_path / "o" / "stats.json").read_text())) == {"dw:4", "q:int4"}
    with pytest.raises(ValueError, match="cov_fn"):
        main(args + ["--variants", "dw-w:4", "--out-root", str(tmp_path / "o2")])


def test_quantised_variant_keeps_b_and_quantises_a(adapters, compressed):
    root, _ = compressed
    src = load_file(str(adapters["t1"] / "adapter_model.safetensors"))
    q = load_file(str(root / "q-sign" / "t1" / "adapter_model.safetensors"))
    for k, v in src.items():
        if ".lora_B." in k:
            assert torch.allclose(q[k].float(), v.float(), rtol=1e-2, atol=1e-6)
        else:
            assert q[k].float().abs().unique().numel() <= RANK


def test_heal_installs_bank_and_b_to_reproduce_compressed_adapter(tiny_sr, compressed):
    from shadow_residual.training.train_heal import install_b

    root, _ = compressed
    ids = torch.randint(0, 300, (2, 12), generator=torch.Generator().manual_seed(4))
    ref = _logits(tiny_sr, root / "dw-w-4" / "t2", ids)
    model = get_peft_model(copy.deepcopy(tiny_sr), LoraConfig(
        r=RANK, lora_alpha=2 * RANK, target_modules=SITES, lora_dropout=0.0))
    install_frozen_a(model, root / "dw-w-4" / "bank")
    assert install_b(model, root / "dw-w-4" / "t2") == 2 * len(SITES)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, ref, atol=1e-4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_B." in n for n in trainable)

    wrong = get_peft_model(copy.deepcopy(tiny_sr), LoraConfig(
        r=RANK, lora_alpha=RANK, target_modules=SITES, lora_dropout=0.0))
    with pytest.raises(ValueError, match="scaling"):
        install_b(wrong, root / "dw-w-4" / "t2")


def test_heal_entry_point_patches_the_factory(tiny_sr, compressed, monkeypatch):
    from shadow_residual.training import factory, train_heal

    root, _ = compressed
    monkeypatch.setattr(factory, "get_shadow_residual_peft_model", lambda *a, **k: get_peft_model(
        copy.deepcopy(tiny_sr), LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=SITES, lora_dropout=0.0)))
    train_heal._patch_factory(str(root / "dw-4" / "bank"), str(root / "dw-4" / "t1"))
    model = factory.get_shadow_residual_peft_model("ignored", None)
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".lora_A." in n)
    b = load_file(str(root / "dw-4" / "t1" / "adapter_model.safetensors"))
    got = {n: p for n, p in model.named_parameters() if ".lora_B." in n}
    for k, v in b.items():
        if ".lora_B." in k:
            name = k.replace(".lora_B.weight", ".lora_B.default.weight")
            assert torch.equal(got[name].detach().to(v.dtype), v)
