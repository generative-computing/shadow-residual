# SPDX-License-Identifier: Apache-2.0
"""SR-FrozenA: compression-derived frozen LoRA A, only B trains (CPU)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from shadow_residual.shadow_residual import ShadowResidualForCausalLM
from shadow_residual.shadow_residual.config_helpers import set_shadow_residual
from shadow_residual.shadow_residual.model_config import ShadowResidualConfig
from shadow_residual.training.compute_frozen_a import collect, compute_a
from shadow_residual.training.compute_weight_a import compute_weight_bank, site_seed
from shadow_residual.training.frozen_a import (
    A_FILE,
    GROUP_SITES,
    MANIFEST_FILE,
    _kmeans_centroids,
    compressed_rows_a,
    install_frozen_a,
    pca_a,
    randomize_lora_b,
    row_space_stats,
    saved_a_mismatches,
    site_key,
    weight_svd_a,
    whitened_svd_a,
)
from shadow_residual.training.rescale_frozen_a import out_name, plan_outputs, rescale_bank

SITES = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "cross_stream"]
RANK = 4


def _cov(n=3000, d=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g) @ torch.randn(d, d, generator=g)
    return x, (x.T @ x / n).double()


def test_whitened_svd_a_is_whitening_and_beats_plain_svd():
    x, c = _cov()
    w = torch.randn(16, 24, generator=torch.Generator().manual_seed(1))
    a, energy = whitened_svd_a(w, c, RANK)
    assert torch.allclose(a @ c @ a.T, torch.eye(RANK, dtype=torch.float64), atol=1e-6)

    def out_err(proj_rows):
        # best output reconstruction W·x ≈ M·(P x) on the data, P = proj_rows
        z = x.double() @ proj_rows.T
        y = x.double() @ w.double().T
        m = torch.linalg.lstsq(z, y).solution
        return float(((y - z @ m) ** 2).sum() / (y ** 2).sum())

    err_white = out_err(a)
    plain_rows = torch.linalg.svd(w.double(), full_matrices=False).Vh[:RANK]
    assert err_white <= out_err(plain_rows) + 1e-9
    assert abs(err_white - (1 - energy)) < 1e-6


def test_pca_a_unit_variance():
    _, c = _cov()
    a, energy = pca_a(c, RANK)
    assert torch.allclose(a @ c @ a.T, torch.eye(RANK, dtype=torch.float64), atol=1e-8)
    assert 0 < energy < 1


def test_site_key_maps_peft_and_saved_names():
    assert site_key("base_model.model.model.layers.7.self_attn.q_proj") == "layers.7.q_proj"
    assert site_key("base_model.model.model.layers.0.mlp.down_proj") == "layers.0.down_proj"
    assert site_key("base_model.model.model.layers.2.cross_stream") == "layers.2.cross_stream"
    assert site_key("base_model.model.lm_head") is None


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


@pytest.fixture
def frozen_a_dir(tiny_sr, tmp_path):
    collector = collect(tiny_sr, _batches())
    tensors, stats = compute_a(tiny_sr, collector, RANK)
    save_file(tensors, str(tmp_path / A_FILE))
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(
        {"rank": RANK, "recipe": {}, "feature_scale": "kaiming", "per_group": stats}))
    return tmp_path


def test_compute_a_covers_every_site_and_shares_gate_up(tiny_sr, frozen_a_dir):
    t = load_file(str(frozen_a_dir / A_FILE))
    n_layers = len(tiny_sr.model.layers)
    assert set(t) == {f"layers.{i}.{s}" for i in range(n_layers) for s in SITES}
    for i in range(n_layers):
        assert torch.equal(t[f"layers.{i}.gate_proj"], t[f"layers.{i}.up_proj"])
        assert t[f"layers.{i}.down_proj"].shape[0] == RANK
        assert torch.isfinite(t[f"layers.{i}.q_proj"]).all()
    stats = json.loads((frozen_a_dir / MANIFEST_FILE).read_text())["per_group"]
    # padding positions are excluded: 2 rows x 20 - 5 padded per batch x 6 batches
    assert stats["layers.0.q"]["tokens"] == 6 * 35
    assert max(v["whiteness_err"] for v in stats.values()) < 1e-4


def test_weight_svd_a_picks_minor_or_principal_directions():
    w = torch.randn(16, 24, generator=torch.Generator().manual_seed(3)).double()
    sig = torch.linalg.svdvals(w)
    minor, e_minor = weight_svd_a(w, RANK, "minor")
    principal, e_principal = weight_svd_a(w, RANK, "principal")
    for a in (minor, principal):
        assert torch.allclose(a @ a.T, torch.eye(RANK, dtype=torch.float64) / 3, atol=1e-10)
    # ‖W Aᵀ‖² = (1/3)·Σ σ² over the chosen directions
    assert torch.isclose((w @ minor.T).pow(2).sum() * 3, sig[-RANK:].pow(2).sum())
    assert torch.isclose((w @ principal.T).pow(2).sum() * 3, sig[:RANK].pow(2).sum())
    assert e_minor < e_principal


def _rows_in_span(a, w):
    """Max residual of A's rows after projection onto the row space of W (0 = inside)."""
    q = torch.linalg.qr(w.double().T).Q
    a = a.double()
    return float((a - (a @ q) @ q.T).abs().max())


def test_compressed_rows_a_prune_keeps_largest_l1_rows():
    w = torch.randn(40, 24, generator=torch.Generator().manual_seed(4))
    w[[3, 17, 29, 35]] *= 10
    a, st = compressed_rows_a(w, RANK, "prune")
    expected = w[[3, 17, 29, 35]].double()
    expected = expected / expected.norm(dim=1, keepdim=True) / math.sqrt(3)
    assert torch.allclose(a, expected, atol=1e-6)
    assert 0 < st["energy"] < 1 and st["rank"] == RANK


def test_compressed_rows_a_fold_recovers_clusters():
    g = torch.Generator().manual_seed(5)
    centers = torch.randn(RANK, 24, generator=g) * 5
    w = centers.repeat_interleave(10, dim=0) + 0.01 * torch.randn(10 * RANK, 24, generator=g)
    a, _ = compressed_rows_a(w, RANK, "fold", seed=0)
    target = (centers / centers.norm(dim=1, keepdim=True) / math.sqrt(3)).double()
    # every true cluster center is matched by one centroid row (order is arbitrary)
    match = (a @ target.T * 3).max(dim=0).values
    assert bool((match > 0.999).all())


def test_compressed_rows_a_sketch_is_seeded_and_in_row_space():
    w = torch.randn(10, 24, generator=torch.Generator().manual_seed(6))
    a1, e1 = compressed_rows_a(w, RANK, "sketch", seed=7)
    a2, _ = compressed_rows_a(w, RANK, "sketch", seed=7)
    a3, _ = compressed_rows_a(w, RANK, "sketch", seed=8)
    assert torch.equal(a1, a2) and not torch.equal(a1, a3)
    assert _rows_in_span(a1, w) < 1e-6
    assert 0 < e1["energy"] < 1


def test_row_space_energy_known_answers():
    # sketch with r >= rank(W) spans all of W's row space: energy 1
    w = torch.randn(3, 24, generator=torch.Generator().manual_seed(10))
    _, st = compressed_rows_a(w, RANK, "sketch")
    assert abs(st["energy"] - 1) < 1e-6 and st["rank"] == 3
    # W with orthogonal rows of norms 3, 2, 1, 1, 1: keeping the 2 largest-L1 rows keeps 13/16
    w = torch.zeros(5, 8)
    for j, s in enumerate([3.0, 2.0, 1.0, 1.0, 1.0]):
        w[j, j] = s
    _, st = compressed_rows_a(w, 2, "prune")
    assert abs(st["energy"] - 13 / 16) < 1e-9 and st["rank"] == 2 and abs(st["cond"] - 1) < 1e-9


def test_row_space_stats_counts_duplicate_rows_once():
    w = torch.randn(6, 24, generator=torch.Generator().manual_seed(11))
    w = torch.cat([w[:1].repeat(4, 1) * 5, w[1:]])
    a, st = compressed_rows_a(w, RANK, "prune")
    assert st["rank"] == 1
    assert abs(st["energy"] - row_space_stats(w[:1], w)["energy"]) < 1e-9


def test_compressed_rows_a_rejects_zero_rows_and_too_few_distinct_rows():
    w = torch.zeros(10, 24)
    w[0] = 1.0
    with pytest.raises(ValueError, match="zero-norm"):
        compressed_rows_a(w, RANK, "prune")
    with pytest.raises(ValueError, match="distinct rows"):
        compressed_rows_a(w.repeat(2, 1) + 1, RANK, "fold")


def test_kmeans_reseeds_empty_clusters_at_distinct_rows(monkeypatch):
    x = torch.randn(30, 6, generator=torch.Generator().manual_seed(12))
    # force k-means++ to pick row 0 for every center, so all but one cluster start empty
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.tensor([0]))
    monkeypatch.setattr(torch, "multinomial", lambda *a, **k: torch.tensor([0]))
    centers = _kmeans_centroids(x, RANK, torch.Generator().manual_seed(0))
    assert torch.unique(centers, dim=0).shape[0] == RANK


def test_compressed_rows_a_accepts_bf16_weights():
    w = torch.randn(64, 24, generator=torch.Generator().manual_seed(13)).to(torch.bfloat16)
    for method in ("prune", "fold", "sketch"):
        a, st = compressed_rows_a(w, RANK, method)
        assert a.dtype == torch.float64 and bool(torch.isfinite(a).all()) and 0 < st["energy"] <= 1


def test_site_seeds_never_collide_across_banks():
    seeds = {site_seed(s, i, g) for s in range(3) for i in range(40) for g in range(4)}
    assert len(seeds) == 3 * 40 * 4


@pytest.mark.parametrize("method", ["prune", "fold", "sketch"])
def test_compressed_rows_a_rows_have_kaiming_norm(method):
    w = torch.randn(50, 24, generator=torch.Generator().manual_seed(9))
    a, _ = compressed_rows_a(w, RANK, method)
    assert a.shape == (RANK, 24)
    assert torch.allclose(a.norm(dim=1), torch.full((RANK,), 1 / math.sqrt(3), dtype=a.dtype), atol=1e-6)
    assert _rows_in_span(a, w) < 1e-6


def test_compressed_rows_a_rejects_unknown_method():
    with pytest.raises(ValueError):
        compressed_rows_a(torch.randn(8, 4), 2, "svd")


@pytest.mark.parametrize("method", ["minor", "principal", "prune", "fold", "sketch"])
def test_weight_bank_covers_every_site_from_weights_only(tiny_sr, frozen_a_dir, method):
    cross = load_file(str(frozen_a_dir / A_FILE))
    t, stats = compute_weight_bank(tiny_sr, RANK, method, cross)
    n_layers = len(tiny_sr.model.layers)
    assert set(t) == {f"layers.{i}.{s}" for i in range(n_layers) for s in SITES}
    for i in range(n_layers):
        layer = tiny_sr.model.layers[i]
        assert torch.equal(t[f"layers.{i}.gate_proj"], t[f"layers.{i}.up_proj"])
        assert torch.equal(t[f"layers.{i}.cross_stream"], cross[f"layers.{i}.cross_stream"])
        assert t[f"layers.{i}.down_proj"].shape == (RANK, layer.mlp.down_proj.in_features)
    assert all(0 < v["energy"] < 1 and v["rank"] == RANK for v in stats.values())


@pytest.mark.parametrize("method", ["fold", "sketch"])
def test_weight_bank_is_seeded(tiny_sr, frozen_a_dir, method):
    cross = load_file(str(frozen_a_dir / A_FILE))
    t0, _ = compute_weight_bank(tiny_sr, RANK, method, cross, seed=0)
    t0b, _ = compute_weight_bank(tiny_sr, RANK, method, cross, seed=0)
    t1, _ = compute_weight_bank(tiny_sr, RANK, method, cross, seed=1)
    assert all(torch.equal(t0[k], t0b[k]) for k in t0)
    assert not torch.equal(t0["layers.0.q_proj"], t1["layers.0.q_proj"])
    # q and o of one layer draw different sketches / k-means seeds
    assert not torch.equal(t0["layers.0.q_proj"], t0["layers.0.o_proj"])


def _peft(tiny_sr):
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=SITES, lora_dropout=0.0)
    return get_peft_model(tiny_sr, cfg)


def _same_row_space(a, b, tol=1e-6):
    qa = torch.linalg.qr(a.double().T).Q
    qb = torch.linalg.qr(b.double().T).Q
    return float((qa @ qa.T - qb @ qb.T).abs().max()) < tol


@pytest.mark.parametrize("mode", ["var", "white", "energy"])
@pytest.mark.parametrize("include_cross", [False, True])
def test_rescale_bank_modes_on_tiny_model(tiny_sr, frozen_a_dir, mode, include_cross):
    cross = load_file(str(frozen_a_dir / A_FILE))
    bank, _ = compute_weight_bank(tiny_sr, RANK, "fold", cross)
    collector = collect(tiny_sr, _batches())
    out, stats = rescale_bank(bank, collector, mode=mode, include_cross=include_cross)
    assert set(out) == set(bank)
    for i in range(len(tiny_sr.model.layers)):
        assert torch.equal(out[f"layers.{i}.gate_proj"], out[f"layers.{i}.up_proj"])
        for group, sites in GROUP_SITES.items():
            key = f"layers.{i}.{sites[0]}"
            assert out[key].dtype == torch.float32 and out[key].device.type == "cpu"
            if group == "cross" and not include_cross:
                assert torch.equal(out[key], bank[key])
                continue
            c = collector.covariance(i, group)
            a_new, a_old = out[key].double(), bank[key].double()
            gram = a_new @ c @ a_new.T
            target = float(c.diagonal().sum()) / (3 * c.shape[0])
            assert _same_row_space(a_new, a_old, tol=1e-4)
            if mode == "var":
                assert torch.allclose(gram.diagonal(), torch.full((RANK,), target, dtype=gram.dtype), rtol=1e-4)
                cos = torch.nn.functional.cosine_similarity(a_new, a_old, dim=1)
                assert bool((cos > 1 - 1e-6).all())
            elif mode == "white":
                assert torch.allclose(gram, target * torch.eye(RANK, dtype=gram.dtype), rtol=1e-3, atol=1e-4 * target)
            else:
                assert abs(float(gram.trace()) - RANK * target) < 1e-4 * RANK * target
                old = (a_old @ c @ a_old.T).diagonal()
                ratio = gram.diagonal() / old
                assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-4)
            assert stats[f"layers.{i}.{group}"]["var_spread_before"] >= 1


class _FakeCollector:
    """Covariances given directly: one channel with 10^4 x the variance of the others."""

    def __init__(self, n_layers, dims):
        self.layers = list(range(n_layers))
        self._c = {}
        for i in range(n_layers):
            for g, d in dims.items():
                diag = torch.ones(d, dtype=torch.float64)
                diag[0] = 1e4
                self._c[(i, g)] = torch.diag(diag)

    def covariance(self, i, g):
        return self._c[(i, g)]


def _fake_bank(n_layers=1, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    bank = {}
    for i in range(n_layers):
        for site in ["q_proj", "o_proj", "down_proj", "cross_stream"]:
            bank[f"layers.{i}.{site}"] = torch.randn(RANK, d, generator=g)
        a = torch.randn(RANK, d, generator=g)
        bank[f"layers.{i}.gate_proj"], bank[f"layers.{i}.up_proj"] = a, a.clone()
    return bank


def test_rescale_bank_tames_an_outlier_channel():
    bank = _fake_bank()
    bank["layers.0.q_proj"][:, 0] = 0
    bank["layers.0.q_proj"][0, 0] = 3.0  # only feature 0 reads the 10^4 x channel
    col = _FakeCollector(1, {g: 16 for g in GROUP_SITES})
    out, stats = rescale_bank(bank, col, mode="var")
    st = stats["layers.0.q"]
    assert st["var_spread_before"] > 1000 and abs(st["var_spread_after"] - 1) < 1e-6
    out_w, stats_w = rescale_bank(bank, col, mode="white")
    assert abs(stats_w["layers.0.q"]["var_spread_after"] - 1) < 1e-6
    _, stats_e = rescale_bank(bank, col, mode="energy")
    st_e = stats_e["layers.0.q"]
    assert abs(st_e["energy_ratio_vs_kaiming_after"] - 1) < 1e-9
    assert abs(st_e["var_spread_after"] - st_e["var_spread_before"]) < 1e-6 * st_e["var_spread_before"]


def test_rescale_bank_rejects_layer_mismatch_and_unshared_gate_up():
    col = _FakeCollector(2, {g: 16 for g in GROUP_SITES})
    with pytest.raises(ValueError, match="bank layers"):
        rescale_bank(_fake_bank(n_layers=1), col)
    with pytest.raises(ValueError, match="bank layers"):
        rescale_bank(_fake_bank(n_layers=3), col)
    bank = _fake_bank(n_layers=2)
    bank["layers.1.up_proj"] = bank["layers.1.up_proj"] + 1
    with pytest.raises(ValueError, match="gate_proj and up_proj"):
        rescale_bank(bank, col)
    with pytest.raises(ValueError, match="mode"):
        rescale_bank(_fake_bank(n_layers=2), col, mode="pca")


def test_rescale_bank_rejects_zero_variance_and_dependent_rows(tiny_sr, frozen_a_dir):
    bank = {k: v.clone() for k, v in load_file(str(frozen_a_dir / A_FILE)).items()}
    bank["layers.0.q_proj"][0] = 0
    with pytest.raises(ValueError, match="zero variance"):
        rescale_bank(bank, collect(tiny_sr, _batches()))
    bank = {k: v.clone() for k, v in load_file(str(frozen_a_dir / A_FILE)).items()}
    bank["layers.0.q_proj"][1] = bank["layers.0.q_proj"][0] * 2
    with pytest.raises(ValueError, match="singular"):
        rescale_bank(bank, collect(tiny_sr, _batches()), mode="white")


def test_rescale_out_name():
    assert out_name(Path("/x/frozen_a_prune_r32_v1"), "white") == "frozen_a_prune_r32_white_v1"


def test_plan_outputs_filters_and_rejects_unmatched_only():
    banks = [Path("/x/frozen_a_prune_r32_v1"), Path("/x/frozen_a_rand_r32_v1/")]
    assert len(plan_outputs(banks, ["var", "white"], None)) == 4
    plan = plan_outputs(banks, ["var", "white"], ["frozen_a_rand_r32_v1:white", "frozen_a_prune_r32_v1:var"])
    assert [(b.name, m) for b, m in plan] == [("frozen_a_prune_r32_v1", "var"), ("frozen_a_rand_r32_v1", "white")]
    with pytest.raises(ValueError, match="match no"):
        plan_outputs(banks, ["var"], ["frozen_a_rand_r32_v1:white"])
    with pytest.raises(ValueError, match="match no"):
        plan_outputs(banks, ["var"], ["frozen_a_prnue_r32_v1:var"])


@pytest.mark.parametrize("mode", ["var", "white", "energy"])
def test_rescale_bank_is_idempotent(mode):
    col = _FakeCollector(1, {g: 16 for g in GROUP_SITES})
    once, _ = rescale_bank(_fake_bank(), col, mode=mode)
    twice, _ = rescale_bank(once, col, mode=mode)
    assert all(torch.allclose(once[k], twice[k], rtol=1e-5, atol=1e-7) for k in once)


def test_rescale_white_handles_ill_conditioned_rows():
    bank = _fake_bank()
    a = bank["layers.0.q_proj"]
    a[1] = a[0] + 1e-3 * a[1]  # rows 0 and 1 nearly parallel: cond(A) ~ 1e3-1e4
    assert torch.linalg.cond(a.double()) > 1e3
    col = _FakeCollector(1, {g: 16 for g in GROUP_SITES})
    out, stats = rescale_bank(bank, col, mode="white")
    w = out["layers.0.q_proj"].double()
    gram = w @ col.covariance(0, "q") @ w.T
    target = float(col.covariance(0, "q").diagonal().sum()) / (3 * 16)
    assert torch.allclose(gram, target * torch.eye(RANK, dtype=gram.dtype), rtol=1e-3, atol=1e-3 * target)


def test_install_freezes_a_and_only_b_trains(tiny_sr, frozen_a_dir):
    ids = torch.randint(0, 300, (2, 12), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        base_logits = tiny_sr(input_ids=ids).logits.clone()
    model = _peft(tiny_sr)
    counts = install_frozen_a(model, frozen_a_dir)
    n_layers = len(tiny_sr.model.layers)
    assert counts == {"installed": n_layers * len(SITES), "frozen": n_layers * len(SITES)}

    t = load_file(str(frozen_a_dir / A_FILE))
    a_before, b_names = {}, []
    for name, p in model.named_parameters():
        if ".lora_A." in name:
            assert not p.requires_grad
            assert torch.equal(p.detach(), t[site_key(name.split(".lora_A.")[0])].to(p.dtype))
            a_before[name] = p.detach().clone()
        elif p.requires_grad:
            assert ".lora_B." in name, f"unexpected trainable param {name}"
            b_names.append(name)
    assert len(b_names) == n_layers * len(SITES)

    # B=0 at init: the adapter starts exactly at the base model.
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, base_logits, atol=1e-5)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    for _ in range(2):
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    params = dict(model.named_parameters())
    for name, before in a_before.items():
        assert torch.equal(params[name].detach(), before), f"frozen A moved: {name}"
    assert any(params[n].detach().abs().sum() > 0 for n in b_names), "B did not train"

    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            assert torch.allclose(model(input_ids=ids).logits, base_logits, atol=1e-5)
        assert not torch.allclose(model(input_ids=ids).logits, base_logits, atol=1e-5)


def test_install_rejects_missing_or_misshaped_a(tiny_sr, frozen_a_dir, tmp_path):
    t = load_file(str(frozen_a_dir / A_FILE))
    missing = tmp_path / "missing"
    missing.mkdir()
    save_file({k: v for k, v in t.items() if "cross_stream" not in k}, str(missing / A_FILE))
    with pytest.raises(KeyError):
        install_frozen_a(_peft(tiny_sr), missing)

    wrong = tmp_path / "wrong"
    wrong.mkdir()
    save_file({k: v[:, :-1].contiguous() for k, v in t.items()}, str(wrong / A_FILE))
    with pytest.raises(ValueError):
        install_frozen_a(_peft(tiny_sr), wrong)


def test_random_b_is_seeded_stock_draw_and_a_stays_frozen(tiny_sr, frozen_a_dir):
    import copy
    import math

    ids = torch.randint(0, 300, (2, 12), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        base_logits = tiny_sr(input_ids=ids).logits.clone()
    m1, m2 = _peft(copy.deepcopy(tiny_sr)), _peft(copy.deepcopy(tiny_sr))
    for m in (m1, m2):
        install_frozen_a(m, frozen_a_dir)
        assert randomize_lora_b(m, seed=7) == len(tiny_sr.model.layers) * len(SITES)
    b1 = {n: p for n, p in m1.named_parameters() if ".lora_B." in n}
    b2 = dict(m2.named_parameters())
    for name, p in b1.items():
        assert torch.equal(p, b2[name]), "same seed must give the same B"
        bound = 1 / math.sqrt(p.shape[1])
        assert p.abs().max() <= bound and p.abs().max() > 0.5 * bound
        assert p.requires_grad
    assert all(not p.requires_grad for n, p in m1.named_parameters() if ".lora_A." in n)
    with torch.no_grad():
        assert not torch.allclose(m1(input_ids=ids).logits, base_logits, atol=1e-5)
        with m1.disable_adapter():
            assert torch.allclose(m1(input_ids=ids).logits, base_logits, atol=1e-5)


def test_entry_point_patches_the_factory(tiny_sr, frozen_a_dir, monkeypatch):
    from shadow_residual.training import factory, train_frozen_a

    monkeypatch.setattr(factory, "get_shadow_residual_peft_model", lambda *a, **k: _peft(tiny_sr))
    train_frozen_a._patch_factory(str(frozen_a_dir))
    model = factory.get_shadow_residual_peft_model("ignored", None)
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".lora_A." in n)


def _labelled(n=3, seq=16, seed=7):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(0, 300, (1, seq), generator=g)
        labels = ids.clone()
        labels[:, : seq // 2] = -100
        out.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    return out


def test_grad_collector_matches_autograd_on_lora_b(tiny_sr):
    from peft.tuners.lora.layer import LoraLayer

    from shadow_residual.training.compute_grad_a import GradCollector, collect_task

    model = _peft(tiny_sr)
    model.train()
    col = GradCollector(model)
    batches = _labelled(n=1)
    out = model(**batches[0])
    out.loss.backward()
    col.end_batch()
    checked = 0
    for name, module in model.named_modules():
        key = site_key(name)
        if key is None or not isinstance(module, LoraLayer):
            continue
        a = module.lora_A["default"].weight.float()
        s = module.scaling["default"]
        expected = s * col.G[key] @ a.T
        assert torch.allclose(module.lora_B["default"].weight.grad.float(), expected, rtol=1e-4, atol=1e-6), key
        checked += 1
    assert checked == len(tiny_sr.model.layers) * len(SITES)
    assert set(col.G) == set(col.F)
    model.zero_grad(set_to_none=True)
    col.reset()
    n_tok = collect_task(model, col, _labelled(n=3))
    assert n_tok == 3 * 8 and len(col.G) == checked
    col.remove()


def test_grad_and_fwsvd_banks(tiny_sr, frozen_a_dir):
    from shadow_residual.training.compute_grad_a import GradCollector, build_banks, collect_task

    model = _peft(tiny_sr)
    model.train()
    col = GradCollector(model)
    collect_task(model, col, _labelled(n=4))
    cross = load_file(str(frozen_a_dir / A_FILE))
    banks = build_banks(model, col.G, col.F, RANK, cross)
    n_layers = len(tiny_sr.model.layers)
    for kind in ("grad", "fwsvd"):
        t, stats = banks[kind]
        assert set(t) == {f"layers.{i}.{s}" for i in range(n_layers) for s in SITES}
        for i in range(n_layers):
            assert torch.equal(t[f"layers.{i}.gate_proj"], t[f"layers.{i}.up_proj"])
            q = t[f"layers.{i}.q_proj"].double()
            assert torch.allclose(q @ q.T, torch.eye(RANK, dtype=q.dtype) / 3, atol=1e-5)
        assert all(0 < next(iter(v.values())) <= 1 + 1e-6 for v in stats.values())
    g_t, f_t = banks["grad"][0], banks["fwsvd"][0]
    assert not torch.equal(g_t["layers.0.cross_stream"], cross["layers.0.cross_stream"])
    assert torch.equal(f_t["layers.0.cross_stream"], cross["layers.0.cross_stream"].float())
    # grad A spans G's top right singular directions: G's energy outside it is the discarded tail
    G = col.G["layers.0.q_proj"].double()
    a = g_t["layers.0.q_proj"].double() * math.sqrt(3)
    kept = float((G @ a.T).pow(2).sum() / G.pow(2).sum())
    assert abs(kept - banks["grad"][1]["layers.0.q"]["grad_energy_top_r"]) < 1e-4
    col.remove()


@pytest.mark.parametrize("manifest_extra", [{"feature_scale": "kaiming"}, {}])
def test_record_provenance_accepts_any_bank_manifest(tiny_sr, frozen_a_dir, tmp_path, manifest_extra):
    from shadow_residual.training import train_frozen_a

    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / A_FILE).write_bytes((frozen_a_dir / A_FILE).read_bytes())
    (bank / MANIFEST_FILE).write_text(json.dumps({"rank": RANK, "recipe": {"mode": "white"}, **manifest_extra}))
    out = tmp_path / "ckpt"
    _peft(tiny_sr).save_pretrained(str(out))
    train_frozen_a._record_provenance(out, str(bank), "zeros", 0)
    rec = json.loads((out / "adapter_config.json").read_text())["frozen_lora_a"]
    assert rec["rank"] == RANK and rec["recipe"] == {"mode": "white"} and rec["lora_b_init"] == "zeros"
    assert rec["feature_scale"] == manifest_extra.get("feature_scale")


def test_saved_adapter_keeps_frozen_a(tiny_sr, frozen_a_dir, tmp_path):
    model = _peft(tiny_sr)
    install_frozen_a(model, frozen_a_dir)
    out = tmp_path / "ckpt"
    model.save_pretrained(str(out))
    assert saved_a_mismatches(out, frozen_a_dir) == []

    saved = load_file(str(out / "adapter_model.safetensors"))
    key = next(k for k in saved if ".lora_A." in k)
    saved[key] = saved[key] + 1
    save_file(saved, str(out / "adapter_model.safetensors"))
    assert saved_a_mismatches(out, frozen_a_dir) == [key]
