# SPDX-License-Identifier: Apache-2.0
"""Compress the LoRA ``A`` matrices of trained SR adapters (docs/autopilot/a-compression).

Each LoRA site holds ``ΔW = s·B·A`` with ``A`` ``[r, in]``, ``B`` ``[out, r]`` and
``s = α/r``. This module rewrites ``A`` in a cheaper form and refits ``B`` so that
``ΔW`` changes as little as possible:

* ``shared_row_space``: one ``A`` of rank R for N adapters (Compress-then-Serve style
  joint basis). Optionally activation-whitened, so the loss is the output error
  ``‖(ΔW − ΔW') Xᵀ‖`` on calibration inputs and not the weight error.
* ``quantize_rows``: per-row int-k / ternary / sign ``A``.
* ``structured_bank``: data-free random ``A`` banks (uniform, sign, sparse, gather,
  cross-layer shared) for B-only training.

``refit_b`` is the closed-form least-squares ``B`` for any new ``A``. All maths runs in
fp64. Results are stock PEFT LoRA checkpoints.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import torch

from shadow_residual.training.frozen_a import _damped_cholesky, site_key

ADAPTER_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"
KAIMING_ROW_NORM = 1 / math.sqrt(3)  # E‖row‖ of PEFT's kaiming_uniform(a=√5) lora_A
_SITE = re.compile(r"\.(q_proj|o_proj|gate_proj|up_proj|down_proj|cross_stream)$")


# ----------------------------------------------------------------------------- I/O


def site_scaling(cfg: dict, site: str, r: int) -> float:
    alpha = cfg.get("alpha_pattern", {}).get(site, cfg["lora_alpha"])
    return alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)


def load_adapter(path: str | Path) -> tuple[dict, dict[str, dict]]:
    """``(adapter_config, {site_key: {"A", "B", "s", "prefix"}})`` with A/B in fp32."""
    from safetensors.torch import load_file

    path = Path(path)
    cfg = json.loads((path / CONFIG_FILE).read_text())
    saved = load_file(str(path / ADAPTER_FILE))
    sites: dict[str, dict] = {}
    for k, v in saved.items():
        for part in ("lora_A", "lora_B"):
            tag = f".{part}."
            if tag in k:
                prefix = k.split(tag)[0]
                key = site_key(prefix)
                if key is None:
                    raise KeyError(f"unrecognised LoRA tensor {k}")
                sites.setdefault(key, {"prefix": prefix})[part[-1]] = v.float()
    for key, d in sites.items():
        if "A" not in d or "B" not in d:
            raise KeyError(f"{key}: missing lora_A or lora_B")
        d["s"] = site_scaling(cfg, _SITE.search(d["prefix"]).group(1), d["A"].shape[0])
    return cfg, sites


def write_adapter(template: str | Path, out: str | Path, sites: dict[str, dict], rank: int,
                  provenance: dict, dtype=torch.bfloat16) -> None:
    """Write a stock PEFT adapter from ``{site_key: {"A","B","prefix"}}`` at rank ``rank``.

    Every site gets ``r = rank`` and ``lora_alpha = 2·rank`` (s = 2, the vanilla SR
    scaling). ``B`` must already be fitted for s = 2. Non-weight files (tokenizer,
    chat template) are copied from ``template``.
    """
    from safetensors.torch import save_file

    template, out = Path(template), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in template.iterdir():
        if f.is_file() and f.name not in (ADAPTER_FILE, CONFIG_FILE, "training_args.bin"):
            shutil.copy2(f, out / f.name)
    cfg = json.loads((template / CONFIG_FILE).read_text())
    targets = cfg["target_modules"]
    if not isinstance(targets, list):
        raise ValueError(f"target_modules must be a list of site names, got {targets!r}")
    cfg.update(r=rank, lora_alpha=2 * rank, alpha_pattern={}, rank_pattern={t: rank for t in targets},
               use_rslora=False)
    cfg["a_compression"] = provenance
    tensors = {}
    for d in sites.values():
        if d["A"].shape[0] != rank or d["B"].shape[1] != rank:
            raise ValueError(f"{d['prefix']}: A {tuple(d['A'].shape)} / B {tuple(d['B'].shape)} != rank {rank}")
        tensors[f"{d['prefix']}.lora_A.weight"] = d["A"].to(dtype).contiguous()
        tensors[f"{d['prefix']}.lora_B.weight"] = d["B"].to(dtype).contiguous()
    save_file(tensors, str(out / ADAPTER_FILE), metadata={"format": "pt"})
    (out / CONFIG_FILE).write_text(json.dumps(cfg, indent=2))


# ----------------------------------------------------------------------------- maths

NEW_SCALING = 2.0


def refit_b(b: torch.Tensor, a_old: torch.Tensor, s_old: float, a_new: torch.Tensor,
            cov: torch.Tensor | None = None, s_new: float = NEW_SCALING) -> torch.Tensor:
    """Least-squares ``B'`` with ``s_new·B'·A_new ≈ s_old·B·A_old`` under input covariance ``cov``.

    Minimises ``‖(s_old B A_old − s_new B' A_new) S‖_F`` with ``S Sᵀ = cov`` (identity
    when ``cov`` is None): ``B' = s_old B A_old C A_newᵀ (A_new C A_newᵀ)⁻¹ / s_new``.
    """
    dev = a_new.device
    b, a_old, a_new = (t.to(device=dev, dtype=torch.float64) for t in (b, a_old, a_new))
    ca = a_new.T if cov is None else cov.to(device=dev, dtype=torch.float64) @ a_new.T  # [in, R]
    gram = a_new @ ca
    rhs = (s_old * b) @ (a_old @ ca)  # [out, R]
    return torch.linalg.solve(gram, rhs.T).T / s_new


def _whitener(cov: torch.Tensor | None, dim: int, device, damp: float):
    if cov is None:
        return None
    return _damped_cholesky(cov.to(device=device, dtype=torch.float64), damp)


def shared_row_space(factors: list[tuple[torch.Tensor, torch.Tensor, float]], rank: int,
                     cov: torch.Tensor | None = None, normalize: bool = True, damp: float = 1e-6,
                     use_b: bool = True) -> tuple[torch.Tensor, float]:
    """The rank-R input map shared by N low-rank updates ``ΔW_i = s_i B_i A_i``.

    Returns ``(A [R, in], retained)``: the top-R right singular directions of the
    stacked ``[w_1 ΔW_1 S; …; w_N ΔW_N S]`` mapped back through ``S⁻¹`` (``S = I``
    without ``cov``), and the fraction of the stacked energy those R directions keep.
    ``normalize`` sets ``w_i = 1/‖ΔW_i S‖_F`` so every adapter weighs the same.
    ``use_b=False`` ignores B (SVD of the stacked, normalised ``A_i S`` only).

    Exact and cheap: every ``ΔW_i S`` lies in the ≤ Σr-dim row space of ``[A_i S]``,
    so the SVD runs on a ``[Σout, Σr]`` matrix after one QR.
    """
    dev = factors[0][1].device
    in_dim = factors[0][1].shape[1]
    S = _whitener(cov, in_dim, dev, damp)
    a_s = [a.to(torch.float64) if S is None else a.to(torch.float64) @ S for _, a, _ in factors]
    total_r = sum(x.shape[0] for x in a_s)
    if rank > min(total_r, in_dim):
        raise ValueError(f"rank {rank} exceeds the joint row-space dimension {min(total_r, in_dim)}")
    q, r_mat = torch.linalg.qr(torch.cat(a_s, 0).T)  # K^T = Q R, K = [A_i S]
    rt = r_mat.T  # K = Rᵀ Qᵀ
    blocks, off = [], 0
    for (b, _, s), a in zip(factors, a_s):
        n = a.shape[0]
        coef = rt[off:off + n]  # A_i S = coef Qᵀ
        off += n
        m = (s * b.to(torch.float64)) @ coef if use_b else coef
        if normalize:
            m = m / m.norm().clamp_min(1e-30)
        blocks.append(m)
    _, sig, wh = torch.linalg.svd(torch.cat(blocks, 0), full_matrices=False)
    v = q @ wh[:rank].T  # [in, R], orthonormal columns
    retained = float((sig[:rank] ** 2).sum() / (sig ** 2).sum().clamp_min(1e-300))
    if S is None:
        return v.T.contiguous(), retained
    return torch.linalg.solve_triangular(S.T, v, upper=True).T.contiguous(), retained


def kaiming_scale_a(a: torch.Tensor, cov: torch.Tensor | None) -> torch.Tensor:
    """Rescale an orthonormal (or ``cov``-whitened) A to stock-LoRA feature energy.

    Without ``cov`` A has orthonormal rows; they get the kaiming row norm 1/√3. With
    ``cov`` (``A C Aᵀ = I``) each feature gets std ``√(tr C / (3·in))``, the std a
    kaiming lora_A row gives on those inputs (as in frozen_a / compute_frozen_a).
    """
    if cov is None:
        return a * KAIMING_ROW_NORM
    return a * math.sqrt(float(cov.diagonal().sum()) / (3 * cov.shape[0]))


def quantize_rows(a: torch.Tensor, mode: str) -> torch.Tensor:
    """Fake-quantised A, per-row symmetric scale.

    ``int8/int4/int3``: round-to-nearest on the absmax grid. ``int2``: ternary
    {−α, 0, α} with TWN's threshold Δ = 0.7·mean|a| and α = mean|a| over |a| > Δ.
    ``sign``: α·sign(a) with α = mean|a| (the L2-optimal 1-bit scale).
    """
    a = a.to(torch.float64)
    if mode == "sign":
        alpha = a.abs().mean(dim=1, keepdim=True)
        return torch.where(a >= 0, alpha, -alpha)
    if mode == "int2":
        delta = 0.7 * a.abs().mean(dim=1, keepdim=True)
        keep = a.abs() > delta
        alpha = (a.abs() * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True).clamp_min(1)
        return torch.sign(a) * keep * alpha
    m = re.fullmatch(r"int(\d+)", mode)
    if not m or int(m.group(1)) < 3:
        raise ValueError(f"unknown quantisation mode {mode!r}")
    qmax = 2 ** (int(m.group(1)) - 1) - 1
    scale = a.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / qmax
    return torch.round(a / scale).clamp(-qmax, qmax) * scale


def relative_error(b, a, s, b_new, a_new, s_new, S: torch.Tensor | None) -> float:
    """``‖(sBA − s'B'A') S‖² / ‖sBA S‖²`` without forming an ``[in, in]`` product per site."""
    dev = S.device if S is not None else a_new.device
    f = lambda t: t.to(device=dev, dtype=torch.float64)  # noqa: E731
    a, a_new, b, b_new = f(a), f(a_new), f(b), f(b_new)
    if S is not None:
        a, a_new = a @ S, a_new @ S
    ref = (s * b) @ a
    err = ref - (s_new * b_new) @ a_new
    return float(err.pow(2).sum() / ref.pow(2).sum().clamp_min(1e-300))


# ----------------------------------------------------------------------------- banks

BANK_KINDS = ("uniform", "sign", "sparse", "gather")


def _draw(kind: str, rank: int, in_dim: int, gen: torch.Generator) -> torch.Tensor:
    """One data-free A ``[rank, in]`` with the kaiming per-entry variance 1/(3·in) (gather: row norm 1/√3)."""
    if kind == "uniform":
        return (torch.rand(rank, in_dim, generator=gen) * 2 - 1) / math.sqrt(in_dim)
    sign = torch.where(torch.rand(rank, in_dim, generator=gen) < 0.5, -1.0, 1.0)
    if kind == "sign":
        return sign / math.sqrt(3 * in_dim)
    if kind == "sparse":
        p = 1 / math.sqrt(in_dim)  # very sparse random projection (Li, Hastie, Church 2006)
        mask = torch.rand(rank, in_dim, generator=gen) < p
        return sign * mask / math.sqrt(3 * in_dim * p)
    if kind == "gather":
        if rank > in_dim:
            raise ValueError("gather needs rank <= in_features")
        cols = torch.randperm(in_dim, generator=gen)[:rank]
        a = torch.zeros(rank, in_dim)
        a[torch.arange(rank), cols] = KAIMING_ROW_NORM
        return a
    raise ValueError(f"unknown bank kind {kind!r}")


def structured_bank(shapes: dict[str, tuple[int, int]], kind: str, seed: int = 0,
                    cross_layer: bool = False, whole_model: bool = False) -> dict[str, torch.Tensor]:
    """A data-free frozen-A bank keyed ``layers.{l}.{site}``.

    gate_proj and up_proj share one draw (they read the same tensor). With
    ``cross_layer`` every layer reuses layer 0's draw for each site. With
    ``whole_model`` every site of every layer uses one ``[rank, max_in]`` draw: a site
    with ``in`` inputs takes its first ``in`` columns times √(max_in/in), which keeps the
    per-entry variance 1/(3·in) (the scalar folds into B at serving time).
    """
    gen = torch.Generator().manual_seed(seed)
    if whole_model:
        if kind == "gather":
            raise ValueError("whole_model needs a dense draw; column slices of a gather A drop rows")
        ranks = {r for r, _ in shapes.values()}
        if len(ranks) != 1:
            raise ValueError(f"whole_model needs one rank, got {sorted(ranks)}")
        max_in = max(i for _, i in shapes.values())
        base = _draw(kind, ranks.pop(), max_in, gen)
        return {k: (base[:, :i] * math.sqrt(max_in / i)).contiguous() for k, (_, i) in shapes.items()}
    out: dict[str, torch.Tensor] = {}
    by_site: dict[str, torch.Tensor] = {}

    def layer_of(k):
        return int(k.split(".")[1])

    for key in sorted(shapes, key=lambda k: (layer_of(k), k)):
        site = key.split(".")[2]
        if site == "up_proj":
            continue
        if cross_layer and site in by_site:
            a = by_site[site]
        else:
            a = _draw(kind, *shapes[key], gen)
            by_site.setdefault(site, a)
        out[key] = a.clone()
        up = key.replace("gate_proj", "up_proj")
        if site == "gate_proj" and up in shapes:
            out[up] = a.clone()
    missing = set(shapes) - set(out)
    if missing:
        raise KeyError(f"no draw for {sorted(missing)[:3]} (up_proj without gate_proj?)")
    return out


def main(argv=None) -> int:
    """Write a structured frozen-A bank (for ``train_frozen_a``) with the site shapes of an existing bank."""
    import argparse

    from safetensors.torch import load_file, save_file

    from shadow_residual.training.frozen_a import A_FILE, MANIFEST_FILE

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--shapes-from", required=True, type=Path, help="bank dir whose keys / in_features to copy")
    p.add_argument("--kind", required=True, choices=BANK_KINDS)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--cross-layer", action="store_true")
    p.add_argument("--whole-model", action="store_true", help="one draw for every site of every layer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    ref = load_file(str(args.shapes_from / A_FILE))
    bank = structured_bank({k: (args.rank, v.shape[1]) for k, v in ref.items()}, args.kind, args.seed,
                           args.cross_layer, args.whole_model)
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(bank, str(args.out / A_FILE))
    nnz = sum(int((v != 0).sum()) for v in bank.values())
    distinct = len({v.numpy().tobytes() for v in bank.values()})
    (args.out / MANIFEST_FILE).write_text(json.dumps({
        "rank": args.rank, "recipe": {"all sites": f"structured {args.kind} bank, cross_layer={args.cross_layer}, "
                                                   f"whole_model={args.whole_model}",
                                      "gate+up": "shared draw"},
        "feature_scale": "kaiming (stock init energy)", "seed": args.seed, "shapes_from": str(args.shapes_from),
        "nonzeros": nnz, "values": sum(v.numel() for v in bank.values()), "tensors": distinct,
    }, indent=2))
    print(f"BANK {args.kind} r={args.rank} cross_layer={args.cross_layer}: {len(bank)} tensors, {nnz} nonzeros",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
