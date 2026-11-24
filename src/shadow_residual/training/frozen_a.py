# SPDX-License-Identifier: Apache-2.0
"""Frozen, compression-derived LoRA ``A`` matrices for SR (docs/sr_frozen_a_design.html).

Every LoRA ``A`` is precomputed from the frozen base model and never trained; only
``B`` learns. Sites reading the same input tensor share one ``A``:

* q_proj, o_proj, down_proj: SVD-LLM truncation-aware whitened SVD of the base
  weight, ``A = V_rᵀ S⁻¹`` with ``S Sᵀ = Cov(x)`` and ``W S = U Σ Vᵀ``.
* gate_proj + up_proj: one shared ``A`` from the whitened SVD of ``[W_gate; W_up]``
  (Basis Sharing), since both read the same normed hidden state.
* cross_stream (zero base weight, reads ``h_base``): unit-variance PCA of its input,
  ``A = Λ_r^{-1/2} P_rᵀ`` (ESPACE / EVA).

Both recipes give ``A Cov Aᵀ = I``: the r frozen features are decorrelated and
unit-variance on the calibration data.

The adapter stays stock PEFT: ``A`` is copied into ``lora_A`` and frozen, and is
saved with the checkpoint like any LoRA ``A``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch

A_FILE = "frozen_a.safetensors"
MANIFEST_FILE = "manifest.json"

# Covariance groups: one per distinct input tensor of a decoder layer.
GROUP_SITES = {
    "q": ("q_proj",),
    "o": ("o_proj",),
    "mlp_in": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
    "cross": ("cross_stream",),
}
SITE_GROUP = {site: g for g, sites in GROUP_SITES.items() for site in sites}

_LAYER_SITE = re.compile(r"layers\.(\d+)\.(?:self_attn\.|mlp\.)?(q_proj|o_proj|gate_proj|up_proj|down_proj|cross_stream)$")


def _damped_cholesky(cov: torch.Tensor, damp: float) -> torch.Tensor:
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    eps = damp * cov.diagonal().mean().clamp_min(1e-30)
    for _ in range(8):
        L, info = torch.linalg.cholesky_ex(cov + eps * eye)
        if int(info) == 0:
            return L
        eps = eps * 10
    raise RuntimeError("covariance is not positive definite even after damping")


def whitened_svd_a(weight: torch.Tensor, cov: torch.Tensor, rank: int, damp: float = 1e-6):
    """SVD-LLM down-projection: the rank-r input map minimizing ‖(W − W') Xᵀ‖_F.

    ``weight`` is ``[out, in]`` (stack several weights along ``out`` to share one A);
    ``cov`` is ``XᵀX / N`` ``[in, in]``. Returns ``(A [rank, in], retained_energy)``
    where ``retained_energy`` is the fraction of the whitened output energy the
    rank-r subspace keeps.
    """
    w = weight.to(torch.float64)
    c = cov.to(device=w.device, dtype=torch.float64)
    S = _damped_cholesky(c, damp)
    _, sig, Vh = torch.linalg.svd(w @ S, full_matrices=False)
    # A = V_rᵀ S⁻¹  ⇔  Aᵀ = S⁻ᵀ V_r : one triangular solve with Sᵀ (upper).
    a = torch.linalg.solve_triangular(S.T, Vh[:rank].T, upper=True).T
    energy = float((sig[:rank] ** 2).sum() / (sig ** 2).sum())
    return a.contiguous(), energy


def pca_a(cov: torch.Tensor, rank: int, damp: float = 1e-6):
    """Unit-variance top-r principal directions of the input (ESPACE / EVA)."""
    c = cov.to(torch.float64)
    evals, evecs = torch.linalg.eigh(c)
    evals, evecs = evals.flip(0), evecs.flip(1)
    floor = damp * evals.clamp_min(0).mean().clamp_min(1e-30)
    top = evals[:rank].clamp_min(floor)
    a = (evecs[:, :rank] / top.sqrt()).T
    energy = float(evals[:rank].clamp_min(0).sum() / evals.clamp_min(0).sum())
    return a.contiguous(), energy


def weight_svd_a(weight: torch.Tensor, rank: int, which: str = "minor"):
    """Data-free A from the base weight alone: r right singular vectors of ``W``.

    ``which="minor"`` takes the r smallest singular values of the thin SVD (MiLoRA),
    ``"principal"`` the r largest (PiSSA). Rows are orthonormal, scaled by 1/√3 so each
    row has the expected norm of PEFT's stock kaiming_uniform(a=√5) lora_A row.
    Returns ``(A [rank, in], fraction of ‖W‖_F² in the chosen directions)``.
    """
    w = weight.to(torch.float64)
    _, sig, Vh = torch.linalg.svd(w, full_matrices=False)
    if which == "minor":
        idx = slice(sig.numel() - rank, sig.numel())
    elif which == "principal":
        idx = slice(0, rank)
    else:
        raise ValueError(f"which must be 'minor' or 'principal', got {which!r}")
    a = Vh[idx] / math.sqrt(3)
    energy = float((sig[idx] ** 2).sum() / (sig ** 2).sum())
    return a.contiguous(), energy


COMPRESS_METHODS = ("prune", "fold", "sketch")


def _kmeans_centroids(x: torch.Tensor, k: int, gen: torch.Generator, iters: int = 100) -> torch.Tensor:
    """Lloyd k-means with k-means++ seeding; stops when the assignment is stable.

    An empty cluster re-seeds at the row farthest from all current centers, one cluster
    at a time, so two empty clusters never get the same row.
    """
    n = x.shape[0]
    centers = x[torch.randint(n, (1,), generator=gen)]
    d2 = torch.cdist(x, centers).squeeze(1).pow(2)
    for _ in range(1, k):
        if not bool((d2 > 0).any()):
            raise ValueError(f"k-means needs at least {k} distinct rows")
        c = x[torch.multinomial(d2, 1, generator=gen)]
        centers = torch.cat([centers, c])
        d2 = torch.minimum(d2, torch.cdist(x, c).squeeze(1).pow(2))
    assign = None
    for _ in range(iters):
        new_assign = torch.cdist(x, centers).argmin(dim=1)
        if assign is not None and torch.equal(new_assign, assign):
            break
        assign = new_assign
        counts = torch.bincount(assign, minlength=k)
        sums = torch.zeros_like(centers).index_add_(0, assign, x)
        valid = counts > 0
        centers = torch.where(valid.unsqueeze(1), sums / counts.clamp_min(1).unsqueeze(1).to(x.dtype), centers)
        for j in (~valid).nonzero().flatten().tolist():
            centers[j] = x[torch.cdist(x, centers[valid]).min(dim=1).values.argmax()]
            valid[j] = True
    return centers


def row_space_stats(a: torch.Tensor, weight: torch.Tensor, tol: float = 1e-6) -> dict:
    """Numerical rank and condition number of A, and the fraction of ‖W‖_F² in its row space."""
    a64, w64 = a.to(torch.float64), weight.to(torch.float64)
    _, s, vh = torch.linalg.svd(a64, full_matrices=False)
    keep = s > tol * s[0]
    basis = vh[keep]
    return {
        "energy": float((w64 @ basis.T).pow(2).sum() / w64.pow(2).sum()),
        "rank": int(keep.sum()),
        "cond": float(s[0] / s[keep][-1]),
    }


def compressed_rows_a(weight: torch.Tensor, rank: int, method: str, seed: int = 0):
    """A built by compressing W's ``out`` rows (output neurons) into ``rank`` rows.

    * ``prune``: the ``rank`` rows with the largest L1 norm (structured neuron pruning).
    * ``fold``: k-means centroids of the rows (model folding).
    * ``sketch``: ``G·W`` with ``G ~ N(0, 1)`` of shape ``[rank, out]`` (random sketch).

    Every row is rescaled to norm 1/√3, the stock kaiming lora_A row norm. Returns
    ``(A [rank, in], row_space_stats(A, W))``.
    """
    w = weight.to(torch.float32)
    gen = torch.Generator().manual_seed(seed)
    if method == "prune":
        rows = w[w.abs().sum(dim=1).topk(rank).indices.sort().values]
    elif method == "fold":
        rows = _kmeans_centroids(w, rank, gen)
    elif method == "sketch":
        rows = torch.randn(rank, w.shape[0], generator=gen) @ w
    else:
        raise ValueError(f"method must be one of {COMPRESS_METHODS}, got {method!r}")
    norms = rows.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(rows).all()) or not bool((norms > 0).all()):
        raise ValueError(f"{method}: compressed rows contain a zero-norm or non-finite row")
    a = (rows / norms / math.sqrt(3)).to(torch.float64)
    return a.contiguous(), row_space_stats(a, weight)


def site_key(module_name: str) -> str | None:
    """``layers.{l}.{site}`` for a PEFT LoRA module name, or None if not an SR site."""
    m = _LAYER_SITE.search(module_name)
    return f"layers.{m.group(1)}.{m.group(2)}" if m else None


def load_frozen_a(path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(Path(path) / A_FILE))


def install_frozen_a(peft_model, path: str | Path, adapter_name: str = "default",
                     freeze: bool = True) -> dict:
    """Copy the precomputed A into every LoRA layer's ``lora_A`` and freeze it.

    Meta parameters (FSDP non-rank-0, filled later by the rank-0 broadcast) are
    only frozen. Every LoRA layer must have an A of matching shape in the file.
    ``freeze=False`` only initialises A (it stays trainable).
    Returns ``{"installed": n, "frozen": n}``.
    """
    from peft.tuners.lora.layer import LoraLayer

    tensors = load_frozen_a(path)
    installed = matched = 0
    for name, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer) or adapter_name not in module.lora_A:
            continue
        key = site_key(name)
        if key is None or key not in tensors:
            raise KeyError(f"no frozen A for LoRA module {name!r} (key {key!r}) in {path}")
        weight = module.lora_A[adapter_name].weight
        a = tensors[key]
        if tuple(weight.shape) != tuple(a.shape):
            raise ValueError(f"{name}: lora_A shape {tuple(weight.shape)} != frozen A {tuple(a.shape)}")
        if weight.device.type != "meta":
            with torch.no_grad():
                weight.copy_(a.to(device=weight.device, dtype=weight.dtype))
            installed += 1
        if freeze:
            weight.requires_grad_(False)
        matched += 1
    if matched == 0:
        raise ValueError("no LoRA layers found to install a frozen A into")
    return {"installed": installed, "frozen": matched if freeze else 0}


def randomize_lora_b(peft_model, seed: int = 0, adapter_name: str = "default") -> int:
    """Re-draw every ``lora_B`` like a stock ``nn.Linear`` (PEFT's ``init_lora_weights=False``).

    kaiming_uniform_(a=√5) on ``[out, r]`` is ``U(-1/√r, 1/√r)``. Drawn layer by layer in
    module order from one seeded generator, so every rank gets the same B. Meta tensors
    (FSDP non-rank-0) are skipped; the rank-0 broadcast fills them.
    """
    import math

    from peft.tuners.lora.layer import LoraLayer

    gen = torch.Generator().manual_seed(seed)
    n = 0
    for _, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer) or adapter_name not in module.lora_B:
            continue
        weight = module.lora_B[adapter_name].weight
        bound = 1 / math.sqrt(weight.shape[1])
        draw = (torch.rand(tuple(weight.shape), generator=gen) * 2 - 1) * bound
        if weight.device.type != "meta":
            with torch.no_grad():
                weight.copy_(draw.to(device=weight.device, dtype=weight.dtype))
        n += 1
    return n


def saved_a_mismatches(adapter_dir: str | Path, path: str | Path, atol: float = 0.0) -> list[str]:
    """Saved ``lora_A`` tensors that differ from the frozen A (after the save dtype cast)."""
    from safetensors.torch import load_file

    saved = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    tensors = load_frozen_a(path)
    bad, seen = [], 0
    for k, v in saved.items():
        if ".lora_A." not in k:
            continue
        key = site_key(k.split(".lora_A.")[0])
        ref = tensors.get(key)
        seen += 1
        if ref is None or tuple(ref.shape) != tuple(v.shape):
            bad.append(k)
            continue
        if not torch.allclose(v.float(), ref.to(v.dtype).float(), atol=atol, rtol=0):
            bad.append(k)
    if seen == 0:
        bad.append("<no lora_A tensors saved>")
    return bad


def read_manifest(path: str | Path) -> dict:
    return json.loads((Path(path) / MANIFEST_FILE).read_text())
