# SPDX-License-Identifier: Apache-2.0
"""Compress the A matrices of N trained SR adapters, training-free (compress_a.py).

    python -m shadow_residual.training.compress_adapters --base-model <path> \
        --adapters ans=<dir> guard=<dir> ... --calib-data a.jsonl b.jsonl ... \
        --variants dw-w:32 astack:32 q:int4 ... --out-root <dir>

Variants (``method:arg``):

=============  ==========================================================================
``astack:R``   one shared A for all adapters: SVD of the stacked, normalised A_i (B ignored)
``dw:R``       one shared A: top-R right singular space of the stacked ΔW_i (Frobenius)
``dw-w:R``     as ``dw`` but activation-whitened (output error on calibration inputs)
``dw-w-gu:R``  as ``dw-w`` and gate_proj + up_proj also share that A (same input)
``q:MODE``     per-adapter A quantised (int8/int4/int3/int2/sign), B unchanged
``qr:MODE``    per-adapter A quantised, B refit by whitened least squares
``tw:R``       per-adapter whitened rank-R truncation of ΔW_i (no sharing)
``dwq:R-MODE`` ``dw:R`` shared A, then quantised per row (MODE as ``q``), B refit whitened
=============  ==========================================================================

Every shared/refit B is the closed-form least-squares fit (``refit_b``). Writes
``<out-root>/<method>-<arg>/<name>/`` (stock PEFT adapter), ``…/bank/`` for shared-A
variants (a frozen-A bank for ``train_heal``), and ``<out-root>/stats.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from shadow_residual.training.compress_a import (
    NEW_SCALING,
    kaiming_scale_a,
    load_adapter,
    quantize_rows,
    refit_b,
    relative_error,
    shared_row_space,
    write_adapter,
)
from shadow_residual.training.frozen_a import A_FILE, MANIFEST_FILE, SITE_GROUP, _damped_cholesky

logger = logging.getLogger(__name__)

SAVE_DTYPE = torch.bfloat16
SHARED = {"astack", "dw", "dw-w", "dw-w-gu", "dwq"}
WHITENED = {"dw-w", "dw-w-gu", "qr", "tw", "dwq"}
METHODS = SHARED | {"q", "qr", "tw"}


def parse_variant(spec: str) -> tuple[str, str]:
    method, _, arg = spec.partition(":")
    if method not in METHODS or not arg:
        raise ValueError(f"bad variant {spec!r}; methods: {sorted(METHODS)}")
    if method == "dwq":
        rank, _, mode = arg.partition("-")
        int(rank)
        quantize_rows(torch.zeros(1, 4), mode)
    elif method in SHARED | {"tw"}:
        int(arg)
    return method, arg


def slug(spec: str) -> str:
    return spec.replace(":", "-")


def _split(key: str) -> tuple[int, str]:
    _, layer, site = key.split(".")
    return int(layer), site


@torch.no_grad()
def compress_all(adapters: dict[str, dict[str, dict]], variants: list[str], cov_fn=None,
                 device="cpu", damp: float = 1e-6):
    """Apply every variant to every adapter.

    ``adapters``: ``{name: {site_key: {"A","B","s","prefix"}}}`` (``load_adapter``).
    ``cov_fn(layer, group) -> cov``: input covariance, required by whitened variants.
    Returns ``(out, banks, stats)``: ``out[spec][name]`` = sites for ``write_adapter``,
    ``banks[spec]`` = shared A per site key, ``stats[spec]`` = error summary.
    """
    names = list(adapters)
    keys = sorted(adapters[names[0]])
    for n in names[1:]:
        if sorted(adapters[n]) != keys:
            raise ValueError(f"adapter {n} has different LoRA sites than {names[0]}")
    parsed = {spec: parse_variant(spec) for spec in variants}
    if cov_fn is None and any(m in WHITENED for m, _ in parsed.values()):
        raise ValueError("whitened variants need cov_fn")

    out = {spec: {n: {} for n in names} for spec in variants}
    banks = {spec: {} for spec, (m, _) in parsed.items() if m in SHARED}
    errs = {spec: {} for spec in variants}  # spec -> site -> list of (frob, white)
    retained = {spec: {} for spec in variants}
    covs: dict[tuple[int, str], torch.Tensor] = {}
    whiteners: dict[tuple[int, str], torch.Tensor] = {}

    def cov_of(layer, site):
        g = (layer, SITE_GROUP[site])
        if cov_fn is None:
            return None
        if g not in covs:
            covs.clear()
            whiteners.clear()
            covs[g] = cov_fn(*g).to(device=device, dtype=torch.float64)
            whiteners[g] = _damped_cholesky(covs[g], damp)
        return covs[g]

    def stored(a):
        # B is refit against the A that the bf16 checkpoint actually holds
        return a.to(SAVE_DTYPE).to(torch.float64)

    def put(spec, name, key, a_new, b_new):
        src = adapters[name][key]
        out[spec][name][key] = {"A": a_new.float().cpu(), "B": b_new.float().cpu(), "prefix": src["prefix"]}
        layer, site = _split(key)
        c = cov_of(layer, site)
        S = whiteners[(layer, SITE_GROUP[site])] if c is not None else None
        e_f = relative_error(src["B"], src["A"], src["s"], b_new, a_new, NEW_SCALING, None)
        e_w = relative_error(src["B"], src["A"], src["s"], b_new, a_new, NEW_SCALING, S) if S is not None else None
        errs[spec].setdefault(site, []).append((e_f, e_w))

    def factors(key):
        return [(adapters[n][key]["B"].to(device), adapters[n][key]["A"].to(device), adapters[n][key]["s"])
                for n in names]

    by_layer: dict[int, list[str]] = {}
    for key in keys:
        by_layer.setdefault(_split(key)[0], []).append(key)
    for layer in sorted(by_layer):
        lkeys = sorted(by_layer[layer], key=lambda k: SITE_GROUP[_split(k)[1]])
        for key in lkeys:
            _, site = _split(key)
            for spec, (method, arg) in parsed.items():
                if method in SHARED:
                    if method == "dw-w-gu" and site in ("gate_proj", "up_proj"):
                        if site == "up_proj":
                            continue  # done together with gate_proj
                        pair = [key, key.replace("gate_proj", "up_proj")]
                        fs = factors(pair[0]) + factors(pair[1])
                    else:
                        pair, fs = [key], factors(key)
                    cov = cov_of(layer, site) if method in ("dw-w", "dw-w-gu") else None
                    rank = int(arg.partition("-")[0])
                    a_dir, ret = shared_row_space(fs, rank, cov=cov, damp=damp, use_b=(method != "astack"))
                    a_new = stored(kaiming_scale_a(a_dir, cov))
                    if method == "dwq":
                        a_new = stored(quantize_rows(a_new, arg.partition("-")[2])).to(device)
                        cov = cov_of(layer, site)
                    for k in pair:
                        banks[spec][k] = a_new.float().cpu()
                        retained[spec][k] = ret
                        for n in names:
                            d = adapters[n][k]
                            put(spec, n, k, a_new, refit_b(d["B"], d["A"], d["s"], a_new, cov))
                elif method == "q":
                    for n in names:
                        d = adapters[n][key]
                        put(spec, n, key, stored(quantize_rows(d["A"], arg)), d["B"].double() * d["s"] / NEW_SCALING)
                elif method == "qr":
                    cov = cov_of(layer, site)
                    for n in names:
                        d = adapters[n][key]
                        a_q = stored(quantize_rows(d["A"], arg)).to(device)
                        put(spec, n, key, a_q, refit_b(d["B"], d["A"], d["s"], a_q, cov))
                elif method == "tw":
                    cov = cov_of(layer, site)
                    for n in names:
                        d = adapters[n][key]
                        a_dir, ret = shared_row_space(factors(key)[names.index(n):names.index(n) + 1],
                                                      int(arg), cov=cov, damp=damp)
                        a_new = stored(kaiming_scale_a(a_dir, cov))
                        retained[spec][f"{key}/{n}"] = ret
                        put(spec, n, key, a_new, refit_b(d["B"], d["A"], d["s"], a_new, cov))
        logger.info("layer %d compressed for %d variants", layer, len(variants))

    stats = {}
    for spec in variants:
        per_site = {}
        for site, vals in errs[spec].items():
            per_site[site] = {
                "rel_err_frob": sum(v[0] for v in vals) / len(vals),
                "rel_err_white": (sum(v[1] for v in vals) / len(vals)) if vals[0][1] is not None else None,
            }
        rets = list(retained[spec].values())
        stats[spec] = {"per_site": per_site, "retained_mean": (sum(rets) / len(rets)) if rets else None}
    return out, banks, stats


def _rank_of(sites: dict[str, dict]) -> int:
    ranks = {d["A"].shape[0] for d in sites.values()}
    if len(ranks) != 1:
        raise ValueError(f"mixed ranks {ranks}")
    return ranks.pop()


def write_outputs(out_root: Path, templates: dict[str, Path], out, banks, stats) -> None:
    from safetensors.torch import save_file

    for spec, per_name in out.items():
        vdir = out_root / slug(spec)
        for name, sites in per_name.items():
            write_adapter(templates[name], vdir / name, sites, _rank_of(sites),
                          {"variant": spec, "source": str(templates[name]), "stats": stats[spec]},
                          dtype=SAVE_DTYPE)
        if spec in banks:
            (vdir / "bank").mkdir(parents=True, exist_ok=True)
            save_file(banks[spec], str(vdir / "bank" / A_FILE))
            rank = next(iter(banks[spec].values())).shape[0]
            (vdir / "bank" / MANIFEST_FILE).write_text(json.dumps({
                "rank": rank, "recipe": {"all sites": f"compress_adapters {spec} of {sorted(templates)}"},
                "feature_scale": "kaiming", "sources": {k: str(v) for k, v in templates.items()},
            }, indent=2))
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapters", required=True, nargs="+", help="name=dir")
    p.add_argument("--variants", required=True, nargs="+")
    p.add_argument("--calib-data", nargs="*", default=[])
    p.add_argument("--rows-per-file", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--damp", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", required=True, type=Path)
    args = p.parse_args(argv)

    templates = {}
    for item in args.adapters:
        name, _, d = item.partition("=")
        templates[name] = Path(d)
    adapters = {n: load_adapter(d)[1] for n, d in templates.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cov_fn = None
    if args.calib_data:
        from transformers import AutoTokenizer

        from shadow_residual.shadow_residual.build import build_sr_base
        from shadow_residual.training.compute_frozen_a import _batches, _calibration_rows, collect

        tok = AutoTokenizer.from_pretrained(args.base_model)
        model = build_sr_base(args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        rows = _calibration_rows(args.calib_data, args.rows_per_file, args.seed)
        logger.info("calibrating on %d rows", len(rows))
        collector = collect(model, _batches(tok, rows, args.max_tokens, args.batch_size, device))
        collector.layers = None
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cov_fn = collector.covariance
        logger.info("calibration tokens per group: %d", next(iter(collector.count.values())))

    out, banks, stats = compress_all(adapters, args.variants, cov_fn, device=device, damp=args.damp)
    write_outputs(args.out_root, templates, out, banks, stats)
    for spec in args.variants:
        s = stats[spec]
        print("COMPRESS_SUMMARY " + json.dumps({"variant": spec, "retained_mean": s["retained_mean"],
                                                "per_site": s["per_site"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
