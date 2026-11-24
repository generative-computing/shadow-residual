# SPDX-License-Identifier: Apache-2.0
"""Multi-task ``train.py`` with one shared, trainable LoRA ``A`` and fixed per-task ``B``.

    accelerate launch ... -m shadow_residual.training.train_shared_a_fixed_b \
        --init-a-bank <bank dir> --fixed-b ans:<adapter dir> ... \
        --task ans:TRAIN:VAL ... <every other train.py argument>

The A-step of an alternating A/B optimisation. Every task's ``lora_B`` is copied from an
already-trained adapter and frozen; the shared ``A`` (tied across tasks by
``multitask.setup_shared_a``) starts from a bank in the ``frozen_a.safetensors`` layout
and is the only trainable tensor. ``--export-init-only`` writes the step-0 adapters
(bank A + source B) without training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import torch

from shadow_residual.training import multitask, train
from shadow_residual.training.compress_a import ADAPTER_FILE, CONFIG_FILE, load_adapter, write_adapter
from shadow_residual.training.frozen_a import A_FILE, install_frozen_a, load_frozen_a, site_key
from shadow_residual.training.train_heal import install_b

logger = logging.getLogger(__name__)


def parse_fixed_b(specs: list[str]) -> dict[str, str]:
    """``NAME:ADAPTER_DIR`` strings -> ``{name: dir}``."""
    out: dict[str, str] = {}
    for spec in specs:
        name, sep, path = spec.partition(":")
        if not sep or not name or not path:
            raise ValueError(f"--fixed-b expects NAME:ADAPTER_DIR, got {spec!r}")
        if name in out:
            raise ValueError(f"duplicate --fixed-b task {name!r}")
        out[name] = path
    return out


def install_shared_a_fixed_b(peft_model, task_names, bank: str | Path, fixed_b: dict[str, str]) -> dict:
    """Initialise the tied A from ``bank`` (trainable) and copy + freeze every task's B."""
    from peft.tuners.lora.layer import LoraLayer

    names = list(task_names)
    if set(names) != set(fixed_b):
        raise ValueError(f"--fixed-b tasks {sorted(fixed_b)} != --task tasks {sorted(names)}")
    counts = install_frozen_a(peft_model, bank, adapter_name=names[0], freeze=False)
    n_b = {name: install_b(peft_model, fixed_b[name], adapter_name=name) for name in names}
    for module in peft_model.modules():
        if isinstance(module, LoraLayer):
            for name in names:
                module.lora_B[name].weight.requires_grad_(False)
    return {"a_installed": counts["installed"], "b_installed": n_b}


def _patch_setup(bank: str, fixed_b: dict[str, str]) -> None:
    setup = multitask.setup_shared_a

    def setup_fixed_b(peft_model, task_names, lora_config, dtype=None, tie_a: bool = True):
        if not tie_a:
            raise ValueError("fixed-B training needs the tied (shared) A; drop --untie-a")
        router = setup(peft_model, task_names, lora_config, dtype=dtype, tie_a=True)
        logger.info("Shared A from %s, fixed B: %s", bank,
                    install_shared_a_fixed_b(peft_model, task_names, bank, fixed_b))
        return router

    multitask.setup_shared_a = setup_fixed_b


def _tensors(adapter_dir: str | Path, part: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    saved = load_file(str(Path(adapter_dir) / ADAPTER_FILE))
    tag = f".{part}."
    return {site_key(k.split(tag)[0]): v for k, v in saved.items() if tag in k}


def check_exports(out: str | Path, fixed_b: dict[str, str], bank: str | Path) -> dict:
    """Saved B == source B exactly, A identical across tasks, and A moved off the bank."""
    out = Path(out)
    names = list(fixed_b)
    a_ref = _tensors(out / names[0], "lora_A")
    for name in names:
        saved_b, src_b = _tensors(out / name, "lora_B"), _tensors(fixed_b[name], "lora_B")
        if set(saved_b) != set(src_b):
            raise RuntimeError(f"{name}: saved lora_B sites differ from {fixed_b[name]}")
        bad = [k for k, v in saved_b.items() if not torch.equal(v, src_b[k].to(v.dtype))]
        if bad:
            raise RuntimeError(f"{name}: {len(bad)} lora_B differ from {fixed_b[name]}, e.g. {bad[:3]}")
        a = _tensors(out / name, "lora_A")
        if set(a) != set(a_ref) or any(not torch.equal(a[k], a_ref[k]) for k in a_ref):
            raise RuntimeError(f"{name}: lora_A is not identical to {names[0]}'s")
    init = load_frozen_a(bank)
    unchanged = sum(torch.equal(v, init[k].to(v.dtype)) for k, v in a_ref.items())
    if unchanged == len(a_ref):
        raise RuntimeError("no lora_A moved from the init bank: A did not train")
    return {"a_tensors": len(a_ref), "a_unchanged": unchanged}


def _record_provenance(adapter_dir: Path, bank: str, source_b: str, trained: bool) -> None:
    cfg_path = adapter_dir / CONFIG_FILE
    cfg = json.loads(cfg_path.read_text())
    cfg["shared_a_fixed_b"] = {
        "init_a_bank": str(bank),
        "init_a_sha256": hashlib.sha256((Path(bank) / A_FILE).read_bytes()).hexdigest(),
        "fixed_b_from": str(source_b),
        "a_trained": trained,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))


def export_init(out: str | Path, bank: str | Path, fixed_b: dict[str, str]) -> None:
    """Write ``<out>/<task>/``: each source adapter with its A replaced by the bank's."""
    out = Path(out)
    init = load_frozen_a(bank)
    for name, src in fixed_b.items():
        cfg, sites = load_adapter(src)
        ranks = {d["A"].shape[0] for d in sites.values()}
        if len(ranks) != 1:
            raise ValueError(f"{src}: mixed LoRA ranks {ranks}")
        for key, d in sites.items():
            if tuple(init[key].shape) != tuple(d["A"].shape):
                raise ValueError(f"{key}: bank A {tuple(init[key].shape)} != adapter A {tuple(d['A'].shape)}")
            d["A"] = init[key].float()
        dtype = next(iter(_tensors(src, "lora_B").values())).dtype
        write_adapter(src, out / name, sites, ranks.pop(), dtype=dtype,
                      provenance={"method": "shared-a-fixed-b init", "bank": str(bank)})
        _record_provenance(out / name, str(bank), src, trained=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--init-a-bank", required=True)
    p.add_argument("--fixed-b", action="append", required=True, help="NAME:ADAPTER_DIR, one per --task")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--export-init-only", action="store_true",
                   help="write bank-A + source-B adapters to --output-dir and exit (no training)")
    args, rest = p.parse_known_args(argv)
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    fixed_b = parse_fixed_b(args.fixed_b)

    if args.export_init_only:
        export_init(args.output_dir, args.init_a_bank, fixed_b)
        print(f"INIT_EXPORT done: {sorted(fixed_b)} -> {args.output_dir}", flush=True)
        return 0

    _patch_setup(args.init_a_bank, fixed_b)
    rc = train.main(rest + ["--output-dir", args.output_dir])
    if rc == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        stats = check_exports(args.output_dir, fixed_b, args.init_a_bank)
        for name, src in fixed_b.items():
            _record_provenance(Path(args.output_dir) / name, args.init_a_bank, src, trained=True)
        print(f"B_FIXED_CHECK passed: B == source for {sorted(fixed_b)}; shared A {stats}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
