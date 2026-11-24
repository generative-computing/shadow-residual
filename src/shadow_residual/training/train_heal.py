# SPDX-License-Identifier: Apache-2.0
"""``train.py`` that heals a compressed adapter: shared A frozen, B starts from the compressed B.

    accelerate launch ... -m shadow_residual.training.train_heal \
        --frozen-a <variant>/bank --init-b-from <variant>/<name> <every train.py argument>

Both come from ``compress_adapters``. The A bank is installed and frozen exactly as in
``train_frozen_a``; then every ``lora_B`` is overwritten with the compressed adapter's
B, so step 0 equals the training-free compressed adapter. The config rank and α must
match the compressed adapter (r = R, α = 2R).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import torch

from shadow_residual.training import factory, train
from shadow_residual.training.compress_a import ADAPTER_FILE, CONFIG_FILE, site_scaling
from shadow_residual.training.frozen_a import A_FILE, install_frozen_a, saved_a_mismatches, site_key

logger = logging.getLogger(__name__)


def install_b(peft_model, adapter_dir: str | Path, adapter_name: str = "default") -> int:
    """Copy every ``lora_B`` from a saved adapter; the LoRA scaling must match."""
    from peft.tuners.lora.layer import LoraLayer
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    saved = load_file(str(adapter_dir / ADAPTER_FILE))
    cfg = json.loads((adapter_dir / CONFIG_FILE).read_text())
    by_key = {site_key(k.split(".lora_B.")[0]): v for k, v in saved.items() if ".lora_B." in k}
    n = 0
    for name, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer) or adapter_name not in module.lora_B:
            continue
        key = site_key(name)
        if key not in by_key:
            raise KeyError(f"no lora_B for {name!r} (key {key!r}) in {adapter_dir}")
        weight = module.lora_B[adapter_name].weight
        b = by_key[key]
        if tuple(weight.shape) != tuple(b.shape):
            raise ValueError(f"{name}: lora_B shape {tuple(weight.shape)} != saved {tuple(b.shape)}")
        want = site_scaling(cfg, key.split(".")[2], b.shape[1])
        if abs(module.scaling[adapter_name] - want) > 1e-9:
            raise ValueError(f"{name}: scaling {module.scaling[adapter_name]} != saved adapter's {want}")
        if weight.device.type != "meta":
            with torch.no_grad():
                weight.copy_(b.to(device=weight.device, dtype=weight.dtype))
        n += 1
    if n == 0:
        raise ValueError("no LoRA layers found to install B into")
    return n


def _patch_factory(frozen_a: str, init_b_from: str) -> None:
    bad = saved_a_mismatches(init_b_from, frozen_a)
    if bad:
        raise ValueError(f"{init_b_from} was not compressed onto bank {frozen_a}: {len(bad)} lora_A differ, e.g. {bad[:2]}")
    build = factory.get_shadow_residual_peft_model

    def build_healing(*args, **kwargs):
        model = build(*args, **kwargs)
        logger.info("Frozen A from %s: %s", frozen_a, install_frozen_a(model, frozen_a))
        logger.info("lora_B from %s: %d tensors", init_b_from, install_b(model, init_b_from))
        return model

    factory.get_shadow_residual_peft_model = build_healing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--frozen-a", required=True)
    p.add_argument("--init-b-from", required=True)
    p.add_argument("--output-dir", default=None)
    args, rest = p.parse_known_args(argv)
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    rest += ["--output-dir", args.output_dir]

    _patch_factory(args.frozen_a, args.init_b_from)
    rc = train.main(rest)
    if rc == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        out = Path(args.output_dir)
        bad = saved_a_mismatches(out, args.frozen_a)
        if bad:
            raise RuntimeError(f"saved lora_A differs from the frozen A for {len(bad)} tensors, e.g. {bad[:3]}")
        cfg = json.loads((out / CONFIG_FILE).read_text())
        sha = hashlib.sha256((Path(args.frozen_a) / A_FILE).read_bytes()).hexdigest()
        cfg["a_compression_heal"] = {"frozen_a": args.frozen_a, "frozen_a_sha256": sha, "init_b_from": args.init_b_from}
        (out / CONFIG_FILE).write_text(json.dumps(cfg, indent=2))
        print(f"HEAL_CHECK passed: every saved lora_A equals {args.frozen_a}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
