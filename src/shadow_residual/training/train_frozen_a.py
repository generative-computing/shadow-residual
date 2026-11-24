# SPDX-License-Identifier: Apache-2.0
"""``train.py`` with the SR-FrozenA adapter: precomputed, frozen LoRA A; only B trains.

    accelerate launch ... -m shadow_residual.training.train_frozen_a \
        --frozen-a <dir from compute_frozen_a> <every train.py argument>

The frozen A is copied into ``lora_A`` right after the SR PEFT model is built (on
every materialized rank, before the Trainer builds its optimizer), so A never
receives an optimizer update. After training, rank 0 checks that the saved
``lora_A`` still equals the frozen A and records the provenance in
``adapter_config.json``. The checkpoint is an ordinary PEFT LoRA adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from shadow_residual.training import factory, train
from shadow_residual.training.frozen_a import (
    A_FILE,
    install_frozen_a,
    randomize_lora_b,
    read_manifest,
    saved_a_mismatches,
)

logger = logging.getLogger(__name__)


def _patch_factory(frozen_a: str, lora_b_init: str = "zeros", lora_b_seed: int = 0) -> None:
    build = factory.get_shadow_residual_peft_model

    def build_with_frozen_a(*args, **kwargs):
        model = build(*args, **kwargs)
        counts = install_frozen_a(model, frozen_a)
        logger.info("Frozen A from %s: %s", frozen_a, counts)
        if lora_b_init == "random":
            n = randomize_lora_b(model, seed=lora_b_seed)
            logger.info("lora_B re-drawn kaiming_uniform(a=sqrt5), seed %d: %d tensors", lora_b_seed, n)
        return model

    factory.get_shadow_residual_peft_model = build_with_frozen_a


def _record_provenance(out: Path, frozen_a: str, lora_b_init: str, lora_b_seed: int) -> None:
    sha = hashlib.sha256((Path(frozen_a) / A_FILE).read_bytes()).hexdigest()
    manifest = read_manifest(frozen_a)
    cfg_path = out / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["frozen_lora_a"] = {
        "source": str(frozen_a),
        "sha256": sha,
        "rank": manifest["rank"],
        "recipe": manifest["recipe"],
        "feature_scale": manifest.get("feature_scale"),
        "lora_b_init": lora_b_init if lora_b_init == "zeros" else f"random(seed={lora_b_seed})",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--frozen-a", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--lora-b-init", choices=["zeros", "random"], default="zeros",
                   help="zeros = stock LoRA (adapter starts at the base); random = stock nn.Linear draw")
    p.add_argument("--lora-b-seed", type=int, default=0)
    args, rest = p.parse_known_args(argv)
    if args.output_dir is None:
        raise SystemExit("--output-dir is required (the frozen-A check reads the saved adapter)")
    rest += ["--output-dir", args.output_dir]

    _patch_factory(args.frozen_a, args.lora_b_init, args.lora_b_seed)
    rc = train.main(rest)
    if rc == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        out = Path(args.output_dir)
        bad = saved_a_mismatches(out, args.frozen_a)
        if bad:
            raise RuntimeError(f"saved lora_A differs from the frozen A for {len(bad)} tensors, e.g. {bad[:3]}")
        _record_provenance(out, args.frozen_a, args.lora_b_init, args.lora_b_seed)
        print(f"FROZEN_A_CHECK passed: every saved lora_A equals {args.frozen_a}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
