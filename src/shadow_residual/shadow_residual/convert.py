# SPDX-License-Identifier: Apache-2.0
"""Convert an upstream HF Granite checkpoint into a standalone SR checkpoint.

The SR model is otherwise assembled in memory on every load (upstream config →
SR config, fused→unfused weight slicing, zero cross-stream sites — see
:mod:`shadow_residual.shadow_residual.build`). Converting once writes that model
to disk as a regular HF checkpoint (``model_type: "shadow_residual"``) that loads
with stock ``from_pretrained`` and can be pushed to the Hub::

    sr-convert --base ibm-granite/granite-4.1-3b --output ./granite-4.1-3b-sr \\
               --repo-id AlonMal/granite-4.1-3b-sr --push

    import shadow_residual.shadow_residual  # registers the SR model type
    model = AutoModelForCausalLM.from_pretrained("AlonMal/granite-4.1-3b-sr")

The checkpoint carries the upstream tokenizer, chat template and generation
config, so one model id is enough for training and serving. The frozen all-zero
cross-stream weights are not written (``_keys_to_ignore_on_save``); they are
re-zeroed on load. Conversion is generic over the bases ``build_sr_base``
supports (dense Granite 4.x, tied or untied lm_head, and granitemoe).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch

from .build import build_sr_base, is_sr_checkpoint, load_sr_checkpoint

logger = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def convert_to_sr_checkpoint(
    base_model_name_or_path: str,
    output_dir: Union[str, os.PathLike],
    *,
    torch_dtype: Union[torch.dtype, str, None] = "auto",
    max_shard_size: str = "5GB",
    verify: bool = True,
    repo_id: Optional[str] = None,
    push: bool = False,
    private: bool = False,
    token: Optional[str] = None,
) -> Path:
    """Convert ``base_model_name_or_path`` (upstream Granite) to an SR checkpoint.

    Args:
        base_model_name_or_path: upstream HF Granite model id or local path.
        output_dir: directory the SR checkpoint is written to.
        torch_dtype: dtype of the saved weights. ``"auto"`` (default) keeps the
            upstream checkpoint's dtype.
        max_shard_size: safetensors shard size passed to ``save_pretrained``.
        verify: reload the written checkpoint with strict ``from_pretrained`` and
            check every tensor is identical to the converted model. Holds two
            copies of the model in host RAM; disable for very large bases.
        repo_id: Hub repo id (``"user/name"``) the checkpoint is published under;
            used in the generated model card, and as the upload target with
            ``push``. Defaults to the output directory name in the card.
        push: upload the written checkpoint to ``repo_id``.
        private: create the Hub repo as private (only when it doesn't exist yet).
        token: Hub token; defaults to the cached login / ``HF_TOKEN``.

    Returns:
        The output directory.
    """
    if push and not repo_id:
        raise ValueError("push=True needs a repo_id to upload to.")
    if is_sr_checkpoint(base_model_name_or_path):
        raise ValueError(
            f"{base_model_name_or_path!r} is already an SR checkpoint "
            "(model_type 'shadow_residual'); convert from the upstream base instead."
        )
    dtype = _resolve_dtype(base_model_name_or_path, torch_dtype)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Building SR model from %s (dtype=%s) ...", base_model_name_or_path, dtype)
    sr_model = build_sr_base(base_model_name_or_path, torch_dtype=dtype)
    generation_config = _load_upstream_generation_config(base_model_name_or_path)
    if generation_config is not None:
        sr_model.generation_config = generation_config

    logger.info("Saving SR checkpoint to %s ...", out)
    sr_model.save_pretrained(out, max_shard_size=max_shard_size)
    tokenizer = _load_upstream_tokenizer(base_model_name_or_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    _write_model_card(out, base_model_name_or_path, model_id=repo_id or out.name)

    if verify:
        _verify_round_trip(sr_model, out, dtype)
    del sr_model

    if push:
        _push_to_hub(out, repo_id, base_model_name_or_path, private=private, token=token)
    return out


def _resolve_dtype(base_model_name_or_path: str, torch_dtype) -> torch.dtype:
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype not in (None, "auto"):
        return _DTYPES[torch_dtype]
    from transformers import AutoConfig

    dtype = getattr(AutoConfig.from_pretrained(base_model_name_or_path), "dtype", None)
    if isinstance(dtype, str):
        dtype = _DTYPES.get(dtype)
    return dtype if isinstance(dtype, torch.dtype) else torch.float32


def _load_upstream_generation_config(base_model_name_or_path: str):
    from transformers import GenerationConfig

    try:
        return GenerationConfig.from_pretrained(base_model_name_or_path)
    except OSError:
        logger.info("No generation_config.json on %s; using the model default.", base_model_name_or_path)
        return None


def _load_upstream_tokenizer(base_model_name_or_path: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(base_model_name_or_path)
    except (OSError, ValueError) as e:
        logger.warning(
            "No tokenizer found on %s (%s); the SR checkpoint will not carry one, "
            "so training/serving must load the tokenizer from elsewhere.",
            base_model_name_or_path, e,
        )
        return None


def _upstream_license(base_model_name_or_path: str) -> Optional[str]:
    from huggingface_hub import ModelCard

    local_readme = Path(base_model_name_or_path) / "README.md"
    try:
        card = ModelCard.load(str(local_readme) if local_readme.is_file() else base_model_name_or_path)
    except Exception:  # no card / offline — the license line is optional
        return None
    return getattr(card.data, "license", None)


def _write_model_card(out: Path, base_model_name_or_path: str, *, model_id: str) -> None:
    is_hub_id = not Path(base_model_name_or_path).exists()
    license_ = _upstream_license(base_model_name_or_path)
    meta = []
    if license_:
        meta.append(f"license: {license_}")
    if is_hub_id:
        meta.append(f"base_model: {base_model_name_or_path}")
    meta += ["library_name: transformers", "tags:", "- shadow-residual"]
    base_ref = (
        f"[{base_model_name_or_path}](https://huggingface.co/{base_model_name_or_path})"
        if is_hub_id else f"`{base_model_name_or_path}`"
    )
    card = f"""---
{chr(10).join(meta)}
---

# {model_id.split("/")[-1]}

{base_ref} converted to the **Shadow Residual (SR)** architecture — the base model
for training SR adapters with stock PEFT LoRA.

SR runs two streams through every decoder layer: a frozen base stream (the
unadapted base model) and a trainable adapter stream (LoRA deltas plus a per-layer
low-rank cross-stream injection from the base). The weights here are the base
model's own weights, re-laid-out for SR (fused projections split into separate
`q/k/v` and `gate/up` projections). With no adapter attached the model produces
the base model's outputs.

## Usage

Loading needs the `shadow-residual` package installed (it registers the
`shadow_residual` model type with `transformers`; no `trust_remote_code`).

```python
import shadow_residual.shadow_residual  # registers the SR model type
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_id}")
tokenizer = AutoTokenizer.from_pretrained("{model_id}")
```

Attach a trained SR adapter with stock PEFT:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(model, "<adapter>")
```

The frozen cross-stream sites are all-zero by construction and are not stored;
they are re-created on load.
"""
    (out / "README.md").write_text(card)


def _verify_round_trip(sr_model, out: Path, dtype: torch.dtype) -> None:
    logger.info("Verifying: reloading %s with strict from_pretrained ...", out)
    reloaded = load_sr_checkpoint(str(out), torch_dtype=dtype)
    got = reloaded.state_dict()
    expected = sr_model.state_dict()
    if set(got) != set(expected):
        raise RuntimeError(
            f"SR checkpoint round-trip changed the key set: "
            f"missing={sorted(set(expected) - set(got))[:5]} "
            f"extra={sorted(set(got) - set(expected))[:5]}"
        )
    differ = [k for k, v in expected.items() if not torch.equal(v, got[k])]
    if differ:
        raise RuntimeError(f"SR checkpoint round-trip changed {len(differ)} tensor(s), e.g. {differ[:5]}")
    logger.info("Verified: %d tensors identical after reload.", len(expected))


def _push_to_hub(out: Path, repo_id: str, base_model_name_or_path: str, *, private: bool, token) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    logger.info("Uploading %s to https://huggingface.co/%s ...", out, repo_id)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(out),
        commit_message=f"Convert {base_model_name_or_path} to shadow-residual",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sr-convert",
        description="Convert an upstream HF Granite checkpoint into an SR checkpoint.",
    )
    p.add_argument("--base", required=True, help="Upstream HF model id or local path.")
    p.add_argument("--output", required=True, help="Directory to write the SR checkpoint to.")
    p.add_argument(
        "--dtype", default="auto", choices=["auto", *_DTYPES],
        help="Dtype of the saved weights (default: the upstream checkpoint's dtype).",
    )
    p.add_argument("--max-shard-size", default="5GB")
    p.add_argument(
        "--no-verify", dest="verify", action="store_false",
        help="Skip the strict reload check (it holds two model copies in host RAM).",
    )
    p.add_argument(
        "--repo-id", help="Hub repo the checkpoint is published under, e.g. user/name "
        "(written into the model card; the upload target with --push).",
    )
    p.add_argument("--push", action="store_true", help="Upload the checkpoint to --repo-id.")
    p.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    return p


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    convert_to_sr_checkpoint(
        args.base,
        args.output,
        torch_dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        verify=args.verify,
        repo_id=args.repo_id,
        push=args.push,
        private=args.private,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["convert_to_sr_checkpoint"]
