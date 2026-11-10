# SPDX-License-Identifier: Apache-2.0
"""Evaluate a read-out-stream SR adapter on one intrinsic with the adapter benchmark's code.

Same job as ``scripts/bench_eval.py`` (the benchmark's ``sr`` reference column on
granite-4.2-3b), except the SR base is built with the ``adapter_stream`` the
adapter was trained under (read from its ``adapter_config.json``). The harness's
``hf_generate.load_model`` calls ``build_sr_base`` without it, so its ``sr``
branch is replaced by one that passes it; generation, the loaded-weights check
and scoring are the harness's own.

    python scripts/readout_bench_eval.py --bench-harness <dir> --base-model <path> \
        --adapter <adapter dir> --intrinsic answerability \
        --eval-root /danieloh_cos/granite-switch/adapter-bench/granite-4.2-3b/v1/eval --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INTRINSICS = {
    "answerability": ("answerability", 200, "accuracy"),
    "guardian_core": ("guardian", 24, "accuracy"),
    "query_clarification": ("query_clarification", 160, "overall_accuracy"),
    "requirement_check": ("requirements", 200, "balanced_accuracy"),
}


def _sr_loader(original):
    def load_model(job: dict):
        if job["column"] != "sr":
            return original(job)
        import torch
        from peft import PeftConfig, PeftModel

        from shadow_residual.shadow_residual.build import build_sr_base
        from shadow_residual.training.generation_utils import (
            read_adapter_stream_from_adapter,
            read_share_moe_routing_from_adapter,
        )

        adapter_dir = job["adapter_dir"]
        stream = read_adapter_stream_from_adapter(adapter_dir)
        print(f"[readout_bench_eval] adapter_stream={stream}", flush=True)
        model = build_sr_base(
            job["base_model"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            share_moe_routing=read_share_moe_routing_from_adapter(adapter_dir),
            adapter_stream=stream,
        )
        config = PeftConfig.from_pretrained(adapter_dir)
        config.task_type = config.task_type or "CAUSAL_LM"
        model = PeftModel.from_pretrained(model, adapter_dir, config=config)
        if stream == "readout":
            _check_gamma_loaded(model, adapter_dir)
        return model.to("cuda").eval()

    return load_model


def _check_gamma_loaded(model, adapter_dir: str) -> None:
    """The harness checks only LoRA weights; check the saved readout_norm gammas too."""
    from safetensors.torch import load_file

    saved = {
        k: v for k, v in load_file(str(Path(adapter_dir) / "adapter_model.safetensors")).items()
        if ".readout_norm." in k
    }
    if not saved:
        raise RuntimeError("read-out adapter has no readout_norm weights")
    state = model.state_dict()
    bad = []
    for k, v in saved.items():
        loaded = state.get(k.replace(".readout_norm.weight", ".readout_norm.modules_to_save.default.weight"))
        if loaded is None or not loaded.float().cpu().equal(v.to(loaded.dtype).float()):
            bad.append(k)
    if bad:
        raise RuntimeError(f"{len(bad)} readout_norm weights not loaded, e.g. {bad[:3]}")
    moved = sum(float((v.float() - 1).abs().max()) > 0 for v in saved.values())
    print(f"[readout_bench_eval] {len(saved)} readout_norm gammas loaded ({moved} moved from 1)", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-harness", required=True, type=Path)
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--intrinsic", required=True, choices=sorted(INTRINSICS))
    p.add_argument("--eval-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.bench_harness))
    from benchmarks.adapter_eval import hf_generate
    from benchmarks.adapter_eval.scorers import ScoreContext, get_scorer
    from benchmarks.adapter_eval.staged import read_jsonl

    hf_generate.load_model = _sr_loader(hf_generate.load_model)

    scorer, max_new, headline = INTRINSICS[args.intrinsic]
    eval_dir = args.eval_root / args.intrinsic
    args.out.mkdir(parents=True, exist_ok=True)
    job = {
        "key": f"{args.intrinsic}/sr",
        "column": "sr",
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter),
        "eval_path": str(eval_dir / "evaluation.jsonl"),
        "out_path": str(args.out / "predictions.jsonl"),
        "status_path": str(args.out / "status.json"),
        "limit": args.limit,
        "max_new_tokens": max_new,
        "max_model_len": 32768,
        "token_budget": 262144,
        "max_batch": 64,
        "documents": "tool_json_after_question",
        "chat_template_kwargs": {"enable_thinking": False},
        "instruction": None,
    }
    (args.out / "job.json").write_text(json.dumps(job, indent=2))
    hf_generate.main([str(args.out / "job.json")])
    status = json.loads((args.out / "status.json").read_text())
    if not status.get("ok"):
        print(f"GENERATION FAILED: {status}", flush=True)
        return 1
    rows = read_jsonl(args.out / "predictions.jsonl")
    result = get_scorer(scorer)(rows, ScoreContext(eval_dir=eval_dir, env={}))
    (args.out / "score.json").write_text(
        json.dumps({"metrics": result.metrics, "details": result.details, "status": status}, indent=2)
    )
    print(f"SCORE {args.intrinsic} {headline}={result.metrics[headline]:.4f} "
          f"metrics={json.dumps(result.metrics)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
