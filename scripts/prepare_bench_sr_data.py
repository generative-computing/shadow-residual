# SPDX-License-Identifier: Apache-2.0
"""Render the adapter benchmark's 4 intrinsic training sets into SR training rows.

Documents are split out of each row exactly as the benchmark's
``adapter_bench/train/render.py`` does (``document_turns`` / ``rebuild_documents``
per dataset) and placed back with the benchmark's own
``prompts.with_documents(..., "tool_json_after_question")`` — the style
``adapters.yaml`` uses to prompt SR on granite-4.2 — so training prompts match
the eval prompts. Output rows carry only ``messages`` (documents are already in
them), so the SR trainer's own document injection never runs.

    python scripts/prepare_bench_sr_data.py --bench-harness <dir with benchmarks/> \
        --datasets-root /danieloh_cos/granite-switch/adapter-bench/datasets/train \
        --out /workspace/bench_sr_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASETS = {
    "answerability": {"id": "g42-alora-4.1-split", "document_turns": True, "rebuild": "index_and_content"},
    "guardian_core": {"id": "guardian-ood-12k", "document_turns": False, "rebuild": None},
    "query_clarification": {"id": "unified-compare-data-v4", "document_turns": True, "rebuild": None},
    "requirement_check": {"id": "core-library-train-v2", "document_turns": False, "rebuild": None},
}
STYLE = "tool_json_after_question"


def documents_of(content) -> list:
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return [{"text": content}]
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        return [d if isinstance(d, dict) else {"text": str(d)} for d in content]
    return [{"text": json.dumps(content, ensure_ascii=False)}]


def split_documents(row: dict, document_turns: bool, rebuild: str | None):
    messages = [m for m in row.get("messages") or [] if isinstance(m, dict)]
    documents = row.get("documents")
    if isinstance(documents, str):
        documents = documents_of(documents)
    documents = list(documents or [])
    kept = []
    for m in messages:
        if document_turns and m.get("role") in ("documents", "tool"):
            if rebuild == "index_and_content":
                documents.append({"document_id": len(documents), "content": m.get("content")})
            else:
                documents += documents_of(m.get("content"))
        else:
            kept.append(m)
    return kept, documents


def render_row(row: dict, spec: dict, with_documents) -> dict | None:
    messages, documents = split_documents(row, spec["document_turns"], spec["rebuild"])
    if not messages or messages[-1].get("role") != "assistant":
        return None
    placed, leftover = with_documents(messages, documents, STYLE)
    if leftover:
        return None
    out = {"messages": placed}
    if row.get("tools"):
        out["tools"] = row["tools"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-harness", required=True, type=Path,
                   help="directory containing benchmarks/adapter_eval (granite-switch checkout)")
    p.add_argument("--datasets-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.bench_harness))
    from benchmarks.adapter_eval.prompts import with_documents

    summary = {}
    for intrinsic, spec in DATASETS.items():
        src = args.datasets_root / intrinsic / spec["id"]
        dest = args.out / intrinsic
        dest.mkdir(parents=True, exist_ok=True)
        summary[intrinsic] = {}
        for split in ("train", "val"):
            kept, dropped = [], 0
            with (src / f"{split}.jsonl").open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = render_row(json.loads(line), spec, with_documents)
                    if r is None:
                        dropped += 1
                    else:
                        kept.append(r)
            with (dest / f"{split}.jsonl").open("w") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            summary[intrinsic][split] = {"rows": len(kept), "dropped": dropped}
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
