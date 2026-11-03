#!/usr/bin/env bash
# Local (CPU/MPS) launcher for the SR-adapter-as-tool answerability run on
# Granite 4.2-3b. Trains the small balanced subset for 1 epoch, skips in-job
# generation, then runs the structural/accuracy/consistency eval.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/Users/antonp/models/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export BASE_MODEL=ibm-research/granite-4.2-3b-prerelease-r260622a
export DATA_ROOT=data/tool_adapter
export RUN_DIR=runs/tool_adapter_g42_3b
export EVAL_DATA=data/tool_adapter/val_run.jsonl
mkdir -p "$RUN_DIR"

echo "[$(date)] starting training" | tee -a "$RUN_DIR/run.log"
.venv/bin/python -m shadow_residual.training.train \
  --config src/shadow_residual/config/sr-tool-adapter-answerability-g42.yaml \
  --train-data data/tool_adapter/train_run.jsonl \
  --val-data   data/tool_adapter/val_run.jsonl \
  --output-dir "$RUN_DIR" \
  --skip-post-train-generate >> "$RUN_DIR/run.log" 2>&1
rc=$?
echo "[$(date)] training exit rc=$rc" | tee -a "$RUN_DIR/run.log"
if [ $rc -ne 0 ]; then exit $rc; fi

echo "[$(date)] starting eval" | tee -a "$RUN_DIR/run.log"
.venv/bin/python scripts/eval_tool_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$RUN_DIR" \
  --val data/tool_adapter/val_run.jsonl \
  --limit 60 --max-new-tokens 160 \
  --dump "$RUN_DIR/eval_dump.jsonl" \
  --out  "$RUN_DIR/eval.json" >> "$RUN_DIR/run.log" 2>&1
echo "[$(date)] eval done" | tee -a "$RUN_DIR/run.log"
cat "$RUN_DIR/eval.json"
