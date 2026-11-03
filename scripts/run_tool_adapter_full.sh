#!/usr/bin/env bash
# Full-scale (CPU/MPS) launcher for the SR-adapter-as-tool answerability run on
# Granite 4.2-3b. Trains the FULL set (3191 rows) for 2 epochs, skips in-job
# generation, then runs the structural/accuracy/consistency eval over the full
# 354-row val set. Sibling of run_tool_adapter_local.sh (120-row flow check).
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/Users/antonp/models/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DISABLED=true
export BASE_MODEL=ibm-research/granite-4.2-3b-prerelease-r260622a
export DATA_ROOT=data/tool_adapter
export RUN_DIR=runs/tool_adapter_g42_3b_full
export EVAL_DATA=data/tool_adapter/val.jsonl
mkdir -p "$RUN_DIR"

echo "[$(date)] starting training" | tee -a "$RUN_DIR/run.log"
.venv/bin/python -m shadow_residual.training.train \
  --config src/shadow_residual/config/sr-tool-adapter-answerability-g42.yaml \
  --train-data data/tool_adapter/train.jsonl \
  --val-data   data/tool_adapter/val.jsonl \
  --output-dir "$RUN_DIR" \
  --epochs 2 \
  --skip-post-train-generate >> "$RUN_DIR/run.log" 2>&1
rc=$?
echo "[$(date)] training exit rc=$rc" | tee -a "$RUN_DIR/run.log"
if [ $rc -ne 0 ]; then exit $rc; fi

echo "[$(date)] starting eval" | tee -a "$RUN_DIR/run.log"
.venv/bin/python scripts/eval_tool_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$RUN_DIR" \
  --val data/tool_adapter/val.jsonl \
  --limit 354 --max-new-tokens 160 \
  --dump "$RUN_DIR/eval_dump.jsonl" \
  --out  "$RUN_DIR/eval.json" >> "$RUN_DIR/run.log" 2>&1
echo "[$(date)] eval done" | tee -a "$RUN_DIR/run.log"
cat "$RUN_DIR/eval.json"
