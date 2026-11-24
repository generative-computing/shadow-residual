#!/usr/bin/env bash
# Local (CPU/MPS) launcher for the PLAIN SR answerability run on Granite 4.2-3b.
# Standard answerability flow (no tool-call): the adapter reads question+docs and
# answers with the verdict label. Uses the stock g42 config.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/Users/antonp/models/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DISABLED=true
export BASE_MODEL=ibm-granite/granite-4.2-3b
export DATA_ROOT=data/plain_answerability
export RUN_DIR=runs/plain_answerability_g42_3b
export EVAL_DATA=data/plain_answerability/val_run.jsonl
mkdir -p "$RUN_DIR"

echo "[$(date)] starting training" | tee -a "$RUN_DIR/run.log"
.venv/bin/python -m shadow_residual.training.train \
  --config src/shadow_residual/config/sr-qo-mlp-r32-c32-sharedkv-g42.yaml \
  --train-data data/plain_answerability/train_run.jsonl \
  --val-data   data/plain_answerability/val_run.jsonl \
  --output-dir "$RUN_DIR" \
  --epochs 1 \
  --skip-post-train-generate >> "$RUN_DIR/run.log" 2>&1
rc=$?
echo "[$(date)] training exit rc=$rc" | tee -a "$RUN_DIR/run.log"
if [ $rc -ne 0 ]; then exit $rc; fi

echo "[$(date)] starting eval" | tee -a "$RUN_DIR/run.log"
.venv/bin/python scripts/eval_plain_answerability.py \
  --base-model "$BASE_MODEL" \
  --adapter "$RUN_DIR" \
  --val data/plain_answerability/val_run.jsonl \
  --limit 60 --max-new-tokens 12 \
  --dump "$RUN_DIR/eval_dump.jsonl" \
  --out  "$RUN_DIR/eval.json" >> "$RUN_DIR/run.log" 2>&1
echo "[$(date)] eval done" | tee -a "$RUN_DIR/run.log"
cat "$RUN_DIR/eval.json"
