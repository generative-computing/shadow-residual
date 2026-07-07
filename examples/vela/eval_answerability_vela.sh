#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Answerability evaluation on Vela2
#
# Supports multiple modes via EVAL_MODE env var:
#   switch-vllm (default) : granite-switch model via vLLM
#   switch                : granite-switch model via HF backend
#   peft                  : PEFT LoRA adapter via HF backend
#
# For switch modes, composes the model if not already built.
# Set ADAPTER_TECHNOLOGY to control which adapter variant is used:
#   alora (default) : uses aLoRA adapters (composer default preference)
#   lora            : forces LoRA-only by removing alora dirs before compose
#
# For peft mode, uses BASE_MODEL, ADAPTER_PATH, ADAPTER_SUB_PATH env vars.
# ============================================================================

DATASET_PATH="${DATASET_PATH:-/sft-stage-bucket-vela2/answerability_data/answerability_eval}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-/sft-stage-bucket-vela2/answerability_data/results}"
INTRINSIC_NAME="${INTRINSIC_NAME:-answerability}"
EVAL_MODE="${EVAL_MODE:-switch-vllm}"
BASE_MODEL="${BASE_MODEL:-ibm-granite/granite-4.0-micro}"
ADAPTER_PATH="${ADAPTER_PATH:-ibm-granite/granite-lib-rag-r1.0}"
ADAPTER_SUB_PATH="${ADAPTER_SUB_PATH:-answerability/granite-4.0-micro/alora}"
ADAPTER_TECHNOLOGY="${ADAPTER_TECHNOLOGY:-alora}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# DMF-based base model pull (set DMF_MODEL to enable, e.g. "granite-4.1-3b")
DMF_MODEL="${DMF_MODEL:-}"
DMF_NAMESPACE="${DMF_NAMESPACE:-base_training}"
DMF_TAG="${DMF_TAG:-model_shared}"
DMF_REVISION="${DMF_REVISION:-r260401a}"
DMF_MODEL_DIR="${DMF_MODEL_DIR:-/workspace/base_model}"
LAKEHOUSE_TOKEN="${LAKEHOUSE_TOKEN:-}"

# Local adapter path on PVC (set to use local adapters instead of HF hub)
LOCAL_ADAPTER_PATH="${LOCAL_ADAPTER_PATH:-}"

# ── Ensure libcudart.so.12 is on LD_LIBRARY_PATH (pip nvidia-cuda-runtime-cu12)
CUDA_RT_LIB=$(python3 -c "import nvidia.cuda_runtime, os; print(os.path.join(nvidia.cuda_runtime.__path__[0], 'lib'))" 2>/dev/null || true)
if [ -n "$CUDA_RT_LIB" ]; then
    export LD_LIBRARY_PATH="${CUDA_RT_LIB}:${LD_LIBRARY_PATH:-}"
    echo "Added CUDA runtime lib to LD_LIBRARY_PATH: $CUDA_RT_LIB"
fi

echo "============= EVAL MODE: $EVAL_MODE =============="
echo "============= ADAPTER TECHNOLOGY: $ADAPTER_TECHNOLOGY =============="

# ── DMF base model pull (if DMF_MODEL is set) ────────────────────────────────
if [ -n "$DMF_MODEL" ]; then
    echo "============= PULLING BASE MODEL VIA DMF =============="
    echo "Model: $DMF_MODEL, Namespace: $DMF_NAMESPACE, Tag: $DMF_TAG, Revision: $DMF_REVISION"

    export LAKEHOUSE_TOKEN="${LAKEHOUSE_TOKEN}"
    export LAKEHOUSE_ENVIRONMENT="${LAKEHOUSE_ENVIRONMENT:-PROD}"

    pip install --no-input dmf-lib 2>/dev/null \
        || pip install --no-input "dmf-lib @ git+https://${GHP_PERSONAL_TOKEN}@github.ibm.com/arc/dmf-library.git" \
        || { echo "ERROR: Could not install dmf-lib"; exit 1; }

    mkdir -p "$DMF_MODEL_DIR"
    cd "$DMF_MODEL_DIR"
    dmf model pull "$DMF_MODEL" --namespace "$DMF_NAMESPACE" -t "$DMF_TAG" --revision "$DMF_REVISION" --dir .
    cd -

    # Find the actual model directory (DMF creates a subdirectory)
    DMF_PULLED_DIR=$(find "$DMF_MODEL_DIR" -name "config.json" -maxdepth 3 -exec dirname {} \; | head -1)
    if [ -z "$DMF_PULLED_DIR" ]; then
        echo "ERROR: Could not find config.json after DMF pull"
        exit 1
    fi
    echo "Base model pulled to: $DMF_PULLED_DIR"
    BASE_MODEL="$DMF_PULLED_DIR"
fi

if [ "$EVAL_MODE" = "peft" ]; then
    # ── PEFT mode: use base model + adapter directly ─────────────────────────
    echo "============= RUNNING PEFT EVALUATION =============="
    if [ -n "$LOCAL_ADAPTER_PATH" ]; then
        # Local adapter: path is the direct adapter dir, sub-path is "."
        PEFT_ADAPTER_PATH="${LOCAL_ADAPTER_PATH}/${ADAPTER_SUB_PATH}"
        PEFT_SUB_PATH="."
    else
        PEFT_ADAPTER_PATH="$ADAPTER_PATH"
        PEFT_SUB_PATH="$ADAPTER_SUB_PATH"
    fi
    python eval/answerability/answerability_eval.py \
        --mode peft \
        --base-model "$BASE_MODEL" \
        --adapter-path "$PEFT_ADAPTER_PATH" \
        --adapter-sub-path "$PEFT_SUB_PATH" \
        --dataset-path "$DATASET_PATH" \
        --output-path "$EVAL_RESULTS_DIR" \
        --device cuda \
        --batch-size "$BATCH_SIZE"
else
    # ── Switch modes: compose model if needed, then evaluate ─────────────────
    GRANITE_SWITCH_MODEL_PATH="${MODEL_CHECKPOINTS_DIR}/granite-switch-${ADAPTER_TECHNOLOGY}"

    if [ "${FORCE_RECOMPOSE:-0}" = "1" ] && [ -d "$GRANITE_SWITCH_MODEL_PATH" ]; then
        echo "FORCE_RECOMPOSE=1: removing old composed model at $GRANITE_SWITCH_MODEL_PATH"
        rm -rf "$GRANITE_SWITCH_MODEL_PATH"
    fi

    if [ ! -d "$GRANITE_SWITCH_MODEL_PATH" ]; then
        echo "============= COMPOSING GRANITE SWITCH MODEL (${ADAPTER_TECHNOLOGY}) =============="

        # Determine adapter source — direct path to the adapter directory
        if [ -n "$LOCAL_ADAPTER_PATH" ]; then
            ADAPTER_DIR="${LOCAL_ADAPTER_PATH}/${ADAPTER_SUB_PATH}"
            echo "Using local adapter from: $ADAPTER_DIR"
        else
            ADAPTER_DIR=""
        fi

        if [ -n "$ADAPTER_DIR" ]; then
            python -m granite_switch.composer.compose_granite_switch \
                --base-model "$BASE_MODEL" \
                --adapters "$ADAPTER_DIR" \
                --output "$GRANITE_SWITCH_MODEL_PATH"
        elif [ -n "$ADAPTER_PATH" ]; then
            # No local adapter — use HF hub adapter path
            echo "Using HF hub adapter: $ADAPTER_PATH"
            python -m granite_switch.composer.compose_granite_switch \
                --base-model "$BASE_MODEL" \
                --adapters "$ADAPTER_PATH" \
                --output "$GRANITE_SWITCH_MODEL_PATH"
        else
            echo "ERROR: No adapters specified (set LOCAL_ADAPTER_PATH or ADAPTER_PATH)"
            exit 1
        fi
    else
        echo "============= USING EXISTING GRANITE SWITCH MODEL =============="
        echo "Path: $GRANITE_SWITCH_MODEL_PATH"
    fi

    # ── Build eval command ─────────────────────────────────────────────────
    EVAL_CMD=(
        python eval/answerability/answerability_eval.py
        --mode "$EVAL_MODE"
        --model-path "$GRANITE_SWITCH_MODEL_PATH"
        --dataset-path "$DATASET_PATH"
        --output-path "$EVAL_RESULTS_DIR"
        --intrinsic-name "$INTRINSIC_NAME"
        --batch-size "$BATCH_SIZE"
    )

    # Pass tensor-parallel-size for vLLM mode (defaults to GPUS_PER_POD or 1)
    if [ "$EVAL_MODE" = "switch-vllm" ]; then
        TP_SIZE="${TENSOR_PARALLEL_SIZE:-${GPUS_PER_POD:-1}}"
        EVAL_CMD+=(--tensor-parallel-size "$TP_SIZE")
        echo "============= vLLM tensor_parallel_size=$TP_SIZE =============="
    fi

    echo "============= RUNNING SWITCH EVALUATION =============="
    "${EVAL_CMD[@]}"
fi

echo "============= DONE =============="
