# SPDX-License-Identifier: Apache-2.0
"""Answerability adapter evaluation for Granite models.

Supports two modes:
  --mode base       : Load base model only, no adapter (CPU/MPS/GPU) — baseline comparison
  --mode peft       : Load base model + PEFT LoRA adapter from HuggingFace (CPU/MPS/GPU)

The dataset_path directory should contain:
    - collection.txt    (Python list of collection names, one per sample)
    - setting.txt       (Python list of setting labels: gold/rag/a/b/c)
    - gold.jsonl
    - rag.jsonl
    - unanswerable_v2.a.jsonl
    - unanswerable_v2.b.jsonl
    - unanswerable_v2.c.jsonl

Each JSONL line has the format:
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},   # optional prior turns
        {"role": "documents", "content": ["doc1", "doc2", ...]},
        {"role": "answerability", "content": "answerable"|"unanswerable"}
    ]}

Example usage (Base mode — no adapter, baseline):
    python eval/answerability/answerability_eval.py --mode base \\
        --base-model ibm-granite/granite-4.0-micro \\
        --dataset-path ./scratch/answerability

Example usage (PEFT mode — HF adapter from hub):
    python eval/answerability/answerability_eval.py --mode peft \\
        --base-model ibm-granite/granite-4.0-micro \\
        --adapter-path ibm-granite/granite-lib-rag-r1.0 \\
        --adapter-sub-path answerability/granite-4.0-micro/alora \\
        --dataset-path ./scratch/answerability

"""

import argparse
import ast
import csv
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATASET_FILES = [
    "gold.jsonl",
    "rag.jsonl",
    "unanswerable_v2.a.jsonl",
    "unanswerable_v2.b.jsonl",
    "unanswerable_v2.c.jsonl",
]


def _eos_token_ids(tokenizer):
    """Return the list of end-of-turn token ids generation should stop on.

    Granite 4.2 (ChatML) ends a turn with <|im_end|>; Granite 4.1 uses
    <|end_of_text|>. Without an explicit stop, greedy decoding runs to
    max_new_tokens and repeats the label ('"unanswerable"unanswerable"...').
    Include the tokenizer's eos plus <|im_end|> when present (its id may differ
    from eos_token_id on some configs).
    """
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != tokenizer.unk_token_id and im_end not in ids:
            ids.append(im_end)
    except Exception:
        pass
    return ids or None


def load_jsonl(path):
    """Load a JSONL file and return a list of parsed dicts."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_sample(sample):
    """Extract conversation messages, documents, and ground truth from a sample.

    Returns:
        messages: list of {"role": "user"|"assistant", "content": ...} dicts
        documents: list of document strings
        label: "answerable" or "unanswerable"
    """
    messages = []
    documents = []
    label = None

    for msg in sample["messages"]:
        role = msg["role"]
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": msg["content"]})
        elif role == "documents":
            documents = msg["content"] if isinstance(msg["content"], list) else []
        elif role == "answerability":
            label = msg["content"]

    return messages, documents, label


def load_dataset(dataset_path, dataset_size=100000):
    """Load all JSONL files and metadata from the dataset directory.

    Returns:
        all_messages: list of message lists
        all_documents: list of document lists
        all_labels: list of ground truth labels
        collection_list: list of collection names
        setting_list: list of setting labels
    """
    all_messages = []
    all_documents = []
    all_labels = []
    metadata_indices = []
    metadata_offset = 0

    for filename in DATASET_FILES:
        filepath = os.path.join(dataset_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue
        samples = load_jsonl(filepath)
        selected_samples = samples[:dataset_size]
        metadata_indices.extend(
            range(metadata_offset, metadata_offset + len(selected_samples))
        )
        metadata_offset += len(samples)

        print(
            f"  {filename}: selected {len(selected_samples)} / {len(samples)} samples"
        )

        for sample in selected_samples:
            msgs, docs, label = parse_sample(sample)
            all_messages.append(msgs)
            all_documents.append(docs)
            all_labels.append(label)

    with open(os.path.join(dataset_path, "collection.txt"), encoding="utf-8") as f:
        collection_list = ast.literal_eval(f.read())

    with open(os.path.join(dataset_path, "setting.txt"), encoding="utf-8") as f:
        setting_list = ast.literal_eval(f.read())

    collection_list = [collection_list[i] for i in metadata_indices]
    setting_list = [setting_list[i] for i in metadata_indices]

    print(f"Loaded {len(all_labels)} samples from {len(DATASET_FILES)} files")
    print(f"  collection.txt: {len(collection_list)} entries")
    print(f"  setting.txt: {len(setting_list)} entries")

    return all_messages, all_documents, all_labels, collection_list, setting_list


# ---------------------------------------------------------------------------
# Inference — Base mode (no adapter, baseline)
# ---------------------------------------------------------------------------

def run_inference_base(base_model_name, all_messages, all_documents,
                       batch_size, device_str, enable_thinking=False,
                       max_new_tokens=30):
    """Run HuggingFace inference with the base model only (no adapter).

    This serves as a baseline to measure the adapter's contribution.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Resolve device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_str
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"\nDevice: {device}, dtype: {dtype}")

    print(f"Loading tokenizer from: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device in ("cpu", "mps"):
        model = model.to(device)
    model.eval()

    # Build prompts using the base model's chat template. render_chat folds
    # documents into a system message for ChatML tokenizers (Granite 4.2, whose
    # template ignores documents=) and passes them natively otherwise (4.1) — so
    # the RAG context reaches the model on both, matching training.
    from shadow_residual.training.chat_render import render_chat

    print(f"Formatting {len(all_messages)} prompts...")
    prompts = []
    for messages, documents in zip(all_messages, all_documents):
        doc_dicts = [{"text": d} for d in documents] if documents else []
        prompt = render_chat(
            tokenizer,
            messages,
            documents=doc_dicts,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt)

    # Generate in batches
    print(f"Running inference ({len(prompts)} samples, batch_size={batch_size})...")
    predictions = []
    start = time.time()

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=_eos_token_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(text)

        done = batch_end
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [{done}/{len(prompts)}] {rate:.1f} samples/s", end="\r")

    elapsed = time.time() - start
    print(f"\nInference complete in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} samples/s)")
    return predictions


# ---------------------------------------------------------------------------
# Inference — PEFT mode (HuggingFace + LoRA adapter)
# ---------------------------------------------------------------------------

def run_inference_peft(base_model_name, adapter_path, all_messages, all_documents,
                       batch_size, device_str, enable_thinking=False,
                       max_new_tokens=30):
    """Run HuggingFace inference with a PEFT LoRA adapter.

    Args:
        base_model_name: HF model ID (e.g. "ibm-granite/granite-4.0-micro")
        adapter_path: Local path or HF repo path to the PEFT adapter
        all_messages: list of conversation message lists
        all_documents: list of document string lists
        batch_size: number of samples per batch
        device_str: "auto", "cpu", or "cuda"
    """
    from transformers import AutoTokenizer
    from peft import PeftModel

    from shadow_residual.shadow_residual.build import build_sr_base
    from shadow_residual.training.generation_utils import (
        read_cross_stream_taps_from_adapter,
        read_cross_stream_type_from_adapter,
        read_share_moe_routing_from_adapter,
    )

    # Resolve device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_str
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"\nDevice: {device}, dtype: {dtype}")

    # Load tokenizer from adapter (has the chat template)
    print(f"Loading tokenizer from adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model + adapter through the SR loader. For LoRA/aLoRA
    # checkpoints (no "cross_stream" target) the SR model's runtime gate
    # takes the single-stream early-exit path — semantically equivalent
    # to upstream Granite + stock PEFT, but not bit-identical due to
    # unfused projections. For SR checkpoints this is the only correct
    # path: cross-stream LoRA weights need CrossStream sites to bind to,
    # which only exist on a ShadowResidualForCausalLM.
    print(f"Loading SR base + adapter from: {adapter_path}")
    # Source the MoE routing mode from the adapter itself (train.py records it in
    # adapter_config.json) so a shared-routing adapter is served with the same
    # forward path it was trained under — no hand-passed flag, no silent drift.
    # No-op / False for dense bases and older adapters lacking the key.
    share_moe_routing = read_share_moe_routing_from_adapter(adapter_path)
    print(f"share_moe_routing (from adapter_config.json): {share_moe_routing}")
    # Build exactly the cross-stream topology the adapter was trained with, or its
    # saved tensors have nothing to bind to. Two axes, both sourced from the saved
    # adapter_config.json: WHICH sites (taps) and WHAT KIND of module (type + dim).
    cross_stream_taps = read_cross_stream_taps_from_adapter(adapter_path)
    cross_stream_type, cross_stream_dim = read_cross_stream_type_from_adapter(
        adapter_path
    )
    print(
        f"cross_stream_taps={cross_stream_taps} type={cross_stream_type} "
        f"dim={cross_stream_dim} (from adapter_config.json)"
    )
    base_model = build_sr_base(
        base_model_name,
        torch_dtype=dtype,
        share_moe_routing=share_moe_routing,
        cross_stream_taps=cross_stream_taps,
        cross_stream_type=cross_stream_type,
        cross_stream_dim=cross_stream_dim,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()

    # Build prompts. render_chat renders documents natively for Granite 4.1 and
    # as an agentic <tool_response> for ChatML/4.2 (whose template ignores
    # documents=) — matching how the adapter was trained.
    from shadow_residual.training.chat_render import render_chat

    print(f"Formatting {len(all_messages)} prompts...")
    prompts = []
    for messages, documents in zip(all_messages, all_documents):
        doc_dicts = [{"text": d} for d in documents] if documents else []
        prompt = render_chat(
            tokenizer,
            messages,
            documents=doc_dicts,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt)

    # Generate in batches
    eos_ids = _eos_token_ids(tokenizer)
    print(f"Running inference ({len(prompts)} samples, batch_size={batch_size})...")
    predictions = []
    start = time.time()

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (strip the input)
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(text)

        done = batch_end
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [{done}/{len(prompts)}] {rate:.1f} samples/s", end="\r")

    elapsed = time.time() - start
    print(f"\nInference complete in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} samples/s)")
    return predictions



# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def classify_label(text):
    """Classify a raw prediction string into answerable/unanswerable/others."""
    if "unanswerable" in text:
        return "unanswerable"
    elif "answerable" in text:
        return "answerable"
    return "others"


def process_predictions(raw_predictions):
    """Process raw model outputs into classification labels.

    Granite 4.2 (ChatML) emits a ``<think>...</think>`` reasoning block before
    the answer; the actual label follows the closing tag. When ``</think>`` is
    present, parse only the text after it so the reasoning prefix (e.g.
    ``Okay, the user is asking...``) isn't mistaken for the label. Granite 4.1
    has no think block, so parsing is unchanged for it.
    """
    processed = []
    for pred in raw_predictions:
        pred = pred.strip()
        # 4.2: the verdict is whatever follows the last </think>.
        if "</think>" in pred:
            pred = pred.rsplit("</think>", 1)[1].strip()
        tokens = pred.split()
        first = tokens[0] if tokens else ""
        quoted = re.findall(r'"(.*?)"', first)
        if quoted:
            first = quoted[0]
        if first not in ("answerable", "unanswerable"):
            first = "others"
        processed.append(classify_label(first))
    return processed


def confusion_matrix(y_true, y_pred, classes):
    """Compute confusion matrix for given classes."""
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in classes and pred in classes:
            matrix[classes.index(true), classes.index(pred)] += 1
    return matrix


def compute_metrics(cm):
    """Compute per-class precision, recall, F1, and overall accuracy."""
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)

    for i in range(n):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision[i] = tp / max(tp + fp, 1e-10)
        recall[i] = tp / max(tp + fn, 1e-10)
        f1[i] = 2 * precision[i] * recall[i] / max(precision[i] + recall[i], 1e-10)

    accuracy = np.sum(np.diag(cm)) / max(np.sum(cm), 1)
    return precision, recall, f1, accuracy


def evaluate(result_df):
    """Run the full evaluation pipeline on a results DataFrame.

    Returns a flat metrics dict with keys:
        answerability_precision, answerability_recall, answerability_f1,
        unanswerability_precision, unanswerability_recall, unanswerability_f1,
        accuracy, weighted_f1
    """
    df = result_df[~result_df["collection"].isin(["mt-rag-govt-elser-512-100-20240611"])].copy()
    df = df[df["ground_truth"].isin(["answerable", "unanswerable"])]
    df = df[~((df["ground_truth"] == "answerable") & (df["setting"] == "rag"))]

    print(f"\nFiltered dataset: {len(df)} samples")
    print(df["ground_truth"].value_counts().to_string())

    # classes[0]=unanswerable, classes[1]=answerable, classes[2]=others
    classes = ["unanswerable", "answerable", "others"]
    cm = confusion_matrix(
        df["ground_truth"].tolist(),
        df["prediction"].tolist(),
        classes,
    )
    precision, recall, f1, accuracy = compute_metrics(cm)

    print(f"\nConfusion matrix:")
    print(cm)

    # Weighted F1 over answerable + unanswerable only (others has 0 support)
    support = [df["ground_truth"].value_counts().get(c, 0) for c in classes]
    weighted_f1 = sum(f1[i] * support[i] for i in range(len(classes))) / max(sum(support), 1)

    # Mean F1 (macro average of answerable and unanswerable F1)
    mean_f1 = (f1[1] + f1[0]) / 2.0

    # Per-subset accuracy stats for std dev and confidence interval
    # Compute accuracy for each evaluation subset (gold, a, b, c), then
    # report the std and CI across those group accuracies.
    # Exclude "rag" — after filtering answerable rag samples, only a small
    # unanswerable remnant remains which is not a proper evaluation subset.
    subset_accuracies = []
    for setting_val in sorted(df["setting"].unique()):
        if setting_val == "rag":
            continue
        subset = df[df["setting"] == setting_val]
        if len(subset) > 0:
            subset_acc = (subset["ground_truth"] == subset["prediction"]).mean()
            subset_accuracies.append(subset_acc)
    subset_accuracies = np.array(subset_accuracies)
    acc_std = float(subset_accuracies.std(ddof=1)) if len(subset_accuracies) > 1 else 0.0
    ci_95_lo = float(accuracy - 1.96 * acc_std)
    ci_95_hi = float(accuracy + 1.96 * acc_std)

    # Build flat metrics dict with the canonical metric names
    metrics = {
        "answerability_precision": float(precision[1]),
        "answerability_recall": float(recall[1]),
        "answerability_f1": float(f1[1]),
        "unanswerability_precision": float(precision[0]),
        "unanswerability_recall": float(recall[0]),
        "unanswerability_f1": float(f1[0]),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "mean_f1": float(mean_f1),
        "accuracy_std": float(acc_std),
        "accuracy_ci_95_lo": float(ci_95_lo),
        "accuracy_ci_95_hi": float(ci_95_hi),
    }

    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    return metrics, df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(metrics, result_df, output_path, model_display):
    """Save all evaluation outputs.

    Args:
        metrics: flat dict with keys answerability_precision, answerability_recall,
                 answerability_f1, unanswerability_precision, unanswerability_recall,
                 unanswerability_f1, accuracy, weighted_f1
        result_df: full predictions DataFrame
        output_path: directory to write files
        model_display: model name for display in tables
    """
    os.makedirs(output_path, exist_ok=True)

    # Full predictions
    result_df.to_csv(os.path.join(output_path, "result_df.csv"), index=False)

    # JSONL summary — all 8 metrics + metadata
    summary = {"intrinsic": "answerability", "total_samples": len(result_df)}
    for key, val in metrics.items():
        summary[key] = round(val, 4)

    with open(os.path.join(output_path, "eval_summary.jsonl"), "w") as f:
        f.write(json.dumps(summary) + "\n")

    # Markdown summary
    m = metrics
    md_lines = [
        "## Answerability Evaluation Results\n",
        f"**Model:** {model_display}\n",
        "| Metric | Value |",
        "|:-------|------:|",
        f"| answerability_precision | {m['answerability_precision']:.4f} |",
        f"| answerability_recall | {m['answerability_recall']:.4f} |",
        f"| answerability_f1 | {m['answerability_f1']:.4f} |",
        f"| unanswerability_precision | {m['unanswerability_precision']:.4f} |",
        f"| unanswerability_recall | {m['unanswerability_recall']:.4f} |",
        f"| unanswerability_f1 | {m['unanswerability_f1']:.4f} |",
        f"| accuracy | {m['accuracy']:.4f} |",
        f"| weighted_f1 | {m['weighted_f1']:.4f} |",
        f"| mean_f1 | {m['mean_f1']:.4f} |",
        f"| accuracy_std | {m['accuracy_std']:.4f} |",
        f"| accuracy_ci_95 | {m['accuracy_ci_95_lo']:.4f} - {m['accuracy_ci_95_hi']:.4f} |",
    ]
    with open(os.path.join(output_path, "eval_summary.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # CSV summary — one row per model, metrics as columns
    metric_names = [
        "answerability_precision", "answerability_recall", "answerability_f1",
        "unanswerability_precision", "unanswerability_recall", "unanswerability_f1",
        "accuracy", "weighted_f1", "mean_f1", "accuracy_std", "accuracy_ci_95_lo", "accuracy_ci_95_hi",
    ]
    with open(os.path.join(output_path, "eval_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + metric_names)
        w.writerow([model_display] + [f"{metrics[k]:.4f}" for k in metric_names])

    print(f"\nResults saved to: {output_path}/")
    print(f"  result_df.csv        — full predictions")
    print(f"  eval_summary.jsonl   — metrics (JSON)")
    print(f"  eval_summary.md      — metrics (Markdown)")
    print(f"  eval_summary.csv     — metrics (CSV table)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Answerability evaluation for Granite models (base or PEFT)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["base", "peft"],
        default="peft",
        help="Inference mode: "
             "'base' = base model only (no adapter, baseline), "
             "'peft' = base model + HF PEFT LoRA adapter",
    )

    # Base / PEFT mode arguments
    parser.add_argument(
        "--base-model", type=str, default="ibm-granite/granite-4.0-micro",
        help="[base/peft] Base model HF ID or local path",
    )
    parser.add_argument(
        "--adapter-path", type=str,
        default="ibm-granite/granite-lib-rag-r1.0",
        help="[peft] HF repo ID or local path containing the adapter. "
             "Use with --adapter-sub-path for nested adapters.",
    )
    parser.add_argument(
        "--adapter-sub-path", type=str,
        default="answerability/granite-4.0-micro/alora",
        help="[peft] Sub-directory within --adapter-path for the specific adapter",
    )

    # Shared HF arguments
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="[base/peft] Batch size for HF inference",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="[base/peft] Device: 'auto', 'cpu', 'mps', or 'cuda'",
    )

    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="[base/peft] Forward enable_thinking=True to the chat template so the "
             "model emits a <think>...</think> reasoning block (Granite 4.2). "
             "Default off. The prediction parser reads the label after </think> "
             "when the block is present, so scoring works either way.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=30,
        help="[base/peft] Max new tokens to generate per sample. Default 30 — "
             "enough for Granite 4.2's '<think></think>\"unanswerable\"' target "
             "(the old value of 6 truncated it, causing repeat-loops). Generation "
             "stops at the eos/<|im_end|> token regardless.",
    )

    # Shared arguments
    parser.add_argument(
        "--dataset-path", type=str, default="./scratch/answerability",
        help="Path to the test dataset directory",
    )
    parser.add_argument(
        "--dataset-size", type=int, default=100000,
        help="Maximum number of samples to load from each dataset JSONL file",
    )
    parser.add_argument(
        "--output-path", type=str, default="./scratch/results",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()
    if args.dataset_size < 1:
        parser.error("--dataset-size must be at least 1")

    print("=" * 80)
    print(f"Answerability Evaluation — {args.mode.upper()} mode")
    print("=" * 80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset size per file: {args.dataset_size}")
    print(f"Output:  {args.output_path}")

    if args.mode == "base":
        print(f"Base model: {args.base_model}")
        model_display = args.base_model.split("/")[-1] + " (base, no adapter)"
    else:  # peft
        # Resolve adapter path (handle HF repo with sub-path)
        from huggingface_hub import snapshot_download
        if os.path.isdir(args.adapter_path):
            adapter_local = os.path.join(args.adapter_path, args.adapter_sub_path)
        else:
            print(f"Downloading adapter from: {args.adapter_path}")
            repo_root = snapshot_download(
                args.adapter_path,
                allow_patterns=[f"{args.adapter_sub_path}/**"],
            )
            adapter_local = os.path.join(repo_root, args.adapter_sub_path)
        print(f"Base model:  {args.base_model}")
        print(f"Adapter:     {adapter_local}")
        model_display = args.base_model.split("/")[-1] + " + aLoRA"

    # Load data
    print("\n" + "=" * 80)
    print("Loading dataset")
    print("=" * 80)
    all_messages, all_documents, all_labels, collection_list, setting_list = \
        load_dataset(args.dataset_path, args.dataset_size)

    assert len(all_labels) == len(collection_list) == len(setting_list), (
        f"Sample count mismatch: {len(all_labels)} samples, "
        f"{len(collection_list)} collections, {len(setting_list)} settings"
    )

    # Inference
    print("\n" + "=" * 80)
    print("Running inference")
    print("=" * 80)
    if args.mode == "base":
        raw_predictions = run_inference_base(
            args.base_model,
            all_messages, all_documents,
            args.batch_size, args.device,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
        )
    else:  # peft
        raw_predictions = run_inference_peft(
            args.base_model, adapter_local,
            all_messages, all_documents,
            args.batch_size, args.device,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
        )

    # Process predictions
    predictions = process_predictions(raw_predictions)

    # Build results DataFrame
    result_df = pd.DataFrame({
        "collection": collection_list,
        "setting": setting_list,
        "ground_truth": all_labels,
        "raw_prediction": raw_predictions,
        "prediction": predictions,
    })

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    metrics, filtered_df = evaluate(result_df)

    # Save
    save_results(metrics, result_df, args.output_path, model_display)


if __name__ == "__main__":
    main()
