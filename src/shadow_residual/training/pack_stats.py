# SPDX-License-Identifier: Apache-2.0
"""Batch-size tuning for padding-free packed training (experimental).

Simulates greedy packing at various batch sizes and reports fill rate,
skip rate, throughput, and unique-example coverage per epoch.

Usage as CLI::

    python -m shadow_residual.training.pack_stats \\
        --dataset_path /path/to/hf_dataset \\
        --max_seq_length 8192 \\
        --activation_sequence 100264 78191 100265

Usage as library::

    from shadow_residual.training.pack_stats import recommend_batch_size

    bs = recommend_batch_size(lengths, max_seq_length=8192)
"""

import argparse
import random
from dataclasses import dataclass


@dataclass
class PackStats:
    """Packing statistics for a given batch size."""

    batch_size: int
    fill_pct: float
    skip_pct: float
    steps_per_epoch: int
    tokens_per_step: int
    tokens_per_epoch: int
    unique_examples_per_epoch: int


def simulate_packing(
    lengths: list[int],
    max_seq_length: int,
    batch_size: int,
    seed: int = 42,
) -> PackStats:
    """Simulate one epoch of greedy packing.

    Args:
        lengths: Sequence lengths (after control token insertion).
        max_seq_length: Maximum packed sequence length.
        batch_size: Number of examples the DataLoader sends per step.
        seed: Random seed for shuffling.

    Returns:
        PackStats with fill rate, skip rate, throughput, etc.
    """
    shuffled = lengths[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total_packed_tokens = 0
    total_used = 0
    total_skipped = 0
    num_packs = 0

    for batch_start in range(0, len(shuffled), batch_size):
        batch = shuffled[batch_start:batch_start + batch_size]

        current_len = 0
        for l in batch:
            el = min(l, max_seq_length)
            if current_len + el > max_seq_length:
                total_skipped += 1
            else:
                current_len += el
                total_used += 1

        total_packed_tokens += current_len
        num_packs += 1

    fill_pct = total_packed_tokens / (num_packs * max_seq_length) * 100
    skip_pct = total_skipped / len(shuffled) * 100
    tokens_per_step = total_packed_tokens // num_packs

    return PackStats(
        batch_size=batch_size,
        fill_pct=fill_pct,
        skip_pct=skip_pct,
        steps_per_epoch=num_packs,
        tokens_per_step=tokens_per_step,
        tokens_per_epoch=total_packed_tokens,
        unique_examples_per_epoch=total_used,
    )


def recommend_batch_size(
    lengths: list[int],
    max_seq_length: int,
    min_fill_pct: float = 90.0,
    max_skip_pct: float = 50.0,
    candidates: list[int] | None = None,
    seed: int = 42,
) -> tuple[int, list[PackStats]]:
    """Recommend a batch size for packed training.

    Picks the smallest batch size that achieves ``min_fill_pct`` fill rate
    while staying under ``max_skip_pct`` skip rate.  If no candidate meets
    both criteria, picks the one with the best fill rate under the skip
    constraint.

    Args:
        lengths: Sequence lengths (after control token insertion).
        max_seq_length: Maximum packed sequence length.
        min_fill_pct: Target minimum fill rate (default 90%).
        max_skip_pct: Maximum acceptable skip rate (default 50%).
        candidates: Batch sizes to evaluate (default: powers of 2 + extras).
        seed: Random seed for shuffling.

    Returns:
        (recommended_batch_size, all_stats) — the recommended batch size
        and the full list of PackStats for all candidates.
    """
    if candidates is None:
        candidates = [1, 4, 8, 12, 16, 24, 32, 48, 64]

    all_stats = []
    for bs in candidates:
        stats = simulate_packing(lengths, max_seq_length, bs, seed=seed)
        all_stats.append(stats)

    qualifying = [
        s for s in all_stats
        if s.fill_pct >= min_fill_pct and s.skip_pct <= max_skip_pct
    ]

    if qualifying:
        best = min(qualifying, key=lambda s: s.batch_size)
    else:
        under_skip = [s for s in all_stats if s.skip_pct <= max_skip_pct]
        if under_skip:
            best = max(under_skip, key=lambda s: s.fill_pct)
        else:
            best = min(all_stats, key=lambda s: s.skip_pct)

    return best.batch_size, all_stats


def format_stats_table(
    all_stats: list[PackStats],
    recommended_bs: int,
    num_examples: int,
) -> str:
    """Format packing stats as an ASCII table."""
    lines = []
    header = (
        f"{'':>3s}  {'bs':>4s}  {'fill%':>6s}  {'skip%':>6s}  "
        f"{'steps/ep':>9s}  {'tok/step':>9s}  "
        f"{'tok/epoch':>12s}  {'examples/ep':>12s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for s in all_stats:
        marker = "-->" if s.batch_size == recommended_bs else "   "
        lines.append(
            f"{marker}  {s.batch_size:4d}  {s.fill_pct:5.1f}%  {s.skip_pct:5.1f}%  "
            f"{s.steps_per_epoch:9,}  {s.tokens_per_step:9,}  "
            f"{s.tokens_per_epoch:12,}  "
            f"{s.unique_examples_per_epoch:,}/{num_examples:,}"
        )

    return "\n".join(lines)


def analyze_dataset(
    lengths: list[int],
    max_seq_length: int,
    min_fill_pct: float = 90.0,
    max_skip_pct: float = 50.0,
) -> int:
    """Analyze a dataset and print batch-size recommendation.

    Returns:
        Recommended batch size.
    """
    import statistics

    num_examples = len(lengths)
    print(f"Dataset: {num_examples:,} examples")
    print(
        f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
        f"mean={statistics.mean(lengths):.0f}, median={statistics.median(lengths):.0f}"
    )
    print(f"max_seq_length: {max_seq_length:,}")
    truncated = sum(1 for l in lengths if l > max_seq_length)
    if truncated:
        print(f"WARNING: {truncated:,} examples ({100*truncated/num_examples:.1f}%) exceed max_seq_length and will be truncated")
    print()

    recommended_bs, all_stats = recommend_batch_size(
        lengths, max_seq_length,
        min_fill_pct=min_fill_pct,
        max_skip_pct=max_skip_pct,
    )

    table = format_stats_table(all_stats, recommended_bs, num_examples)
    print(table)
    print()
    print(
        f"Recommended: --per_device_train_batch_size {recommended_bs}  "
        f"(fill={next(s.fill_pct for s in all_stats if s.batch_size == recommended_bs):.1f}%, "
        f"skip={next(s.skip_pct for s in all_stats if s.batch_size == recommended_bs):.1f}%)"
    )

    return recommended_bs


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tune batch size for padding-free packed training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m shadow_residual.training.pack_stats \\\n"
            "      --dataset_path ./my_hf_dataset \\\n"
            "      --max_seq_length 8192 \\\n"
            "      --activation_sequence 100264 78191 100265"
        ),
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--activation_sequence", type=int, nargs="+", default=None)
    parser.add_argument("--min_fill_pct", type=float, default=90.0)
    parser.add_argument("--max_skip_pct", type=float, default=50.0)
    args = parser.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.dataset_path)

    lengths = [len(x) for x in ds["input_ids"]]

    if args.activation_sequence is not None:
        lengths = [l + 1 for l in lengths]

    analyze_dataset(
        lengths,
        max_seq_length=args.max_seq_length,
        min_fill_pct=args.min_fill_pct,
        max_skip_pct=args.max_skip_pct,
    )


if __name__ == "__main__":
    main()
