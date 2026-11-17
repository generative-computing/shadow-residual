# SPDX-License-Identifier: Apache-2.0
"""Paired bootstrap of (arm - vanilla SR) per intrinsic headline metric, re-scored with the bench scorers.

1000 paired resamples of eval rows (same method as the shared-A study). Runs on a pod
that mounts danieloh-cos:

    python scripts/readout_paired_bootstrap.py readout-c256-g42
    python scripts/readout_paired_bootstrap.py readout-attn-c256-qo32-g42
"""
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/danieloh_cos/granite-switch/sr_shared_a/bench_harness_2506b7c")
from benchmarks.adapter_eval.scorers import ScoreContext, get_scorer  # noqa: E402

V = "/danieloh_cos/granite-switch/sr_shared_a/runs/v1"
R = "/danieloh_cos/granite-switch/sr_readout/runs/v1"
E = Path("/danieloh_cos/granite-switch/adapter-bench/granite-4.2-3b/v1/eval")
CFG = {
    "answerability": ("answerability", "accuracy", "ans"),
    "guardian_core": ("guardian", "accuracy", "guard"),
    "query_clarification": ("query_clarification", "overall_accuracy", "qc"),
    "requirement_check": ("requirements", "balanced_accuracy", "req"),
}
B = 1000


def main(arm: str) -> None:
    out = {}
    for t, (scorer, head, short) in CFG.items():
        van = sorted(glob.glob(f"{V}/vanilla-sr-qo-mlp-r32-c32-skv-g42-{short}/run_*/eval/{t}/predictions.jsonl"))
        arm_runs = sorted(glob.glob(f"{R}/{arm}-{short}/run_*/eval/{t}/predictions.jsonl"))
        if not van or not arm_runs:
            print(f"{t}: missing predictions (vanilla {len(van)}, {arm} {len(arm_runs)})", flush=True)
            continue
        rv = [json.loads(line) for line in open(van[-1])]
        ra = [json.loads(line) for line in open(arm_runs[-1])]
        assert len(rv) == len(ra), (t, len(rv), len(ra))
        for a, b in zip(rv, ra):
            assert a["messages"] == b["messages"]
        fn = get_scorer(scorer)
        ctx = ScoreContext(eval_dir=E / t, env={})

        def m(rows):
            return fn(rows, ctx).metrics[head]

        mv, ma = m(rv), m(ra)
        rng = random.Random(0)
        n = len(rv)
        diffs = []
        for _ in range(B):
            idx = [rng.randrange(n) for _ in range(n)]
            diffs.append(m([ra[i] for i in idx]) - m([rv[i] for i in idx]))
        diffs.sort()
        lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B) - 1]
        p = min(2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / B, 1.0)
        same = sum(a["generated_content"].strip() == b["generated_content"].strip() for a, b in zip(rv, ra))
        out[t] = dict(vanilla=mv, arm=ma, delta=ma - mv, ci95=[lo, hi], p_two_sided=p, n=n,
                      identical_outputs=same, vanilla_run=van[-1], arm_run=arm_runs[-1])
        print(f"BOOT {arm} {t:22s} vanilla {100*mv:6.2f} arm {100*ma:6.2f} delta {100*(ma-mv):+5.2f} "
              f"CI95 [{100*lo:+5.2f}, {100*hi:+5.2f}] p~{p:.3f} identical_outputs {same}/{n}", flush=True)
    json.dump(out, open(f"{R}/paired_bootstrap_{arm}_vs_vanilla.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
