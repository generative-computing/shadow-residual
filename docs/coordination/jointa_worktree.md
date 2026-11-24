# Joint shared-A training study (session 69cd3514) — works in a worktree

Started 2026-11-22. Goal: jointly train one shared LoRA A (r=32) so that all 4
intrinsics beat an fp32-LoRA vanilla SR baseline.

- All code for this study lives in the git worktree `scratch/wt-jointa` (branch
  `jointa-exp`, based on 5558a47). The shared checkout is NOT edited by this study.
- The untied-A edits (`--untie-a`, `tie_a=`) in the shared `train.py` /
  `multitask.py` / `test_multitask.py` belong to session 516bee84. They are kept
  as is in the shared tree and are also copied into the worktree.
- COS root: `/danieloh_cos/granite-switch/sr_joint_a/`. Vela job prefix: `antonp-ja-`.
- Briefing: `docs/autopilot/sr-joint-a/index.html`.

Planned additions in the worktree (not in the shared tree):
`model.lora_dtype` / `--lora-dtype fp32` (fp32 LoRA master weights on a bf16 base),
`--mt-sampler blocks`, `--mt-skip-absent`, `--mt-a-lr-scale`, `--mt-a-freeze-frac`,
`--mt-init-a`, and `training/a_movement.py`.
