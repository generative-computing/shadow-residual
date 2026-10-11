# SPDX-License-Identifier: Apache-2.0
"""Convert training-shaped rows to eval format for held-out generation.

Training rows embed the gold answer as the final assistant turn (ground_truth is
None). Generating on them leaks the answer (the model copies it). This strips the
final assistant turn into `ground_truth` and removes it from `messages`, so the
generator sees a prompt WITHOUT the answer.

Usage:
    python scripts/strip_completion.py <in.jsonl> <out.jsonl>
Reports how many rows were stripped. Rows whose last message isn't an assistant
turn are passed through unchanged (with any existing ground_truth kept).
"""
import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: strip_completion.py <in.jsonl> <out.jsonl>")
        return 2
    inp, out = sys.argv[1], sys.argv[2]
    n = w = 0
    with open(inp) as f, open(out, "w") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            r = json.loads(line)
            m = r.get("messages", [])
            if m and m[-1].get("role") == "assistant":
                r["ground_truth"] = m[-1].get("content")
                r["messages"] = m[:-1]
                w += 1
            o.write(json.dumps(r) + "\n")
    print(f"stripped {w}/{n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
