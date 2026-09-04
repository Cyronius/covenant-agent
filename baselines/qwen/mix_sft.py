"""Mix SFT corpora with per-file repeat counts and a sample cap.

Used to build the S2R continuation set: the verified real-turn rows
(harness/real_train_build.py) upsampled against a replay slice of the S2
corpus, so a warm-started adapter learns the real request distribution
without forgetting the curriculum it already fits.

  python -m baselines.qwen.mix_sft --out data/sft_s2r.jsonl \
      data/sft_real_train.jsonl:4 data/sft_s2.jsonl:1:10800

Each spec is PATH[:REPEAT[:CAP]] — CAP samples that many rows (seeded)
before repeating. Rows are shuffled together; the summary prints the
realised share of each source so the mix is auditable.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="baselines.qwen.mix_sft")
    ap.add_argument("specs", nargs="+", metavar="PATH[:REPEAT[:CAP]]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool, summary = [], []
    for spec in args.specs:
        parts = spec.split(":")
        path, repeat = parts[0], int(parts[1]) if len(parts) > 1 else 1
        cap = int(parts[2]) if len(parts) > 2 else None
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        n_src = len(rows)
        if cap is not None and cap < len(rows):
            rows = rng.sample(rows, cap)
        rows = rows * repeat
        pool += rows
        summary.append((path, n_src, cap, repeat, len(rows)))

    rng.shuffle(pool)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")

    print(f"{len(pool)} rows -> {out}")
    for path, n_src, cap, repeat, n in summary:
        print(f"  {n:6d} ({n/len(pool):5.1%})  {path}  "
              f"[{n_src} rows{f', cap {cap}' if cap else ''}, x{repeat}]")


if __name__ == "__main__":
    main()
