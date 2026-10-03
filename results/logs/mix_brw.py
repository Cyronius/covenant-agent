"""The borrowed-worlds retrain mixes (.claude/plans/borrowed-worlds.md step 6).

Three 30,000-row corpora: R23's clt_train rows plus the new kinds at about
10%, 20% and 30%, because R23 showed a corpus change can cost the plain exam
11 points and the right share has to be measured, not guessed. The new rows
are drawn stratified by source so each mix keeps the same composition; the
30% mix takes all of them.

  python results/logs/mix_brw.py        # writes data/brw{10,20,30}_train.jsonl
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOTAL = 30000
BASE = ROOT / "data" / "clt_train.jsonl"
NEW = {
    "famb": ROOT / "data" / "famb_episodes.jsonl",       # family A + C, briefs
    "wsb": ROOT / "data" / "wsb_episodes.jsonl",         # workshop crafting
    "rmb": ROOT / "data" / "rmb_episodes.jsonl",         # BabyAI-style rooms
    "svr": ROOT / "data" / "service_retail.jsonl",       # tau2 retail
    "sva": ROOT / "data" / "service_airline.jsonl",      # tau2 airline
}
SHARES = {"brw10": 3000, "brw20": 6000, "brw30": None}   # None: every new row


def lines(path: Path) -> list:
    return [ln for ln in path.open(encoding="utf-8") if ln.strip()]


def main() -> None:
    base = lines(BASE)
    pools = {k: lines(p) for k, p in NEW.items()}
    n_new = sum(len(v) for v in pools.values())
    print(f"base {len(base)} rows; new {n_new} rows: "
          + ", ".join(f"{k} {len(v)}" for k, v in pools.items()))
    for name, want in SHARES.items():
        rng = random.Random(20260929)
        want = n_new if want is None else want
        picked = []
        for k, pool in pools.items():
            take = round(want * len(pool) / n_new)
            picked += rng.sample(pool, min(take, len(pool)))
        rest = rng.sample(base, TOTAL - len(picked))
        rows = picked + rest
        rng.shuffle(rows)
        out = ROOT / "data" / f"{name}_train.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(r if r.endswith("\n") else r + "\n" for r in rows)
        print(f"{out.name}: {len(rows)} rows, {len(picked)} new "
              f"({len(picked) / len(rows):.1%})")


if __name__ == "__main__":
    main()
