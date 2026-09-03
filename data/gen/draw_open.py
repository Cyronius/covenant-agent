"""B3 draw: sample the converted open-data pairs to the real-session shape
(results/S2.md §B1 sizing rule; plan lane-c-retrain §3).

Real traffic is write-heavy, multi-step, and a quarter abstains; the open
sets are mostly single reads. So the draw is by role, not by head:
  all usable hermes rows          (write-heavy, multi-call)
  N_ABSTAIN glaive refusals       (the only ABORT UNSUPPORTED source)
  N_WRITE   write/send/delete-effect rows from glaive + ToolACE
  N_READ    single reads, chosen for schema diversity (distinct tool-name
            sets round-robin, never the head of the file)

  python -m data.gen.draw_open --out data/open_pairs/b3_draw.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PAIRS = ROOT / "data" / "open_pairs"


def load(name_glob: str) -> list:
    rows = []
    for p in sorted(PAIRS.glob(name_glob)):
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows


def schema_key(row: dict) -> str:
    return "|".join(sorted(t["name"] for t in row["context"]["tools"]))


def diverse(rows: list, n: int, rng: random.Random) -> list:
    """Round-robin over distinct schemas so no API dominates."""
    by = collections.defaultdict(list)
    for r in rows:
        by[schema_key(r)].append(r)
    keys = list(by)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(by[k])
    out = []
    while len(out) < n and keys:
        for k in list(keys):
            if by[k]:
                out.append(by[k].pop())
                if len(out) >= n:
                    break
            else:
                keys.remove(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="data.gen.draw_open")
    ap.add_argument("--out", required=True)
    ap.add_argument("--abstain", type=int, default=400)
    ap.add_argument("--write", type=int, default=400)
    ap.add_argument("--read", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    hermes = load("hermes*.jsonl")
    glaive = load("glaive*.jsonl")
    toolace = load("toolace*.jsonl")
    seen = set()

    def dedupe(rows):
        out = []
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append(r)
        return out

    hermes = dedupe(hermes)
    glaive = dedupe(glaive)
    toolace = dedupe(toolace)
    is_write = lambda r: bool(set(r.get("effects") or []) & {"WRITE", "SEND", "DELETE", "PAY"})  # noqa: E731
    abstains = [r for r in glaive if r["expected_status"] == "aborted"]
    writes = [r for r in glaive + toolace if r["expected_status"] == "ok" and is_write(r)]
    reads = [r for r in glaive + toolace if r["expected_status"] == "ok" and not is_write(r)]

    draw = list(hermes)
    draw += diverse(abstains, args.abstain, rng)
    draw += diverse(writes, args.write, rng)
    draw += diverse(reads, args.read, rng)
    rng.shuffle(draw)
    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in draw:
            f.write(json.dumps(r) + "\n")
    by_src = collections.Counter(r["source"] for r in draw)
    n_multi = sum(1 for r in draw if len([l for l in r["reference"]["authoring"][0].splitlines() if l.startswith("CALL")]) >= 2)
    print(f"{len(draw)} pairs -> {out}: {dict(by_src)}; abstain {sum(r['expected_status']=='aborted' for r in draw)}, "
          f"write-effect {sum(is_write(r) for r in draw)}, multi-call {n_multi}; "
          f"pools: hermes {len(hermes)}, glaive abstains {len(abstains)}, writes {len(writes)}, reads {len(reads)}")


if __name__ == "__main__":
    main()
