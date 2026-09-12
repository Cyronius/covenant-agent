"""Failure-mode breakdown for E-rpg runs, one line per arm plus the detail
RPG.md's analysis reads: which tools actually executed, what the typechecker
rejected, and how often a required slot got NULL.

  python results/logs/rpg_breakdown.py results/logs/*_e_rpg.jsonl
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

CALL = re.compile(r"^\s*CALL (T\d+)((?: [^\s]+)*)", re.MULTILINE)


def load(path):
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def arm(path: Path) -> None:
    rows = load(path)
    if not rows:
        return
    n = len(rows)
    turns = [t for r in rows for t in r["turn_log"]]
    calls = [c for t in turns for c in t.get("calls", [])]
    diag = collections.Counter()
    for t in turns:
        for d in t.get("diagnostics") or []:
            diag[d.split(" line")[0][:40]] += 1
    tools = collections.Counter(c["name"] for c in calls)
    bad = collections.Counter(c["name"] for c in calls if not c["ok"])
    nulls = sum(1 for c in calls if any(a is None for a in (c.get("args") or [])))
    shapes = collections.Counter(
        len(CALL.findall(t["program"])) for t in turns)
    blocks = collections.Counter()
    for t in turns:
        for kw in ("IF", "FOREACH", "TRY", "PARALLEL", "ABORT"):
            if any(ln.strip().startswith(kw) for ln in t["program"].splitlines()):
                blocks[kw] += 1

    print(f"\n== {path.name}")
    print(f"   model {rows[0].get('model')}  condition {rows[0].get('condition')}")
    print(f"   won {sum(r['won'] for r in rows)}/{n}  died {sum(r['dead'] for r in rows)}/{n}"
          f"  key {sum(r['key_taken'] for r in rows)}/{n}"
          f"  door {sum(r['door_opened'] for r in rows)}/{n}")
    print(f"   turns {len(turns)}  did not compile {sum(1 for t in turns if t['status'] == 'static_error')}"
          f"  runtime error {sum(1 for t in turns if t['status'] == 'error')}"
          f"  ok {sum(1 for t in turns if t['status'] == 'ok')}")
    print(f"   calls {len(calls)} ({len(calls) - sum(c['ok'] for c in calls)} failed)"
          f"  NULL args {nulls}")
    print(f"   tools executed {dict(tools)}")
    if bad:
        print(f"   failed by tool  {dict(bad)}")
    if diag:
        print(f"   diagnostics     {dict(diag)}")
    print(f"   CALLs per turn  {dict(sorted(shapes.items()))}")
    print(f"   block use       {dict(blocks)}")
    gen = [t["gen_ms"] for t in turns if t.get("gen_ms")]
    if gen:
        print(f"   gen_ms mean {sum(gen) / len(gen):.0f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        arm(Path(p))
