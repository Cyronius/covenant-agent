"""The 42-unseen-world comparison, read off the .score.json files.

One table, because one question: does the model pick the right tool when the
type signature stops telling it which one that is?

Each trained model is scored on two halves of the same holdout -- the same
tasks, the same reference programs, differing only in whether every tool has
signature-identical siblings whose descriptions are the only separator
(harness/decoys.py). So the plain-to-decoy drop is attributable to the decoys
and nothing else: same model, same cache, same layout, same worlds.

The bar is `chance_tool_sig`, not `chance_tool`. A model that maps English to
a type shape and never reads a description still picks uniformly inside the
signature collision group, and on the decoyed half that scores 43.5%. Beating
the flat `chance_tool` (~2.4%) says nothing at all -- that was the defect
results/GROUNDING.md found, and reporting it this way is how it stays found.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def rows(out: Path) -> dict:
    found = defaultdict(dict)
    for p in sorted(out.glob("ho42_*.score.json")):
        name = p.name[len("ho42_"):-len(".score.json")]
        if "_holdout" not in name:
            continue
        half = "decoy" if name.endswith("_holdout_decoy") else "plain"
        run = name.split("_holdout")[0]
        found[run][half] = json.loads(p.read_text(encoding="utf-8"))["summary"]
    return found


def pct(x) -> str:
    return f"{x:6.1%}" if isinstance(x, (int, float)) else "     -"


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    found = rows(out)
    if not found:
        raise SystemExit(f"no ho42_*_holdout*.score.json under {out}")

    print(f"{'run':16s} {'half':6s} {'n':>5s} {'goal':>7s} {'same_tool':>10s} "
          f"{'shape-only':>11s} {'margin':>8s} {'decoy_called':>13s}")
    for run in sorted(found):
        for half in ("plain", "decoy"):
            s = found[run].get(half)
            if not s:
                continue
            c = s.get("call_agreement", {})
            compared = max(c.get("compared", 0), 1)
            same = c.get("same_tool", 0) / compared
            shape = c.get("chance_tool_sig", 0) / compared
            dec = (s["decoy_called"] / s["decoyed"]) if s.get("decoyed") else None
            print(f"{run:16s} {half:6s} {s['n']:5d} {pct(s['goal']/max(s['n'],1))} "
                  f"{pct(same)} {pct(shape)} {same - shape:+8.1%} {pct(dec)}")

    print()
    print("margin = same_tool - shape-only. On the decoy half this is the")
    print("grounding measurement: positive means the model read something the")
    print("signature does not carry. On the plain half shape-only is 100% by")
    print("construction, so a negative margin there is expected and means only")
    print("that the model is imperfect, not that it failed to read anything.")

    drops = []
    for run, halves in found.items():
        if {"plain", "decoy"} <= set(halves):
            p, d = halves["plain"], halves["decoy"]
            pc, dc = p.get("call_agreement", {}), d.get("call_agreement", {})
            ps = pc.get("same_tool", 0) / max(pc.get("compared", 1), 1)
            ds = dc.get("same_tool", 0) / max(dc.get("compared", 1), 1)
            drops.append((run, ps, ds))
    if drops:
        print()
        print("plain -> decoy, same model:")
        for run, ps, ds in sorted(drops):
            print(f"  {run:16s} {ps:6.1%} -> {ds:6.1%}   ({ds - ps:+.1%})")


if __name__ == "__main__":
    main()
