"""Turn a directory of scored runs into the comparison the experiment exists for.

The headline is not accuracy. It is accuracy against forward passes, because the
two arms spend compute differently: the control pays one pass per token it
emits, while the diffusion arm pays whatever step count it was given. Comparing
them at equal steps would flatter the diffusion arm; comparing at equal passes is
the honest axis.

    python evaluate.py --score --gen-out out/diff_s0_k8.jsonl --split test
    ... for each file ...
    python report.py --dir out
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_name(stem: str) -> dict:
    """out/diff_s0_k8_repair -> {arm: diffusion, seed: 0, steps: 8, repair: True};
    a run tag may precede the arm (out/s1_diff_s0_k8 from run_step1.sh)."""
    d: dict = {"arm": "diffusion" if re.search(r"(^|_)diff(_|$)", stem) else "ar",
               "seed": None, "steps": None, "repair": "repair" in stem,
               "split": "holdout" if "holdout" in stem else "test",
               "big": "big" in stem,
               # The pointer head is a different model, not a different setting
               # of the same one, so it never shares a row with the baseline.
               "variant": "pointer" if "_ptr" in stem else "base"}
    m = re.search(r"_s(\d+)", stem)
    if m:
        d["seed"] = int(m.group(1))
    m = re.search(r"_k(\d+)", stem)
    if m:
        d["steps"] = int(m.group(1))
    return d


def agg(vals: list[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1%}"
    return f"{mean(vals):.1%} +-{stdev(vals):.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.dir).glob("*.score.json")):
        stem = f.name.replace(".score.json", "")
        meta = parse_name(stem)
        s = json.loads(f.read_text(encoding="utf-8"))["summary"]
        n = max(s["n"], 1)
        rows.append({**meta, "file": stem, "n": s["n"],
                     "compile": s["compile"] / n, "goal": s["goal"] / n,
                     "exact": s["exact"] / n, "passes": s["mean_passes"],
                     "by_level": s.get("by_level", {})})

    want = [r for r in rows if r["split"] == args.split and not r["big"]]
    if not want:
        raise SystemExit(f"no scored runs for split={args.split} in {args.dir}")

    # Group across seeds so a single number never stands in for a distribution.
    groups: dict = defaultdict(list)
    for r in want:
        groups[(r["variant"], r["arm"], r["steps"], r["repair"])].append(r)

    print(f"=== {args.split}: accuracy against forward passes ===\n")
    print(f"{'model':9} {'arm':10} {'steps':>6} {'repair':>7} {'seeds':>6} {'passes':>8} "
          f"{'compile':>16} {'goal':>16}")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2] or 0, k[3])):
        var, arm, steps, rep = key
        g = groups[key]
        print(f"{var:9} {arm:10} {str(steps or '-'):>6} {'yes' if rep else '-':>7} "
              f"{len(g):>6} {mean(r['passes'] for r in g):8.1f} "
              f"{agg([r['compile'] for r in g]):>16} "
              f"{agg([r['goal'] for r in g]):>16}")

    ctrl = groups.get(("base", "ar", None, False), [])
    if ctrl:
        cp, cg = mean(r["passes"] for r in ctrl), mean(r["goal"] for r in ctrl)
        print(f"\nControl spends {cp:.1f} passes for {cg:.1%} goal success.")
        # The question the experiment asks: is there a step count at which the
        # diffusion arm matches the control for fewer passes?
        better = [(k, groups[k]) for k in groups
                  if k[1] == "diffusion"
                  and mean(r["goal"] for r in groups[k]) >= cg
                  and mean(r["passes"] for r in groups[k]) < cp]
        if better:
            k, g = min(better, key=lambda kv: mean(r["passes"] for r in kv[1]))
            p = mean(r["passes"] for r in g)
            print(f"{k[0]} diffusion matches or beats it at {k[2]} steps"
                  f"{' with repair' if k[3] else ''} for {p:.1f} passes "
                  f"({cp / p:.1f}x fewer).")
        else:
            best = max((k for k in groups if k[1] == "diffusion"),
                       key=lambda k: mean(r["goal"] for r in groups[k]), default=None)
            if best:
                bg = mean(r["goal"] for r in groups[best])
                print(f"Diffusion never matches the control. Its best is "
                      f"{bg:.1%} at {best[2]} steps ({best[0]}).")

    # What the compiler critic bought, at equal step count.
    for var, steps in sorted({(k[0], k[2]) for k in groups
                              if k[1] == "diffusion" and k[2]}):
        plain = groups.get((var, "diffusion", steps, False))
        rep = groups.get((var, "diffusion", steps, True))
        if plain and rep:
            pg, rg = mean(r["goal"] for r in plain), mean(r["goal"] for r in rep)
            pp, rp = mean(r["passes"] for r in plain), mean(r["passes"] for r in rep)
            print(f"\nCompiler in the loop at {steps} steps: goal "
                  f"{pg:.1%} -> {rg:.1%} ({rg - pg:+.1%}) for "
                  f"{pp:.1f} -> {rp:.1f} passes.")

    held = [r for r in rows if r["split"] == "holdout"]
    if held:
        print("\n=== held-out world (unseen tools; generalization) ===")
        for r in sorted(held, key=lambda r: (r["variant"], r["arm"])):
            print(f"  {r['variant']:9} {r['arm']:10} n={r['n']:4d} compile {r['compile']:6.1%} "
                  f"goal {r['goal']:6.1%}")

    big = [r for r in rows if r["big"]]
    if big:
        print("\n=== capacity check (d=512, 6+6 layers) ===")
        for r in sorted(big, key=lambda r: r["arm"]):
            print(f"  {r['arm']:10} compile {r['compile']:6.1%} goal {r['goal']:6.1%}")
        if len({r["arm"] for r in big}) == 2:
            bd = next(r["goal"] for r in big if r["arm"] == "diffusion")
            ba = next(r["goal"] for r in big if r["arm"] == "ar")
            sd = mean([r["goal"] for r in want if r["arm"] == "diffusion"] or [0])
            sa = mean([r["goal"] for r in want if r["arm"] == "ar"] or [0])
            flipped = (sd > sa) != (bd > ba)
            print(f"  ordering {'FLIPS' if flipped else 'holds'} at scale"
                  + (" -- the small result was a capacity artifact" if flipped else ""))


if __name__ == "__main__":
    main()
