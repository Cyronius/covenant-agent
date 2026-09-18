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
    a run tag may precede the arm (out/s1_diff_s0_k8 from run_step1.sh).

    Step 2 adds two more fields to the convention, both optional so every older
    name still parses: `_L<n>` is the loop count the block was run at and
    `_dl<n>` is the number of distinct decoder layers in it. `_R<n>` marks a
    checkpoint trained at a loop count sampled from 1..n, which is the same
    weights evaluated at several L."""
    d: dict = {"arm": "diffusion" if re.search(r"(^|_)diff(_|$)", stem) else "ar",
               "seed": None, "steps": None, "repair": "repair" in stem,
               "split": "holdout" if "holdout" in stem else "test",
               "big": "big" in stem,
               # The weight format, the pointer head and the loop-index bias are
               # each a different model rather than a different setting of one,
               # so none of them ever shares a row with the baseline. Step 3
               # compares formats at matched resident bytes, so the format wins
               # the row when a name carries one.
               "variant": (re.search(r"_(fp|int8|u4|tern)(_|$)", stem).group(1)
                           if re.search(r"_(fp|int8|u4|tern)(_|$)", stem) else
                           "pointer" if "_ptr" in stem else
                           "loopemb" if re.search(r"_le(_|$)", stem) else "base")}
    m = re.search(r"_s(\d+)", stem)
    if m:
        d["seed"] = int(m.group(1))
    m = re.search(r"_k(\d+)", stem)
    if m:
        d["steps"] = int(m.group(1))
    m = re.search(r"_L(\d+)", stem)
    d["loops"] = int(m.group(1)) if m else 1
    m = re.search(r"_dl(\d+)", stem)
    d["layers"] = int(m.group(1)) if m else None
    m = re.search(r"_R(\d+)", stem)
    d["rand_loops"] = int(m.group(1)) if m else 0
    return d


def agg(vals: list[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1%}"
    return f"{mean(vals):.1%} +-{stdev(vals):.1%}"


def loop_report(rows: list[dict]) -> None:
    """Step 2: does goal success rise with the loop count at fixed parameters?

    The gate is a trend, not a threshold, so this prints the trend and says
    plainly whether it holds. Two lines are drawn, and confusing them would
    answer the wrong question:

      looped     one block of `dl` distinct layers applied L times. Parameters
                 are the same at every L; only compute grows. This is the line
                 the gate is about, and the one the NPU gets for free.
      unlooped   L=1 with more distinct layers. Compute grows the same way and
                 parameters grow with it. This is what looping has to come close
                 to for depth to be substituting for parameters (bet 2).
    """
    print()
    print("=== step 2: the loop sweep, goal against loop count ===")
    print()
    # The arm is part of the key: the two arms spend a loop differently (the
    # control applies the block per token, the diffusion arm per denoising step),
    # so averaging them together would hide both trends.
    looped = defaultdict(list)
    for r in rows:
        if r["rand_loops"]:
            continue
        looped[(r["arm"], r["layers"] or 1, r["loops"])].append(r)
    print(f"{'arm':10} {'block':>7} {'loops':>6} {'blocks/prog':>12} {'seeds':>6} "
          f"{'compile':>16} {'goal':>16}")
    for key in sorted(looped):
        arm, dl, L = key
        g = looped[key]
        print(f"{arm:10} {str(dl) + 'L':>7} {L:>6} {mean(r['blocks'] for r in g):12.0f} "
              f"{len(g):>6} {agg([r['compile'] for r in g]):>16} "
              f"{agg([r['goal'] for r in g]):>16}")

    # The gate: at a fixed arm and block, goal rises with L.
    for arm, dl in sorted({(k[0], k[1]) for k in looped}):
        series = sorted((L, mean(r["goal"] for r in looped[(arm, dl, L)]))
                        for (a2, d2, L) in looped if (a2, d2) == (arm, dl))
        if len(series) < 3:
            continue
        first, best = series[0], max(series, key=lambda p: p[1])
        spread = [stdev([r["goal"] for r in looped[(arm, dl, L)]])
                  for L, _ in series if len(looped[(arm, dl, L)]) > 1]
        noise = max(spread) if spread else 0.0
        rises = best[1] - first[1] > max(noise, 0.005)
        print()
        print(f"  {arm}, block of {dl} layer(s): goal {first[1]:.1%} at "
              f"L={first[0]} -> {best[1]:.1%} at L={best[0]}"
              + (f", seed spread up to {noise:.1%}" if spread else ""))
        print(f"  gate: goal {'RISES' if rises else 'does not rise'} with the "
              f"loop count at fixed parameters")

    rand = defaultdict(list)
    for r in rows:
        if r["rand_loops"]:
            rand[(r["rand_loops"], r["loops"])].append(r)
    if rand:
        print()
        print("  one checkpoint trained at a sampled loop count, run at each "
              "setting (the effort dial):")
        for (n, L) in sorted(rand):
            g = rand[(n, L)]
            print(f"    trained 1..{n:<3} run at L={L:<3} "
                  f"compile {agg([r['compile'] for r in g])} "
                  f"goal {agg([r['goal'] for r in g])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out",
                    help="one directory of scored runs, or several separated by "
                         "commas: seeds that ran in different sessions land in "
                         "different directories and belong in one table")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = []
    files = sorted(f for d in args.dir.split(",")
                   for f in Path(d.strip()).glob("*.score.json"))
    for f in files:
        stem = f.name.replace(".score.json", "")
        meta = parse_name(stem)
        s = json.loads(f.read_text(encoding="utf-8"))["summary"]
        n = max(s["n"], 1)
        # The scored file knows its own loop count and block depth; the name is
        # the fallback for files written before evaluate.py recorded them.
        loops = s.get("loops") or meta["loops"]
        layers = s.get("dec_layers") or meta["layers"]
        rows.append({**meta, "file": stem, "n": s["n"],
                     "compile": s["compile"] / n, "goal": s["goal"] / n,
                     "exact": s["exact"] / n, "passes": s["mean_passes"],
                     "loops": loops, "layers": layers,
                     # What the NPU actually spends: one forward pass applies
                     # the block layers x loops times, and on chip the loops are
                     # the free part. Reported so a win from looping is never
                     # confused with a win from more distinct weights.
                     "blocks": s["mean_passes"] * loops * (layers or 1),
                     "by_level": s.get("by_level", {})})

    want = [r for r in rows if r["split"] == args.split and not r["big"]]
    if not want:
        raise SystemExit(f"no scored runs for split={args.split} in {args.dir}")
    if len({r["file"] for r in rows}) != len(rows):
        raise SystemExit("the same run name appears in more than one directory; "
                         "a duplicate would count as an extra seed")

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

    loop_rows = [r for r in want if r["loops"] > 1 or r["rand_loops"]
                 or (r["layers"] or 0) not in (0, 4)]
    if loop_rows:
        loop_report(want)

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
