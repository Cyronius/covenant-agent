"""Tables for results/R27.md from score_staged.sh's outputs
(.claude/plans/staged-decoder-experts.md steps 0b and 2).

  python results/logs/staged_summary.py            # markdown to stdout

Reads models/tiny/runs/pod_staged/out/*.score.json (sandbox scores),
results/logs/staged/*_cal.json (thresholds) and results/logs/staged/*.log
(play.py exams). A missing file prints as a dash, so it runs on a partial set.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models/tiny/runs/pod_staged/out"
LOGS = ROOT / "results/logs/staged"
sys.path.insert(0, str(ROOT / "models/tiny"))
from staged import task_type  # noqa: E402

ARMS = ["clt_base", "d3_base", "d3_ctrl6", "d3_draft", "d3_DR", "d3_D2R", "d3_DRx", "d3_DRxf", "d3_DRc", "d3_D2Rc", "d3_ctrl6p", "d3_DRp", "d3_D2Rp"]
STAGED = ("d3_DR", "d3_D2R", "d3_DRx", "d3_DRxf", "d3_DRc", "d3_D2Rc", "d3_DRp", "d3_D2Rp")
ROUNDS = {"d3_DRc": 4, "d3_D2Rc": 8}   # committing drafts: one read-out per round
SEEDS = ["s0", "s1"]
HALVES = ["plain", "decoy", "flip", "clut", "test"]
EXAMS = ["e_brief_boatyard", "e_brief_rooms_after", "e_brief_house", "e_brief_coursebuilder",
         "e_service_telecom", "e_demo_requests"]
LOADS = {"clt_base": (2, 2), "d3_base": (2, 2), "d3_ctrl6": (2, 2), "d3_draft": (2, 2),
         "d3_DR": (2, 3), "d3_D2R": (3, 4), "d3_DRx": (2, 3), "d3_DRxf": (2, 3),
         "d3_DRc": (2, 3), "d3_D2Rc": (3, 4), "d3_ctrl6p": (2, 2), "d3_DRp": (2, 3),
         "d3_D2Rp": (3, 4)}   # weight loads per program: (exited, refined)


def score(path: Path):
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return 100 * s["goal"] / max(s["n"], 1)


def goals(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["task_id"]: bool(r.get("goal")) for r in
            json.loads(path.read_text(encoding="utf-8"))["rows"] if "goal" in r}


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{sum(vals) / len(vals):.1f} ({min(vals):.1f}-{max(vals):.1f})"


def table_halves(draft: bool = False):
    suf = ".draft.score.json" if draft else ".score.json"
    print("| arm | " + " | ".join(HALVES) + " |")
    print("|---|" + "---|" * len(HALVES))
    for arm in ARMS:
        if draft and arm not in STAGED:
            continue
        cells = [fmt([score(OUT / f"{arm}_{s}_{h}{suf}") for s in SEEDS]) for h in HALVES]
        print(f"| {arm}{' (draft readout)' if draft else ''} | " + " | ".join(cells) + " |")


def per_type(arm: str, seed: str, suf: str = ".score.json") -> dict:
    gen = OUT / f"{arm}_{seed}_test.jsonl"
    g = goals(OUT / f"{arm}_{seed}_test{suf}")
    if not gen.exists() or not g:
        return {}
    acc = defaultdict(lambda: [0, 0])
    for line in gen.open(encoding="utf-8"):
        r = json.loads(line)
        if r["task_id"] in g:
            t = task_type(r["world"])
            acc[t][0] += g[r["task_id"]]
            acc[t][1] += 1
    return {t: 100 * a / n for t, (a, n) in acc.items()}


def table_types():
    types = ["data", "obsact", "pages"]
    print("| arm | " + " | ".join(types) + " |")
    print("|---|---|---|---|")
    for arm in ARMS:
        if arm == "clt_base":
            continue
        pts = [per_type(arm, s) for s in SEEDS]
        print(f"| {arm} | " + " | ".join(fmt([p.get(t) for p in pts]) for t in types) + " |")


def table_exit():
    print("| run | threshold | exit share (val) | val goal: refined / draft / at threshold | "
          "plain at threshold (exit share) | loads per program |")
    print("|---|---|---|---|---|---|")
    for arm in STAGED:
        for s in SEEDS:
            cal = LOGS / f"{arm}_{s}_cal.json"
            if not cal.exists():
                continue
            c = json.loads(cal.read_text(encoding="utf-8"))
            v = c["val"]
            at = LOGS / f"{arm}_{s}_at_thr.log"
            plain = "-"
            if at.exists():
                m = re.search(r"_plain\.jsonl: n=\d+ exit ([\d.]+)%\s+goal ([\d.]+)%", at.read_text())
                if m:
                    plain = f"{m.group(2)} ({m.group(1)}%)"
            lo, hi = LOADS[arm]
            loads = lo * v["exit_share"] + hi * (1 - v["exit_share"])
            thr = "never" if c["thr"] is None else f"{c['thr']:.3f}"
            print(f"| {arm}_{s} | {thr} | {100 * v['exit_share']:.0f}% | "
                  f"{100 * v['goal_refined']:.1f} / {100 * v['goal_draft']:.1f} / {100 * v['goal']:.1f} | "
                  f"{plain} | {loads:.2f} |")


def exam_goal(path: Path):
    if not path.exists():
        return None
    m = re.search(r"(\d+) tasks: goal (\d+)/", path.read_text(encoding="utf-8", errors="replace"))
    return (int(m.group(2)), int(m.group(1))) if m else None


def table_exams():
    short = [e.replace("e_brief_", "").replace("e_service_", "").replace("e_", "") for e in EXAMS]
    print("| arm (mode) | " + " | ".join(short) + " |")
    print("|---|" + "---|" * len(EXAMS))
    for arm in ARMS:
        modes = ["never"] + (["thr"] if arm in STAGED else [])
        if arm == "d3_draft":
            modes = ["always"]
        for mode in modes:
            cells = []
            for ex in EXAMS:
                vals = [exam_goal(LOGS / f"{ex}_{arm}_{s}_{mode}.log") for s in SEEDS]
                vals = [v for v in vals if v]
                cells.append(" / ".join(f"{a}" for a, _ in vals) + (f" of {vals[0][1]}" if vals else "") or "-")
            print(f"| {arm} ({mode}) | " + " | ".join(cells) + " |")


def table_speed():
    """R28 point 9: layer runs that must happen one after another, per
    program, from the real program lengths (`passes`), with the early exit
    at each run's threshold. Left to right: layers x slot steps. Staged: the
    draft's 16 passes (4 layers x 4 loops; D2R's 2x4 + 2x4 too) and, for a
    committing draft, one output-head pass per round; plus 2 refiner layers
    per slot step when the draft doesn't exit."""
    import statistics as st
    print("| run | split | goal % | steps one after another: mean / median | layers of weights moved | exits |")
    print("|---|---|---|---|---|---|")
    for split in ("plain", "test"):
        for arm in ("d3_base", "d3_ctrl6", "d3_ctrl6p") + STAGED:
            for s in SEEDS:
                gen = OUT / f"{arm}_{s}_{split}.jsonl"
                g = goals(OUT / f"{arm}_{s}_{split}.score.json")
                staged = arm in STAGED
                cal = LOGS / f"{arm}_{s}_cal.json"
                if not gen.exists() or not g or staged and not cal.exists():
                    continue
                if staged:
                    thr = json.loads(cal.read_text(encoding="utf-8"))["thr"]
                    gd = goals(OUT / f"{arm}_{s}_{split}.draft.score.json")
                steps, weights, ok, ex = [], [], 0, 0
                for line in gen.open(encoding="utf-8"):
                    r = json.loads(line)
                    p = int(r["passes"])
                    if not staged:
                        n = 4 if arm == "d3_base" else 6     # ctrl6 and ctrl6p
                        steps.append(n * p)
                        weights.append(n)
                        ok += g.get(r["task_id"], False)
                        continue
                    e = thr is not None and float(r["draft_conf"]) >= thr
                    ex += e
                    draft = 16 + ROUNDS.get(arm, 0)
                    steps.append(draft if e else draft + 2 * p)
                    weights.append(4 if e else 6)
                    ok += (gd if e else g).get(r["task_id"], False)
                print(f"| {arm}_{s} | {split} | {100 * ok / len(steps):.1f} | {st.mean(steps):.0f} / "
                      f"{st.median(steps):.0f} | {st.mean(weights):.1f} | {100 * ex / len(steps):.0f}% |")


def table_copy():
    """R28 point 3: how often the refiner leaves its draft unchanged, and
    what it does when it changes it."""
    print("| run | split | draft kept unchanged | fixes | breaks |")
    print("|---|---|---|---|---|")
    for arm in STAGED:
        for s in SEEDS:
            for split in ("test", "plain"):
                gen = OUT / f"{arm}_{s}_{split}.jsonl"
                ref = goals(OUT / f"{arm}_{s}_{split}.score.json")
                dra = goals(OUT / f"{arm}_{s}_{split}.draft.score.json")
                if not gen.exists() or not ref or not dra:
                    continue
                rows = [json.loads(line) for line in gen.open(encoding="utf-8")]
                kept = sum(r.get("draft_program") == r.get("program") for r in rows)
                fix = sum(1 for k in ref if ref[k] and not dra.get(k, False))
                brk = sum(1 for k in ref if dra.get(k) and not ref[k])
                print(f"| {arm}_{s} | {split} | {100 * kept / len(rows):.0f}% | {fix} | {brk} |")


def main():
    print("## Holdout halves and test, goal % (mean of seeds, range)\n")
    table_halves()
    print()
    table_halves(draft=True)
    print("\n## Test split by task type, goal %\n")
    table_types()
    print("\n## Early exit\n")
    table_exit()
    print("\n## Held-out exams per turn (play.py, backoff 3), goals per seed\n")
    table_exams()
    print("\n## Speed (modelled), with the early exit\n")
    table_speed()
    print("\n## Refiner against its own draft\n")
    table_copy()


if __name__ == "__main__":
    main()
