"""Goal success on one shared set of task ids, across R13's S6-trained planners
and the three R14 runs (.claude/plans/s5-first-s6-cost-split.md).

The S5-trained runs were scored on S6's decoyed and flip exams through an
evaluation-only cache that leaves out the 19 rows with more than 54 tools, so
each "first 2,000" differs slightly; this restricts every file to the ids all
of them contain. Plain-exam rows are split into whole tasks and paused-task
segments, since S5's training rows held only 110 segments.

    python results/logs/s6split_matched.py      # from the repo root
"""
import json
from pathlib import Path

RUNS = Path("models/tiny/runs")
FILES = {
    "plain": {
        "A0 on S6 (R13)": "pod_planner/out2/s6_A0_holdout_plain_mask",
        "SPt on S6 (R13)": "pod_planner/out2/s6_SPt_holdout_plain_mask",
        "A0 on S6, no names (R13)": "pod_diag/out/diag_s6nonames_plain",
        "A0 on S5, old settings (R13)": "pod_diag/out/diag_s5fix_s6plain",
        "run 1: A0 on S5": "pod_s6split/out/s5g_A0_s6plain",
        "run 2: SPt on S5": "pod_s6split/out/s5g_SPt_s6plain",
        "run 3: A0 on S6, options off": "pod_s6split/out/s6off_A0_plain",
    },
    "decoy": {
        "A0 on S6 (R13)": "pod_planner/out2/s6_A0_holdout_decoy_mask",
        "SPt on S6 (R13)": "pod_planner/out2/s6_SPt_holdout_decoy_mask",
        "A0 on S6, no names (R13)": "pod_diag/out/diag_s6nonames_decoy",
        "run 1: A0 on S5": "pod_s6split/out/s5g_A0_s6decoy",
        "run 2: SPt on S5": "pod_s6split/out/s5g_SPt_s6decoy",
        "run 3: A0 on S6, options off": "pod_s6split/out/s6off_A0_decoy",
    },
    "flip": {
        "A0 on S6 (R13)": "pod_planner/out2/s6_A0_holdout_flip_mask",
        "SPt on S6 (R13)": "pod_planner/out2/s6_SPt_holdout_flip_mask",
        "A0 on S6, no names (R13)": "pod_diag/out/diag_s6nonames_flip",
        "run 1: A0 on S5": "pod_s6split/out/s5g_A0_s6flip",
        "run 2: SPt on S5": "pod_s6split/out/s5g_SPt_s6flip",
        "run 3: A0 on S6, options off": "pod_s6split/out/s6off_A0_flip",
    },
}


def rows(prefix: str) -> dict:
    d = json.loads((RUNS / f"{prefix}.score.json").read_text(encoding="utf-8"))
    out = {}
    for r in d["rows"]:
        tid = r.get("task_id") or r.get("id")
        # the S5-cache runs carry S6 plain ids suffixed +s6
        out[tid.replace("+s6", "")] = bool(r.get("goal"))
    return out


def pct(v: list) -> str:
    return f"{100 * sum(v) / len(v):5.1f}" if v else "   --"


def main() -> None:
    report = {}
    for half, files in FILES.items():
        got = {name: rows(p) for name, p in files.items()}
        shared = set.intersection(*(set(g) for g in got.values()))
        whole = sorted(t for t in shared if "#s" not in t)
        seg = sorted(t for t in shared if "#s" in t)
        print(f"\n{half}: {len(shared)} shared ids ({len(whole)} whole tasks, {len(seg)} paused segments)")
        print(f"  {'model':32s}  all    whole  segments")
        report[half] = {"n": len(shared), "n_whole": len(whole), "n_seg": len(seg), "goal": {}}
        for name, g in got.items():
            a = [g[t] for t in sorted(shared)]
            w = [g[t] for t in whole]
            s = [g[t] for t in seg]
            print(f"  {name:32s}  {pct(a)}  {pct(w)}  {pct(s)}")
            report[half]["goal"][name] = {"all": sum(a) / len(a), "whole": sum(w) / max(1, len(w)),
                                         "segments": sum(s) / max(1, len(s))}
    dest = Path("results/logs/s6split/matched.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
