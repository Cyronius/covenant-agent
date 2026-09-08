"""Per-task table across the reactive arms (react_qwen38_27b.sh).

  python results/logs/compare_react.py [results/logs] [qwen38-27b]

Reads <dir>/<tag>_<arm>_<slice>.jsonl for arm in k0/prompt/react and the
three smoke slices; prints one line per task with each arm's outcome, the
pauses and error turns the reactive arms took, and the arm totals. `-`
marks a task an arm has not run yet (the script works on a partial run).
"""
import json
import sys
from pathlib import Path

ARMS = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("k0", "prompt", "react")
SLICES = ("_smoke8b_known", "_smoke8b_demo", "_smoke8b_crowded")


def load(path):
    rows = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["task_id"]] = r
    return rows


def mark(r):
    if r is None:
        return "-"
    if r["goal_success"]:
        return "PASS"
    return {"static_error": "static", "error": "runtime", "aborted": "abort",
            "ok": "wrong", "segment_limit": "limit",
            "planner_exhausted": "exhaust"}.get(r["status"], r["status"])


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/logs")
    tag = sys.argv[2] if len(sys.argv) > 2 else "qwen38-27b"
    totals = {a: [0, 0] for a in ARMS}
    A0, A1, A2 = ARMS
    print(f"{'task':34} {A0:8} {A1:8} {A2:8} pauses({A1[0]}/{A2[0]}) err_turns  {A2} diag")
    for sl in SLICES:
        arms = {a: load(d / f"{tag}_{a}_{sl}.jsonl") for a in ARMS}
        ids = list(arms[A0]) or list(arms[A1]) or list(arms[A2])
        print(f"--- {sl}")
        for tid in ids:
            rs = {a: arms[a].get(tid) for a in ARMS}
            for a in ARMS:
                if rs[a] is not None:
                    totals[a][1] += 1
                    totals[a][0] += bool(rs[a]["goal_success"])
            pp = rs[A1]["pauses"] if rs[A1] else "-"
            rp = rs[A2]["pauses"] if rs[A2] else "-"
            et = rs[A2].get("error_turns", 0) if rs[A2] else "-"
            diag = ""
            if rs[A2] and not rs[A2]["goal_success"]:
                diag = "; ".join(rs[A2]["diagnostics"][-2:])[:70]
            flag = ""
            if rs[A1] and rs[A2] and rs[A1]["goal_success"] != rs[A2]["goal_success"]:
                flag = f" <== {A2} flips" if rs[A2]["goal_success"] else f" <== {A2} breaks"
            print(f"{tid:34} {mark(rs[A0]):8} {mark(rs[A1]):8} "
                  f"{mark(rs[A2]):8} {str(pp)+'/'+str(rp):11} {str(et):9}  {diag}{flag}")
    print("---")
    for a in ARMS:
        ok, n = totals[a]
        print(f"{a:8} {ok}/{n}")


if __name__ == "__main__":
    main()
