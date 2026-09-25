"""Why does a planner's program miss the goal? For every failed task, the
first canvas slot where the generated program departs from the reference,
by kind: tool, field, keyword, comparison, constant, register, length. A
program that does not compile is classified by the same first departure
(its diagnostics are counted separately), because the compile error is
usually downstream of an earlier wrong choice. A wrong tool is split into a
look-alike (same signature as the reference's) and a tool of another
signature, which is almost always another entity's tool in the same
request.

  python fail_kinds.py --cache data_cache_ho42 \
      out/ho42_ar_s0_holdout out/ho42_ar_s1_holdout out/ho42_ar_s2_holdout

Each RUN is a prefix: RUN.jsonl (evaluate.py --generate) and RUN.score.json
(evaluate.py --score). --cache supplies the tasks' tool declarations.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

COMPARISONS = {"EQ", "NE", "LT", "LE", "GT", "GE", "AND", "OR", "NOT", "IN", "CONTAINS"}


def kind(tok: str) -> str:
    for k, pat in (("tool", r"T\d+"), ("field", r"F\d+"), ("register", r"r\d+"),
                   ("constant", r"[SINBDC]\d+")):
        if re.fullmatch(pat, tok):
            return k
    if tok in ("NL", "PAD"):
        return "length"
    return "comparison" if tok in COMPARISONS else "keyword"


def signature(t: dict) -> str:
    """Parameter types and return type: what a typechecker sees, and what a
    look-alike (a twin or an authored decoy) shares with the real tool."""
    return json.dumps([[p.get("type") for p in t.get("params") or []], t.get("returns")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = {r["id"]: r for r in pickle.load(open(Path(args.cache) / "rows.pkl", "rb"))[args.split]}

    n = goal = 0
    first, codes, kw_swaps = Counter(), Counter(), Counter()
    by_level = defaultdict(Counter)
    examples = defaultdict(list)
    for run in args.runs:
        gens = {r["task_id"]: r for r in map(json.loads, open(f"{run}.jsonl", encoding="utf-8"))}
        for s in json.load(open(f"{run}.score.json"))["rows"]:
            g = gens[s["task_id"]]
            lvl = by_level[s["level"]]
            n += 1
            lvl["n"] += 1
            if s["goal"]:
                goal += 1
                lvl["goal"] += 1
                continue
            a, b = g["canvas_tokens"], g["reference_tokens"]
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if i is None:
                cat = "same program, goal missed"
            else:
                ka, kb = kind(a[i]), kind(b[i])
                cat = kb if ka == kb else f"{kb} -> {ka}"
                if cat == "tool":
                    row = rows.get(s["task_id"].split("#")[0]) or rows.get(s["task_id"])
                    tools = {t["sym"]: t for t in row["context"]["tools"]} if row else {}
                    if a[i] in tools and b[i] in tools:
                        same = signature(tools[a[i]]) == signature(tools[b[i]])
                        cat = "tool: look-alike" if same else "tool: another signature"
                if cat == "keyword":
                    kw_swaps[f"{b[i]} -> {a[i]}"] += 1
            if not s["compile"]:
                for c in {str(d).split()[0] for d in s.get("diagnostics") or []}:
                    codes[c] += 1
            first[cat] += 1
            lvl[cat] += 1
            lvl["no compile"] += not s["compile"]
            if len(examples[cat]) < 3:
                req = (rows.get(s["task_id"].split("#")[0]) or {}).get("request", "")
                examples[cat].append({"task": s["task_id"], "request": req,
                                      "generated": g["program"], "reference": g["reference"]})

    fail = n - goal
    compile_fail = sum(c["no compile"] for c in by_level.values())
    print(f"{len(args.runs)} runs, {n} tasks: goal {goal / n:.1%}, failed {fail} "
          f"({fail / n:.1%}), of which {compile_fail} do not compile")
    print("\nfailed tasks by the first departure from the reference:")
    for c, k in first.most_common():
        print(f"  {k:5d}  {k / fail:6.1%} of failures  {k / n:5.1%} of tasks  {c}")
    print("\nkeyword swaps (reference -> generated):",
          ", ".join(f"{c} {k}" for c, k in kw_swaps.most_common(6)))
    print("compile diagnostics (tasks with at least one):",
          ", ".join(f"{c} {k}" for c, k in codes.most_common(6)))
    print("\nby level: goal, n, top departures")
    for lv in sorted(by_level):
        c = by_level[lv]
        top = [(k, v) for k, v in c.most_common() if k not in ("n", "goal", "no compile")][:3]
        print(f"  L{lv:<3d} {c['goal'] / c['n']:6.1%}  n={c['n']:4d}  "
              + ", ".join(f"{k} {v}" for k, v in top))
    if args.json:
        Path(args.json).write_text(json.dumps({
            "n": n, "goal": goal, "failed": fail, "no_compile": compile_fail,
            "first_departure": dict(first), "keyword_swaps": dict(kw_swaps),
            "diagnostics": dict(codes),
            "by_level": {str(k): dict(v) for k, v in sorted(by_level.items())},
            "examples": examples}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
