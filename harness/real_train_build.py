"""Build a *training* set from the real-session pool (plan
.claude/plans/lane-c-retrain.md; companion to harness/real_suite_build.py).

The S2 retrain scored route_write_match 0.6% and correct_abstain 0/73 on
e_real_sessions while scoring ~97% goal on its own synthetic holdout: the
model learned the curriculum, not the request distribution. Real turns are
phrased nothing like the generator's ("what can you help with?", "can you
make this look better?"), and no synthetic abort example looks like a real
no-call turn, so the model never learned the trigger.

This builds SFT-able tasks from `data/real_sessions/b2_pool.jsonl` — the
eligible turns that are NOT in the frozen eval slice — pairing each real
request with a *verified* reference program:

  abstain turns (frontier called nothing)  -> ABORT <reason>
  routing turns (frontier called tools)    -> CALLs to the mapped world
                                              tools, args bound by type

Every synthesized program is executed through the same harness the eval
uses and kept only if it actually scores on the metric it is meant to
teach (route_write_match for routing rows, correct_abstain for abstains).
A target that would not score is a target worth dropping, so the emitted
set is correct by construction rather than by inspection.

Leakage: the pool and the eval slice share no turn id, but a handful of
short requests appear verbatim in both ("Create a summary of key
takeaways"). Those texts are dropped — otherwise the eval measures recall
of a memorized string.

  python -m harness.real_train_build --out data/real_train_tasks.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from harness.real_suite import READ_TOOLS, expected_tools, score_row  # noqa: E402
from harness.real_suite_build import build_rows  # noqa: E402
from harness.run import run_task  # noqa: E402

POOL = ROOT / "data" / "real_sessions" / "b2_pool.jsonl"
CANDIDATES = ROOT / "data" / "real_sessions" / "eval_candidates.jsonl"

# The scorer counts any reason except NOT_FOUND as a correct abstain, but a
# constant reason would teach a degenerate mapping, so pick by what the
# request actually fails to supply.
_META = re.compile(r"\b(what|which|how)\b.*\b(can|could|do)\s+you\b|"
                   r"\byour (capabilities|features)\b|\bhelp with\b", re.I)
_VAGUE = re.compile(r"\b(better|nicer|prettier|improve|enhance|fix|clean\s*up|"
                    r"polish|review|rate|thoughts?|what do you think)\b", re.I)


def abort_reason(request: str) -> str:
    """UNSUPPORTED: asks about the agent, not the course. NEEDS_INFO: names
    an action with no object. AMBIGUOUS: an instruction with no criterion."""
    if _META.search(request):
        return "UNSUPPORTED"
    if _VAGUE.search(request):
        return "AMBIGUOUS"
    return "NEEDS_INFO"


def _bind(param: dict, consts: list, used_str: set) -> str | None:
    """Pick a constant symbol satisfying one parameter's declared type."""
    ty = param["type"]
    exact = [c for c in consts if c["type"] == ty]
    if ty == "STR":
        # prefer a request-derived literal not already spent; the last STR
        # constant is the request verbatim (the writer brief) and is the
        # right default for content params.
        fresh = [c for c in exact if c["sym"] not in used_str]
        pick = (fresh[-1] if fresh else (exact[-1] if exact else None))
        if pick:
            used_str.add(pick["sym"])
        return pick["sym"] if pick else None
    return exact[0]["sym"] if exact else None


def synthesize(task: dict) -> str | None:
    """A program calling the mapped tools, arguments bound by type."""
    ctx = task["context"]
    by_name = {t["name"]: t for t in ctx["tools"]}
    consts = ctx["constants"]
    lines, reg, used_str = [], 0, set()
    for name in task["expected_tools"]:
        tool = by_name.get(name)
        if tool is None:
            return None
        args = []
        for p in tool["params"]:
            if not p.get("required"):
                continue
            sym = _bind(p, consts, used_str)
            if sym is None:
                return None            # no constant of that type: unteachable
            args.append(sym)
        call = f"CALL {tool['sym']}" + ("".join(f" {a}" for a in args))
        if tool.get("returns"):
            call += f" -> r{reg}"
            reg += 1
        lines.append(call)
    if not lines:
        return None
    lines.append("STOP")
    return "\n".join(lines)


def verify(task: dict, program: str) -> tuple[bool, dict]:
    """Run the candidate target and check it scores what it should teach."""
    probe = dict(task)
    probe["reference"] = {"segments": [program]}
    row = run_task(probe, lambda req, ctx, i, regs: program if i == 0 else None)
    row["task_id"] = task["id"]
    s = score_row(row, task)
    if task["expected_status"] == "aborted":
        return bool(s["correct_abstain"]), row
    return bool(s["route_write_match"]), row


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.real_train_build")
    ap.add_argument("--out", default="data/real_train_tasks.jsonl")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(POOL, encoding="utf-8")]
    cand = [json.loads(l) for l in open(CANDIDATES, encoding="utf-8")]
    eval_turns = {c["turn_id"] for c in cand}
    eval_text = {c["request"].strip() for c in cand}

    keep, drops = [], collections.Counter()
    for r in pool:
        if r.get("scrub_flags"):
            drops["not strict-clean"] += 1
        elif r["turn_id"] in eval_turns:
            drops["eval turn id"] += 1
        elif r["request"].strip() in eval_text:
            drops["request text also in eval"] += 1
        else:
            r = dict(r)
            r["frontier_tool_calls"] = [c["name"] if isinstance(c, dict) else c
                                        for c in r["frontier_tool_calls"] if c]
            keep.append(r)
    if args.limit:
        keep = keep[:args.limit]

    tasks = build_rows(keep, args.seed)
    out_rows = []
    for task in tasks:
        if task["expected_status"] == "aborted":
            program = f"ABORT {abort_reason(task['request'])}"
        else:
            if not task["expected_tools"]:
                drops["no mapped tool"] += 1
                continue
            program = synthesize(task)
            if program is None:
                drops["no type-valid binding"] += 1
                continue
        ok, _ = verify(task, program)
        if not ok:
            drops["target did not score"] += 1
            continue
        task["reference"] = {"segments": [program]}
        task["tags"] = [t for t in task["tags"] if t != "unmapped"] + ["real_train"]
        out_rows.append(task)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    kinds = collections.Counter(r["routing_primary"] for r in out_rows)
    print(f"pool {len(pool)} -> eligible {len(keep)} -> verified {len(out_rows)} -> {out}")
    print("  kept by routing:", dict(sorted(kinds.items())))
    print("  abstains:", sum(r["expected_status"] == "aborted" for r in out_rows))
    for k, v in drops.most_common():
        print(f"  dropped {v:5d}  {k}")


if __name__ == "__main__":
    main()
