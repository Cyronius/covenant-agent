"""What does "always write list -> FILTER -> act" score on its own? (plan 2a)

The demo suites read at 35-55% and two changes got them there — the conjunct
cap on `/db_prompt` (35 -> 55) and `static_repair` deleting the dead conjunct
(41 -> 48). Neither says how much of 55% is the *model*, because nobody has
scored the alternative: a planner with no comprehension at all that writes
the corpus's modal skeleton every time. 97.2% of training rows that call a
list tool then `FILTER` (`general-agent-plan.md` §2B), so that skeleton is
what the corpus teaches, and what it earns unaided is the null every demo
number should be read against — the same job `chance_tool_sig` does for
grounding (`results/R9.md` §3).

This planner reads the request for exactly one thing: which words it
contains, used to pick among the task's own declared symbols. It has no
model, no grammar, no reasoning. It writes:

    CALL <list tool>            -> r0     the entity the request names
    FILTER r0 <field> EQ <const> -> r1    the first type-compatible pair
    FOREACH r1 -> r2 / CALL <mutating tool> r2  |  RETURN r1
    STOP

Acting rather than returning is the modal choice (79% of training rows end
in `STOP`, 7.6% in `RETURN`), so it acts wherever the task's tools let it
and returns where they do not — which is every read-only world, the
Database Analyst demo included.

    python -m harness.reflex_baseline --tasks data/holdout/e_db_requests.jsonl
    python -m harness.reflex_baseline --tasks data/holdout/e_demo_requests.jsonl \
        --out results/logs/reflex_e_db_requests.jsonl

Read the number it prints as a floor under the demo scores, not as a model
result: if a trained model is near it, the suite is measuring the reflex.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ir import TaskContext, format_type  # noqa: E402
from harness.run import run_task  # noqa: E402

_WORD = re.compile(r"[a-z0-9]+")
_MIN_PART = 3


def _words(text: str) -> set:
    return set(_WORD.findall((text or "").casefold()))


def _mentions(text: str, words: set) -> bool:
    for part in _WORD.findall((text or "").casefold()):
        if len(part) < _MIN_PART:
            continue
        if any(part == w or part in w or w in part for w in words):
            return True
    return False


def _elem_entity(t) -> Optional[str]:
    """The entity of a `LIST OBJ:x` return, or None for anything else."""
    r = t.returns
    if not r or r[0] != "LIST":
        return None
    inner = r[1]
    return inner[1] if inner[0] in ("OBJ", "ID") else None


def _list_tool(ctx: TaskContext, words: set):
    """A tool returning a list of records: the one whose entity or name the
    request names, else the first declared."""
    cands = [t for t in ctx.tools.values() if _elem_entity(t)]
    if not cands:
        return None
    for t in cands:
        if _mentions(_elem_entity(t), words) or _mentions(t.name, words):
            return t
    return cands[0]


def _compatible(want, have) -> bool:
    if want[0] == "ID" and have[0] in ("ID", "OBJ"):
        return want[1] == have[1]
    if want[0] in ("INT", "FLOAT") and have[0] in ("INT", "FLOAT"):
        return True
    return format_type(want) == format_type(have)


def _filter_clause(ctx: TaskContext, entity: str, words: set):
    """`(field_sym, const_sym)` for the first clause that will typecheck,
    preferring a field the request names — the modal corpus clause is
    `<field> EQ <const>` and this picks the first pair that can be one."""
    fields = [f for f in ctx.fields.values() if f.entity == entity]
    consts = sorted(ctx.constants.values(), key=lambda c: c.sym)
    named = [f for f in fields if _mentions(f.name, words)]
    for pool in (named, fields):
        for f in pool:
            if f.name == "id":
                continue          # filtering a list by its own id is not the reflex
            for c in consts:
                if _compatible(f.type, c.type):
                    return f.sym, c.sym
    return None


def _action_tool(ctx: TaskContext, entity: str, words: set):
    """A mutating tool taking one required `ID:<entity>`: the modal action."""
    cands = []
    for t in ctx.tools.values():
        if "mutates" not in t.effects:
            continue
        req = [p for p in t.params if p.required]
        if len(req) != 1:
            continue
        p = req[0]
        if p.type[0] == "ID" and p.type[1] == entity:
            cands.append(t)
    if not cands:
        return None
    for t in cands:
        if _mentions(t.name, words):
            return t
    return cands[0]


def reflex_program(ctx: TaskContext, request: str) -> str:
    """The corpus's modal skeleton, bound to this task's own symbols."""
    words = _words(request)
    lt = _list_tool(ctx, words)
    if lt is None:
        return "ABORT UNSUPPORTED\n"       # nothing to list: the reflex has no move
    entity = _elem_entity(lt)
    lines = [f"CALL {lt.sym} -> r0"]
    src = "r0"
    clause = _filter_clause(ctx, entity, words)
    if clause:
        lines.append(f"FILTER r0 {clause[0]} EQ {clause[1]} -> r1")
        src = "r1"
    act = _action_tool(ctx, entity, words)
    if act is not None:
        lines.append(f"FOREACH {src} -> r2")
        lines.append(f"  CALL {act.sym} r2")
        lines.append("STOP")
    else:
        lines.append(f"RETURN {src}")
    return "\n".join(lines) + "\n"


def reflex_planner(task: dict):
    """A `harness.run.Planner` that writes the skeleton and nothing else."""
    def plan(request, ctx, seg_idx, registers):
        if seg_idx > 0:
            return None            # the reflex has no continuation
        return reflex_program(ctx, request)
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.reflex_baseline",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n", type=int, default=None, help="first N tasks only")
    ap.add_argument("--out", default=None, help="write the metric rows here")
    ap.add_argument("--show", type=int, default=0,
                    help="print the first N programs it writes")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")]
    if args.n:
        tasks = tasks[:args.n]
    rows = []
    for i, task in enumerate(tasks):
        if i < args.show:
            ctx = TaskContext.from_json(task["context"])
            print(f"--- {task['id']}  {task['request'][:70]}")
            print(reflex_program(ctx, task["request"]).rstrip())
        rows.append(run_task(task, reflex_planner(task)))

    n = len(rows) or 1
    def pct(key, among=None):
        pool = [r for r in rows if among is None or r.get(among) is not None]
        hit = sum(1 for r in pool if r.get(key))
        return f"{hit}/{len(pool)} {100 * hit / (len(pool) or 1):.1f}%"

    print(f"\nreflex null on {Path(args.tasks).name}, {len(rows)} tasks")
    print(f"  goal_success   {pct('goal_success')}")
    print(f"  return_match   {pct('return_match', among='return_match')}")
    print(f"  compile_ok     {pct('compile_ok')}")
    print(f"  filter_padded  {pct('filter_padded', among='filter_padded')}")
    print(f"  correct_abstain{pct('correct_abstain', among='correct_abstain')}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  rows -> {out}")


if __name__ == "__main__":
    main()
