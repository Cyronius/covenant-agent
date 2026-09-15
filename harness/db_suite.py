"""E-db-requests: the Database Analyst demo's free-typed questions, as a real
eval suite. The `crm` sibling of `harness/demo_suite.py`, and the reason it
exists: `data/holdout/e_demo_requests.jsonl` is 70/70 kanban, so the db demo
(client/app/src/worlds/db) shipped with no suite behind it at all.

That gap bites twice over here. The Analyst is a *reading* demo — it asks
questions and shows rows — and `goal_success` is state equality
(harness/metrics.py), which a read can never fail. So even the kanban suite's
read cases were unfalsifiable. Score these on `return_match`, which compares
the rows the program actually returned against the reference's.

Contexts are built the way the demo builds them (server constants_from_crm ->
build_context), so the prompt here is the prompt the browser sends.
Eval-only; never trained on.

  python -m harness.db_suite              # writes data/holdout/e_db_requests.jsonl
  python -m harness.db_suite --show       # constants per case, for authoring $k
  python -m baselines.qwen.run_a --model ... --tasks data/holdout/e_db_requests.jsonl --out ...
  python -m harness.db_suite --report results/logs/<run>.jsonl

Add a case by appending to CASES with the question verbatim and a reference
in authoring form ($k = the k-th constant produced by constants_from_crm —
the indices shift when the question names a customer, so run --show).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from dev_server import constants_from_crm  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.taskbuild import ReferenceError, build_task  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

OUT = ROOT / "data" / "holdout" / "e_db_requests.jsonl"


def db_state() -> dict:
    """The board the demo starts from: /db_new hands the client whatever
    crm.py's default_state is, so the suite takes it from the same place
    instead of mirroring it (the kanban suite has to mirror; this one
    doesn't)."""
    return copy.deepcopy(get_world("crm")["default_state"])


# (question, level, cause tag, reference segments, blocked_on,
#  expected_status - omitted means "ok")
#
# Constant indices assume the question names no customer: $3 true, $4 false,
# $5-$7 customer.plan, $8 open, $9 closed, $10-$12 priority. A question that
# names a customer inserts it at $0 and shifts the rest by one.
CASES = [
    # --- reads: what the Analyst demo actually does --------------------
    ("what customers do we have", 1, "read:bare-list",
     ["CALL @list_customers -> r0\nRETURN r0"], None),
    ("list the staff", 1, "read:bare-list",
     ["CALL @list_staff -> r0\nRETURN r0"], None),
    ("show me every invoice", 1, "read:bare-list",
     ["CALL @list_invoices -> r0\nRETURN r0"], None),
    ("which customers are delinquent", 1, "read:one-predicate",
     ["CALL @list_customers -> r0\n"
      "FILTER r0 @customer.delinquent EQ $3 -> r1\n"
      "RETURN r1"], None),
    ("show me the open tickets", 1, "read:one-predicate",
     ["CALL @list_tickets -> r0\n"
      "FILTER r0 @ticket.status EQ $8 -> r1\n"
      "RETURN r1"], None),
    ("which invoices are unpaid", 1, "read:one-predicate",
     ["CALL @list_invoices -> r0\n"
      "FILTER r0 @invoice.paid EQ $4 -> r1\n"
      "RETURN r1"], None),
    ("who are our enterprise customers", 1, "read:enum-predicate",
     ["CALL @list_customers -> r0\n"
      "FILTER r0 @customer.plan EQ $7 -> r1\n"
      "RETURN r1"], None),
    ("how many tickets are open", 2, "read:count",
     ["CALL @list_tickets -> r0\n"
      "FILTER r0 @ticket.status EQ $8 -> r1\n"
      "COUNT r1 -> r2\n"
      "RETURN r2"], None),
    ("what is our biggest invoice", 2, "read:sort-first",
     ["CALL @list_invoices -> r0\n"
      "SORT r0 @invoice.amount DESC -> r1\n"
      "FIRST r1 -> r2\n"
      "RETURN r2"], None),
    ("which customer has the most tickets", 3, "read:most",
     ["CALL @list_tickets -> r0\n"
      "MOST r0 @ticket.customer -> r1\n"
      "RETURN r1"], None),
    ("show me the overdue invoices", 3, "read:two-predicates",
     ["CALL @list_invoices -> r0\n"
      "FILTER r0 @invoice.due LT NOW AND NOT @invoice.paid EQ $3 -> r1\n"
      "RETURN r1"], None),
    ("which customers does Alice manage", 1, "read:id-predicate",
     ["CALL @list_customers -> r0\n"
      "FILTER r0 @customer.manager EQ $0 -> r1\n"
      "RETURN r1"], None),
    # --- writes: the same board, the acting half ----------------------
    ("mark Globex as delinquent", 0, "write:named-target",
     # $0 = Globex (the question names it), $4 = true
     ["CALL @set_delinquent $0 $4\nSTOP"], None),
    ("close all the open tickets", 3, "write:filter-foreach",
     ["CALL @list_tickets -> r0\n"
      "FILTER r0 @ticket.status EQ $8 -> r1\n"
      "FOREACH r1 -> r2\n"
      "  CALL @close_ticket r2.@ticket.id\n"
      "STOP"], None),
    # --- abstain ------------------------------------------------------
    ("mark Hooli as delinquent", 11, "abstain:no-such-customer",
     ["ABORT NOT_FOUND"], None, "aborted"),
]


def build(show: bool, seeds: int) -> list:
    """Each case under `seeds` symbol assignments (db_<case>_s<seed>), for the
    same reason demo_suite.py does it: the tuned planner's tool choice depends
    on the random T/F numbering, so a per-question pass rate is the honest
    unit, not a single pass/fail."""
    world = get_world("crm")
    state = db_state()
    tasks, skipped = [], 0
    for i, case in enumerate(CASES):
        question, level, cause, segments, blocked = case[:5]
        expected_status = case[5] if len(case) > 5 else "ok"
        constants = constants_from_crm(question, state, world["now"])
        if show:
            print(f"--- {question}")
            for k, c in enumerate(constants):
                print(f"  ${k} {c['type']} :: {c['desc']}")
        if blocked:
            print(f"SKIP (blocked on {blocked}): {question}")
            skipped += 1
            continue
        for seed in range(seeds):
            ctx, sandbox_ctx = build_context(
                world, constants, random.Random(1000 * (seed + 1) + i))
            resolved = [resolve(s, ctx) for s in segments]
            try:
                task = build_task(
                    task_id=f"db_{i:02d}_s{seed}", level=level,
                    world_name="crm", request=question,
                    constants=constants, segments=resolved, state=state,
                    expected_status=expected_status,
                    tags=["db-demo", cause, f"seed:{seed}"],
                    provenance={"source": "db-ui free-typed question",
                                "authoring": segments, "case": i},
                    prebuilt=(ctx, sandbox_ctx))
            except ReferenceError as e:
                print(f"REFERENCE FAILED on {question!r}: {e}")
                raise
            task["input_text"] = serialize_context(question, ctx)
            tasks.append(task)
    print(f"{len(tasks)} tasks built ({seeds} seeds per case), {skipped} cases skipped")
    return tasks


def report(results_path: str) -> None:
    """Per-question pass rate. Reads are reported on return_match (the rows
    that came back); writes and abstains on goal_success (the state). Scoring
    a read on goal_success is what hid the db demo's failures."""
    import collections
    tasks = {t["id"]: t for t in map(json.loads, open(OUT))}
    hits = collections.defaultdict(lambda: [0, 0, ""])
    for r in map(json.loads, open(results_path)):
        t = tasks.get(r["task_id"])
        if not t:
            continue
        read_only = r.get("return_match") is not None
        h = hits[t["request"]]
        h[0] += bool(r["return_match"] if read_only else r["goal_success"])
        h[1] += 1
        h[2] = "return" if read_only else "state"
    for req, (p, n, how) in hits.items():
        print(f"{p}/{n}  [{how}]  {req}")
    tot = sum(p for p, _, _ in hits.values())
    n = sum(n for _, n, _ in hits.values())
    print(f"overall {tot}/{n} = {tot / n:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.db_suite")
    ap.add_argument("--show", action="store_true",
                    help="print each case's constants (for authoring $k refs)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--report", metavar="RESULTS_JSONL",
                    help="print per-question pass rates from a run_a results file")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return
    tasks = build(args.show, args.seeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
