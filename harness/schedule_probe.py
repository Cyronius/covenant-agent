"""Family G's IR probe: what can Agent Core say about two lists?

  python -m harness.schedule_probe

Scheduling is the one family in the plan that is an IR question before it is
a data question (.claude/plans/archive/task-families.md §3 G). Rather than
assume, this hand-writes the canonical asks against
`runtime/worlds/scheduling` and puts each through the real parse -> typecheck
-> compile -> sandbox path, so R1's question - does the IR fit - is answered
by a run rather than by argument. Nothing here generates corpus.

The answer is narrower than "there is a missing primitive", and it took
writing the FOREACH forms out to see it: every one of these asks can be
*performed*. What the last one cannot do is produce the matching set as a
value. Spec 0.5.0 let a FILTER clause compare two fields of the element,
which is what "which shifts need cover" needed; "in that list" still has no
FILTER form, so that work happens inside a loop that acts on each element
and cannot be sorted, counted, or returned.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline import build  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from runtime.worlds import scheduling  # noqa: E402

WORLD = scheduling.WORLD

CONSTANTS = [
    {"type": "ID:person", "value": "person_1", "desc": "Ada Brooks"},
    {"type": "ID:person", "value": "person_2", "desc": "Femi Adler"},
    {"type": "BOOL", "value": True, "desc": "true"},
    {"type": "INT", "value": 8, "desc": "the headcount, 8"},
    {"type": "ID:slot", "value": "slot_person_1_3",
     "desc": "the hour both are free"},
]

FITS = "fits"
ACTION_ONLY = "runs, but only as an action - the set is never a value"

PROBES = [
    {
        "ask": "Book the cheapest free room that seats eight.",
        "program": """CALL @list_rooms -> r0
FILTER r0 @room.free EQ $2 AND NOT @room.seats LT $3 -> r1
SORT r1 @room.cost ASC -> r2
FIRST r2 -> r3
CALL @book_room r3.@room.id $4 -> r4
STOP
""",
        "verdict": FITS,
        "why": "one list and a predicate that fits FILTER's clause form, so "
               "FILTER + SORT + FIRST is the whole job. L4's shape with a "
               "two-clause predicate; nothing new needed.",
        "want": "books Mill, the cheapest free room that seats eight",
    },
    {
        "ask": "Which shifts still need cover?",
        "program": """CALL @list_shifts -> r0
FILTER r0 @shift.covered LT @shift.needs -> r1
RETURN r1
""",
        "verdict": FITS,
        "why": "comparing two fields of one record was legal only in an IF "
               "until spec 0.5.0 let a FILTER clause name a second field of "
               "the element on its right. Before that the under-covered "
               "shifts could be acted on one at a time inside a FOREACH and "
               "never counted, sorted or returned - and this ask is a "
               "question, so the value is the whole answer.",
        "want": "returns [shift_2], the one shift with covered below needs",
    },
    {
        "ask": "Take the earliest hour that is free in both calendars.",
        "program": """CALL @list_slots $1 -> r0
FILTER r0 @slot.free EQ $2 -> r1
MAP r1 @slot.start -> r2
CALL @list_slots $0 -> r3
FILTER r3 @slot.free EQ $2 -> r4
FOREACH r4 -> r5
  IF r2 CONTAINS r5.@slot.start
    CALL @take_slot r5.@slot.id -> r6
STOP
""",
        "verdict": ACTION_ONLY,
        "why": "MAP projects the other calendar to a list of times and "
               "CONTAINS does membership with the list on the left, which an "
               "IF admits. A FILTER clause cannot: its left is always the "
               "element's field, so the list would have to go on the right "
               "and no comparator puts it there. The intersection is walked, "
               "never bound - so `the earliest` is out of reach and this "
               "takes every mutually free hour instead of one.",
        "want": "takes 2 slots where the ask names 1",
    },
]


def main() -> None:
    ctx, sandbox_ctx = build_context(WORLD, CONSTANTS)
    print(f"scheduling probe - {len(PROBES)} asks, "
          f"{len(WORLD['tools'])} tools\n")
    fits = 0
    for probe in PROBES:
        print(f"ASK  {probe['ask']}")
        res = build(resolve(probe["program"], ctx), ctx)
        if not res.compile_ok:
            print("     STATIC  " + "; ".join(res.rendered_diagnostics()[:2]))
        else:
            sres = run_sandbox({
                "js": res.js, "state": scheduling.new_state(),
                "tools": sandbox_ctx["tools"], "fields": sandbox_ctx["fields"],
                "constants": sandbox_ctx["constants"], "now": WORLD["now"],
                "approval": True, "error_injection": [],
                "initial_registers": {}})
            calls = ", ".join(c["name"] for c in sres.get("calls", []))
            print(f"     RUNS    status={sres.get('status')} "
                  f"calls=[{calls}]")
            if "return_value" in sres:
                rv = sres["return_value"]
                ids = [e.get("id") for e in rv] if isinstance(rv, list) else rv
                print(f"             returned: {ids}")
            print(f"             expected: {probe['want']}")
        print(f"     IR      {probe['verdict']}")
        print(f"             {probe['why']}\n")
        fits += probe["verdict"] == FITS
    print(f"{fits}/{len(PROBES)} of the canonical asks fit the IR as it "
          f"stands. The other {len(PROBES) - fits} run, and do the wrong "
          f"thing.")
    print()
    print("What remains of the gap: an IF's condition is `operand cmp")
    print("operand`, so `IF r2 CONTAINS r5.F1` tests membership in a list.")
    print("A FILTER clause puts the element's field on the left, and no")
    print("comparator puts a list on the right, so the intersection of two")
    print("lists can be walked and never bound - which is what SORT, COUNT,")
    print("FIRST and RETURN all need. The other half, comparing two fields")
    print("of the element, landed in spec 0.5.0. The membership half is a")
    print("listed candidate: .claude/plans/ir-filter-predicate.md section 3b.")


if __name__ == "__main__":
    main()
