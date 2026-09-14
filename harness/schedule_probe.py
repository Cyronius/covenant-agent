"""Family G's IR probe: what can Agent Core say about two lists?

  python -m harness.schedule_probe

Scheduling is the one family in the plan that is an IR question before it is
a data question (.claude/plans/archive/task-families.md §3 G). Rather than
assume, this hand-writes the canonical asks against
`runtime/worlds/scheduling` and puts each through the real parse -> typecheck
-> compile -> sandbox path, so R1's question - does the IR fit - is answered
by a run rather than by argument. Nothing here generates corpus.

The answer was narrower than "there is a missing primitive", and it took
writing the FOREACH forms out to see it: every one of these asks could be
*performed*, and two of them could not produce the matching set as a value.
Spec 0.5.0 let a FILTER clause compare two fields of the element ("which
shifts need cover") and 0.6.0 gave it IN, membership with the list on the
right ("free in both calendars"), so all three now bind the set that SORT,
COUNT, FIRST and RETURN need.
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
     "desc": "the hour to book the room for"},
]

FITS = "fits"

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
FILTER r4 @slot.start IN r2 -> r5
SORT r5 @slot.start ASC -> r6
FIRST r6 -> r7
CALL @take_slot r7.@slot.id -> r8
STOP
""",
        "verdict": FITS,
        "why": "MAP projects the other calendar to a list of times and "
               "spec 0.6.0's IN takes that list on the right of a FILTER "
               "clause, so the intersection is a register and not a loop "
               "body. Until then the only membership form was CONTAINS in "
               "an IF, which can act on each mutually free hour and never "
               "sort them, so `the earliest` was out of reach and the "
               "program took every one.",
        "want": "takes 1 slot, the earliest hour free in both calendars",
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
          f"stands.")
    print()
    print("Both halves of the gap this probe found are now in the spec: a")
    print("FILTER clause may compare two fields of the element (0.5.0) and")
    print("may test a field for membership in a list with IN (0.6.0), which")
    print("also replaced CONTAINS's list arm - CONTAINS is substring again.")
    print("What is not measured here is adoption: whether a trained model")
    print("reaches for either form. That is the S5 corpus's question.")


if __name__ == "__main__":
    main()
