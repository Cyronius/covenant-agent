"""E-demo-requests: the free-typed kanban-ui requests people actually tried,
as a real eval suite (plan: .claude/plans/s2-consolidated-program.md §A3).

Each entry is a request exactly as typed, the demo board it was typed
against, and an authored reference. Contexts are built the same way the
demo builds them (server constants_from_board -> build_context), so the
prompt the model sees here is the prompt it saw in the browser. Eval-only;
never trained on.

  python -m harness.demo_suite            # writes data/holdout/e_demo_requests.jsonl
  python -m baselines.qwen.run_a --model ... --tasks data/holdout/e_demo_requests.jsonl --out ...

Add a case by appending to CASES with the request verbatim, a `cause` tag
(what failed and why, from the plan's diagnosis table), and a reference in
authoring form ($k = the k-th constant produced by constants_from_board —
run this module with --show to print the constants for every case).
Cases whose reference needs an IR feature that doesn't exist yet carry
`blocked_on` and are skipped, loudly, until it lands.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from dev_server import constants_from_board  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.taskbuild import ReferenceError, build_task  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

OUT = ROOT / "data" / "holdout" / "e_demo_requests.jsonl"

# client/kanban-ui/src/data/board.ts, verbatim (NOW matches the world's now).
NOW = 1_760_000_000
DAY = 86400
USERS = [("user_1", "Bob Alvarez"), ("user_2", "Priya Nandan"),
         ("user_3", "Theo Marsh"), ("user_4", "Kade Whitfield"),
         ("user_5", "Luz Ferreira")]
CARDS = [  # id, title, status, assignee, due (days from now), urgent, created
    ("card_1", "Fix OAuth redirect loop on staging", "doing", "user_1", -3, True, -30),
    ("card_2", "Write release notes for v4.2", "todo", "user_2", 6, False, -12),
    ("card_3", "Audit tool-call pause UX for DELETE effects", "doing", "user_3", 2, True, -18),
    ("card_4", "Retire legacy webhook handler", "todo", "user_1", -8, False, -40),
    ("card_5", "Design empty-state illustration for board", "todo", "user_2", 13, False, -9),
    ("card_6", "Reproduce race condition in segment resume", "doing", "user_4", 1, True, -15),
    ("card_7", "Onboard Luz to on-call rotation", "todo", "user_5", 8, False, -6),
    ("card_8", "Ship grammar-constrained decode benchmarks", "done", "user_4", -6, False, -35),
]


def demo_state() -> dict:
    return {"entities": {
        "user": [{"id": i, "name": n,
                  "email": f"{n.split()[0].lower()}@understory.test"}
                 for i, n in USERS],
        "card": [{"id": i, "title": t, "status": s, "assignee": a,
                  "due": NOW + d * DAY, "urgent": u, "archived": False,
                  "created": NOW + c * DAY}
                 for i, t, s, a, d, u, c in CARDS]},
        "outbox": [], "payments": []}


# (request, level, cause tag, reference segments | None, blocked_on,
#  expected_status — omitted means "ok")
# Constant indices: cards named by the request come first, then the five
# users (Bob=first user), true/false, todo/doing/done, five bank messages,
# then any literals extracted from the request (title, due), then the
# request itself as the writer brief.
CASES = [
    # --- 2026-09-01/02 owner-reported --------------------------------------
    ("delete all the cards owned by bob", 3, "cause:grammar-terminator (fixed)",
     ["CALL @list_cards -> r0\n"
      "FILTER r0 @card.assignee EQ $0 -> r1\n"
      "FOREACH r1 -> r2\n"
      "  CALL @delete_card r2.@card.id\n"
      "STOP"], None),
    ("move the release notes to doing column", 0, "cause:none (control, passed)",
     ["CALL @set_status $0 $9\nSTOP"], None),
    ("change the race condition issue to be owned by Bob", 0, "cause:tool-desc-grounding",
     ["CALL @assign_card $0 $1\nSTOP"], None),
    ("make Bob the assignee of the race condition issue", 0, "cause:tool-desc-grounding",
     ["CALL @assign_card $0 $1\nSTOP"], None),
    ("assign card 6 to Bob", 0, "cause:none (control, passed)",
     ["CALL @assign_card $0 $1\nSTOP"], None),
    ("list users", 1, "cause:corpus-shape (no RETURN tasks)",
     ["CALL @list_users -> r0\nRETURN r0"], None),
    ("list all the cards", 1, "cause:corpus-shape (no RETURN tasks)",
     ["CALL @list_cards -> r0\nRETURN r0"], None),
    ("archive everything that is done", 3, "cause:model-semantics (same-field AND)",
     ["CALL @list_cards -> r0\n"
      "FILTER r0 @card.status EQ $9 -> r1\n"
      "FOREACH r1 -> r2\n"
      "  CALL @archive_card r2.@card.id\n"
      "STOP"], None),
    ("message bob about card 4", 1, "cause:none (control; any bank message accepted only if it picks $11)",
     ["CALL @send_message $1 $11\nSTOP"], None),
    ("create a new issue for bob's performance review in done column", 2,
     "cause:serializer-extraction (no title/due constants)",
     # $15 = extracted title, $16 = default due (A2), $0 = Bob, $9 = done
     ["CALL @create_card $15 $16 $0 -> r0\n"
      "CALL @set_status r0.@card.id $9\n"
      "STOP"], None),
    ("tell bob which of his cards are overdue", 13, "cause:corpus-shape (writer tool; no prose tasks)",
     # $0 = Bob, $15 = the request verbatim (brief); no literals extracted
     ["CALL @list_cards -> r0\n"
      "FILTER r0 @card.assignee EQ $0 AND @card.due LT NOW -> r1\n"
      "CALL @write_text $15 r1 -> r2\n"
      "CALL @send_message $0 r2\n"
      "STOP"], None),
    ("duplicate the release notes issue", 12, "cause:corpus-shape (copy-from-fields; needs FORMAT)",
     # $0 = card 2, $16 = "Copy of {0}" template (offered because of "duplicate")
     ["CALL @get_card $0 -> r0\n"
      "FORMAT $16 r0.@card.title -> r1\n"
      "CALL @create_card r1 r0.@card.due r0.@card.assignee -> r2\n"
      "STOP"], None),
    ("duplicate the release notes issue and make cyrus the owner", 11,
     "cause:abstain (no such user)",
     ["ABORT NOT_FOUND"], None, "aborted"),
]


def build(show: bool, seeds: int) -> list:
    """Each case is emitted under `seeds` different T/F symbol assignments
    (task ids demo_<case>_s<seed>). The tuned planner's tool choice turned
    out to depend on the random numbering — the same request passed under
    one assignment and failed under another (2026-09-02) — so the suite
    reports a per-request pass rate, not a single pass/fail."""
    world = get_world("kanban")
    state = demo_state()
    tasks, skipped = [], 0
    for i, case in enumerate(CASES):
        request, level, cause, segments, blocked = case[:5]
        expected_status = case[5] if len(case) > 5 else "ok"
        constants = constants_from_board(request, state)
        if show:
            print(f"--- {request}")
            for k, c in enumerate(constants):
                print(f"  ${k} {c['type']} :: {c['desc']}")
        if blocked:
            print(f"SKIP (blocked on {blocked}): {request}")
            skipped += 1
            continue
        for seed in range(seeds):
            ctx, sandbox_ctx = build_context(
                world, constants, random.Random(1000 * (seed + 1) + i))
            resolved = [resolve(s, ctx) for s in segments]
            try:
                task = build_task(
                    task_id=f"demo_{i:02d}_s{seed}", level=level,
                    world_name="kanban", request=request,
                    constants=constants, segments=resolved, state=state,
                    expected_status=expected_status,
                    tags=["demo", cause, f"seed:{seed}"],
                    provenance={"source": "kanban-ui free-typed request",
                                "authoring": segments, "case": i},
                    prebuilt=(ctx, sandbox_ctx))
            except ReferenceError as e:
                print(f"REFERENCE FAILED: {e}")
                raise
            # run_a.py prompts from input_text (the generator adds it too)
            task["input_text"] = serialize_context(request, ctx)
            tasks.append(task)
    print(f"{len(tasks)} tasks built ({seeds} seeds per case), {skipped} cases skipped")
    return tasks


def report(results_path: str) -> None:
    """Per-request pass rate across seeds from a run_a.py results file."""
    import collections
    tasks = {t["id"]: t for t in map(json.loads, open(OUT))}
    hits = collections.defaultdict(lambda: [0, 0])
    for r in map(json.loads, open(results_path)):
        t = tasks.get(r["task_id"])
        if not t:
            continue
        h = hits[t["request"]]
        h[0] += bool(r["goal_success"])
        h[1] += 1
    for req, (p, n) in hits.items():
        print(f"{p}/{n}  {req}")
    tot = sum(p for p, _ in hits.values()); n = sum(n for _, n in hits.values())
    print(f"overall {tot}/{n} = {tot / n:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.demo_suite")
    ap.add_argument("--show", action="store_true",
                    help="print each case's constants (for authoring $k refs)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--report", metavar="RESULTS_JSONL",
                    help="print per-request pass rates from a run_a results file")
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
