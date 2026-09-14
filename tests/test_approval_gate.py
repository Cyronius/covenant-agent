"""What a click on the approval gate actually authorizes.

Two demo bugs, one gate. WRITE wasn't gated at all, so "assign all issues to
cyrus" put four cards on the wrong person with nothing to click
(.claude/plans/host-preflight-and-bulk-gate.md §B). And the DELETE gate
halted at the first destructive call while the approval token it handed back
covered the whole program: one click on "delete_card #2" deleted seven cards
(.claude/plans/preview-run-approval-gate.md). A freeform run now runs to the
end unapproved and reports everything it would do at once.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import dev_server  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context  # noqa: E402
from harness.demo_suite import demo_state  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from runtime.worlds import get_world  # noqa: E402


def run(request: str, program: str, approval=None, state=None):
    world = get_world("kanban")
    state = state or demo_state()
    constants = dev_server.constants_from_board(request, state, world["now"])
    ctx, _ = build_context(world, constants, random.Random(7),
                           symbols="typed", enums=True)
    req = {"context": ctx.to_json(), "world": "kanban", "now": world["now"],
           "text": resolve(program, ctx), "state": state,
           "registers": None, "pause_types": None}
    if approval is not None:
        req["approval"] = approval
    return dev_server.handle_validate(req)


ASSIGN_ALL = ("CALL @list_cards -> r0\n"
              "FOREACH r0 -> r1\n"
              "  CALL @assign_card r1.@card.id $0\n"
              "STOP\n")
ASSIGN_TWO = ("CALL @assign_card $0 $2\n"
              "CALL @assign_card $1 $2\n"
              "STOP\n")
# Verbatim from the demo, 2026-09-14: "let's delete the oldest issue". The
# model invented a status filter; the gate previewed one of the deletions.
DELETE_TODO = ("CALL @list_cards -> r0\n"
               "FILTER r0 @card.status EQ $7 -> r1\n"
               "FOREACH r1 -> r2\n"
               "  CALL @delete_card r2\n"
               "STOP\n")


def test_a_board_wide_assignment_is_blocked_with_the_full_list():
    res = run("assign everything to bob", ASSIGN_ALL)
    assert res["status"] == "effect_blocked"
    err = res["error"]
    assert err["code"] == "BULK_WRITE" and err["effect"] == "WRITE"
    assert err["count"] == len(demo_state()["entities"]["card"])
    # the list is exact: the run finished, so every write it would make is here
    assert [c["name"] for c in err["calls"]] == ["assign_card"] * err["count"]
    assert [c["args"][0] for c in err["calls"]] == \
        [c["id"] for c in demo_state()["entities"]["card"]]


def test_the_blocked_run_changes_nothing_and_approval_applies_it():
    blocked = run("assign everything to bob", ASSIGN_ALL)
    assert blocked["status"] == "effect_blocked"

    ok = run("assign everything to bob", ASSIGN_ALL, approval=True)
    assert ok["status"] == "ok"
    assert {c["assignee"] for c in ok["final_state"]["entities"]["card"]} == {"user_1"}


def test_two_writes_are_not_worth_a_click():
    res = run("assign card 1 and card 2 to bob", ASSIGN_TWO)
    assert res["status"] == "ok"
    cards = {c["id"]: c for c in res["final_state"]["entities"]["card"]}
    assert cards["card_1"]["assignee"] == "user_1"
    assert cards["card_2"]["assignee"] == "user_1"


def test_the_gate_is_opt_in_per_run():
    """Only the demo's freeform path asks for it. A payload without
    `preview`/`bulk_write_limit` — which is every payload harness/run.py
    builds — runs the same program straight through, so no eval number
    moves."""
    world = get_world("kanban")
    state = demo_state()
    constants = dev_server.constants_from_board("assign everything to bob",
                                                state, world["now"])
    ctx, sandbox_ctx = build_context(world, constants, random.Random(7),
                                     symbols="typed", enums=True)
    res = run_sandbox({"js": build(resolve(ASSIGN_ALL, ctx), ctx).js,
                       "state": state, "now": world["now"], "approval": False,
                       **sandbox_ctx})
    assert res["status"] == "ok"
    assert len(res["calls"]) == len(state["entities"]["card"]) + 1  # + list_cards


def test_the_dungeon_is_not_gated():
    """The RPG shares the freeform /validate path, and every one of its
    tools is a WRITE — moving is one. A three-step turn is an ordinary
    turn, not a bulk edit."""
    state = dev_server.handle_rpg_new({})["state"]
    prompt = dev_server.handle_rpg_prompt({"state": state})
    move = next(t["sym"] for t in prompt["context"]["tools"] if t["name"] == "move")
    res = dev_server.handle_validate({
        "context": prompt["context"], "world": "rpg", "now": prompt["now"],
        "text": f"CALL {move} S0\nCALL {move} S2\nCALL {move} S1\nSTOP\n",
        "state": prompt["state"], "registers": None, "pause_types": None})
    assert res["status"] == "ok", res["error"]
    assert len(res["calls"]) == 3


def test_a_loop_of_deletions_previews_every_one_of_them():
    """The gate used to show the first delete and hand back a token good for
    the rest — one click, seven cards gone."""
    res = run("let's delete the oldest issue", DELETE_TODO)
    assert res["status"] == "effect_blocked"
    err = res["error"]
    assert err["code"] == "DESTRUCTIVE" and err["effect"] == "DELETE"
    todo = [c["id"] for c in demo_state()["entities"]["card"]
            if c["status"] == "todo"]
    assert err["count"] == len(todo) > 1
    assert [c["args"][0] for c in err["calls"]] == todo
    assert all(c["name"] == "delete_card" for c in err["calls"])


def test_the_previewed_deletions_are_what_approval_then_does():
    todo = [c["id"] for c in demo_state()["entities"]["card"]
            if c["status"] == "todo"]
    ok = run("let's delete the oldest issue", DELETE_TODO, approval=True)
    assert ok["status"] == "ok"
    left = {c["id"] for c in ok["final_state"]["entities"]["card"]}
    assert left == {c["id"] for c in demo_state()["entities"]["card"]} - set(todo)


def test_one_deletion_still_previews_as_one():
    res = run("delete card 4", "CALL @delete_card $0\nSTOP\n")
    assert res["status"] == "effect_blocked"
    assert res["error"]["code"] == "DESTRUCTIVE" and res["error"]["count"] == 1
    assert res["error"]["calls"][0]["args"] == ["card_4"]


def test_without_preview_the_run_still_halts_at_the_first_destructive_call():
    """spec §7, and what every scored task relies on: no `preview` in the
    payload means the old behavior, exactly."""
    world = get_world("kanban")
    state = demo_state()
    constants = dev_server.constants_from_board("let's delete the oldest issue",
                                                state, world["now"])
    ctx, sandbox_ctx = build_context(world, constants, random.Random(7),
                                     symbols="typed", enums=True)
    res = run_sandbox({"js": build(resolve(DELETE_TODO, ctx), ctx).js,
                       "state": state, "now": world["now"], "approval": False,
                       **sandbox_ctx})
    assert res["status"] == "effect_blocked"
    assert res["error"]["code"] == "EFFECT_BLOCKED"
    assert res["error"]["tool"] and res["error"]["effect"] == "DELETE"
    # the halted call is deliberately not logged (unnecessary_destructive
    # diffs this log against the approved run's)
    assert [c["name"] for c in res["calls"]] == ["list_cards"]
