"""The host resolves the people a request names before the model runs.

"assign all issues to cyrus" assigned four cards to Bob. No constant table
fixes that (plan .claude/plans/host-preflight-and-bulk-gate.md §1); the
board lookup that says "there is no Cyrus" can't be got wrong.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import dev_server  # noqa: E402
from harness.demo_suite import CASES, demo_state  # noqa: E402


def pf(request: str, state=None):
    return dev_server.preflight_people(request, state or demo_state())


def test_an_unknown_assignee_stops_the_turn():
    res = pf("assign all issues to cyrus")
    assert res and res["status"] == "unknown_person" and res["name"] == "cyrus"
    assert "cyrus" in res["message"] and "Bob Alvarez" in res["message"]
    # no invented did-you-mean: nothing on the board is close to "cyrus"
    assert res["suggestion"] is None


def test_the_same_name_inside_another_construction():
    res = pf("duplicate the release notes issue and make cyrus the owner")
    assert res and res["name"] == "cyrus"


def test_a_near_miss_gets_a_did_you_mean():
    res = pf("message prya about her overdue card")
    assert res and res["suggestion"] == "Priya Nandan"
    assert "did you mean Priya Nandan?" in res["message"]


def test_two_people_with_the_same_first_name_are_ambiguous():
    state = demo_state()
    state["entities"]["user"].append(
        {"id": "user_9", "name": "Bob Chen", "email": "bobc@understory.test"})
    res = pf("assign card 6 to bob", state)
    assert res and res["status"] == "ambiguous_person"
    assert res["candidates"] == ["Bob Alvarez", "Bob Chen"]


def test_every_other_demo_request_names_nobody_missing():
    """The false-positive guard. Every request people actually typed at the
    board (harness/demo_suite.py) resolves, except the one about Cyrus —
    including "add a new issue for recompiling fortran", which a looser
    `for X` rule would read as a person."""
    tripped = [c[0] for c in CASES if pf(c[0])]
    assert tripped == ["duplicate the release notes issue and make cyrus the owner"]


def test_requests_that_name_nobody_at_all():
    for request in ("move the release notes to doing column",
                    "add a new issue for recompiling fortran",
                    "archive everything that is done",
                    "set card 3 to done",
                    "assign card 6 to me",
                    "list all the cards",
                    # the possessive rule reading a contraction as a name is
                    # this check's own failure mode — "nobody called let"
                    "let's archive the done cards",
                    "what's overdue?",
                    "who's on card 4",
                    "show me today's overdue cards",
                    "assign the oauth card to whoever is free",
                    "message the person who owns card 3"):
        assert pf(request) is None, request


def test_every_way_of_naming_someone_who_is_here():
    for request in ("give card 1 to luz",
                    "hand the tls cert card to theo",
                    "reassign everything of bob's to priya",
                    "notify kade about card 6",
                    "make Priya the owner of card 2",
                    "delete all the cards owned by bob"):
        assert pf(request) is None, request


def test_the_prompt_endpoint_carries_it():
    res = dev_server.handle_kanban_prompt(
        {"request": "assign all issues to cyrus", "state": demo_state()})
    assert res["preflight"]["status"] == "unknown_person"
    # the prompt is still built — only the client's decision to generate changes
    assert res["grammar"] and res["context"]["tools"]
    ok = dev_server.handle_kanban_prompt(
        {"request": "assign card 6 to Bob", "state": demo_state()})
    assert ok["preflight"] is None
