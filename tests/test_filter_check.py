"""The FILTER-padding flag (plan step 2a, `results/REFLEX.md`).

A metric that never fires and a metric that always fires are equally
useless, so these pin both edges: it fires on the demo's own padding shape
and stays quiet on clauses the request grounds, by field or by value.
"""
import random

import pytest

from core.pipeline import build
from harness.authoring import resolve
from harness.context import build_context
from harness.filter_check import padded_clauses
from harness.reflex_baseline import reflex_program
from runtime.worlds import get_world

CONSTANTS = [
    {"type": "STR", "value": "done", "desc": "the completed status"},
    {"type": "BOOL", "value": False, "desc": "false"},
    {"type": "ID:user", "value": "user_1", "desc": "Bob"},
]


def _ctx():
    return build_context(get_world("kanban"), CONSTANTS, random.Random(5))[0]


@pytest.mark.parametrize("request_text,program,expected", [
    # grounded by the value the request spells
    ("archive the done cards",
     "CALL @list_cards -> r0\nFILTER r0 @card.status EQ $0 -> r1\nSTOP\n", 0),
    # grounded on both clauses: "done" and "archived"
    ("archive the done cards that are not archived yet",
     "CALL @list_cards -> r0\n"
     "FILTER r0 @card.status EQ $0 AND @card.archived EQ $1 -> r1\nSTOP\n", 0),
    # grounded by the field name alone, with NOW on the right
    ("list the overdue cards",
     "CALL @list_cards -> r0\nFILTER r0 @card.due LT NOW -> r1\nRETURN r1\n", 0),
    # grounded by a constant's description, not its value ("Bob" -> user_1)
    ("which cards are Bob's?",
     "CALL @list_cards -> r0\nFILTER r0 @card.assignee EQ $2 -> r1\n"
     "RETURN r1\n", 0),
    # the padding shape: a request that asks for everything, filtered anyway
    ("list every card",
     "CALL @list_cards -> r0\nFILTER r0 @card.archived EQ $1 -> r1\n"
     "RETURN r1\n", 1),
    # both clauses unrequested
    ("list every card",
     "CALL @list_cards -> r0\n"
     "FILTER r0 @card.status EQ $0 AND @card.archived EQ $1 -> r1\n"
     "RETURN r1\n", 2),
])
def test_padding_is_flagged_and_grounding_is_not(request_text, program,
                                                 expected):
    ctx = _ctx()
    res = build(resolve(program, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    assert len(padded_clauses(res.program, ctx, request_text)) == expected


def test_a_filterless_program_is_never_padded():
    ctx = _ctx()
    res = build(resolve("CALL @list_cards -> r0\nRETURN r0\n", ctx), ctx)
    assert padded_clauses(res.program, ctx, "list every card") == []


def test_the_reflex_planner_writes_a_program_that_compiles():
    """The null is only a null if it runs: a skeleton that fails to compile
    would measure the compiler, not the reflex (`results/REFLEX.md` §1
    reports 100% compile on both demo suites)."""
    ctx = _ctx()
    for request in ("what cards are there", "archive the done cards",
                    "message Bob about the release"):
        res = build(reflex_program(ctx, request), ctx)
        assert res.compile_ok, (request, res.rendered_diagnostics())


def test_the_reflex_planner_pads_when_the_request_asks_for_everything():
    """What the null is for: the skeleton fills the FILTER slot whatever the
    request says, which is the habit the corpus teaches."""
    ctx = _ctx()
    res = build(reflex_program(ctx, "what cards are there"), ctx)
    assert padded_clauses(res.program, ctx, "what cards are there")
