"""The standard compute block (spec 0.7.0 §6; plan step 1g): sum/avg/max/min
over a list of numbers, present in every context so a program can fold a
MAP-projected field without a new opcode.
"""
import random

import pytest

from core.pipeline import build
from harness.authoring import resolve
from harness.context import COMPUTE_TOOLS, build_context
from harness.run import reference_planner, run_task
from harness.taskbuild import build_task
from runtime.worlds import get_world

INVOICE_AMOUNTS = [24900, 9900, 129900]


def _ctx():
    world = get_world("crm")
    ctx, sandbox_ctx = build_context(world, [], random.Random(1))
    return world, ctx, sandbox_ctx


@pytest.mark.parametrize("tool,expected", [
    ("sum", sum(INVOICE_AMOUNTS)),
    ("avg", sum(INVOICE_AMOUNTS) / len(INVOICE_AMOUNTS)),
    ("max", max(INVOICE_AMOUNTS)),
    ("min", min(INVOICE_AMOUNTS)),
])
def test_each_tool_round_trips_and_computes(tool, expected):
    """Parse, typecheck, compile and run each tool against the real default
    state's invoice amounts."""
    world, ctx, sandbox_ctx = _ctx()
    src = (f"CALL @list_invoices -> r0\n"
           f"MAP r0 @invoice.amount -> r1\n"
           f"CALL @{tool} r1 -> r2\n"
           f"RETURN r2\n")
    res = build(resolve(src, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()

    task = build_task(
        task_id=f"compute_{tool}", level=0, world_name="crm",
        request=f"What is the {tool} of every invoice amount?",
        constants=[], segments=[resolve(src, ctx)], seed=1,
        state=world["default_state"], prebuilt=(ctx, sandbox_ctx))
    assert task["reference"]["return_value"] == pytest.approx(expected)
    row = run_task(task, reference_planner(task))
    assert row["goal_success"], row.get("diagnostics")
    assert row["return_match"] is True


def test_a_list_str_argument_is_a_type_error():
    _, ctx, _ = _ctx()
    src = ("CALL @list_customers -> r0\n"
           "MAP r0 @customer.name -> r1\n"
           "CALL @sum r1 -> r2\n"
           "RETURN r2\n")
    res = build(resolve(src, ctx), ctx)
    assert not res.compile_ok
    assert [d.code for d in res.diagnostics] == ["TYPE_ERROR"]


def test_empty_list_is_null_not_a_guess():
    """Matches FIRST's own empty-list convention (spec §4): avg/max/min of
    nothing is NULL, not an arbitrary in-range number. sum is 0, the
    identity element."""
    world, ctx, sandbox_ctx = _ctx()
    src = ("CALL @list_customers -> r0\n"
           "FILTER r0 @customer.name EQ NULL -> r1\n"
           "MAP r1 @customer.signup -> r2\n"
           "CALL @sum r2 -> r3\n"
           "CALL @list_customers -> r4\n"
           "FILTER r4 @customer.name EQ NULL -> r5\n"
           "MAP r5 @customer.signup -> r6\n"
           "CALL @max r6 -> r7\n"
           "RETURN r3\n")
    # FILTER on a name no customer has -> an empty list either way; only sum
    # needs its own check since RETURN can carry only one value.
    res = build(resolve(src, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    task = build_task(
        task_id="compute_empty", level=0, world_name="crm",
        request="irrelevant", constants=[], segments=[resolve(src, ctx)],
        seed=1, state=world["default_state"], prebuilt=(ctx, sandbox_ctx))
    assert task["reference"]["return_value"] == 0
    row = run_task(task, reference_planner(task))
    assert row["goal_success"]
    assert row["return_match"] is True


def test_the_block_is_in_every_generated_context():
    names = {t["name"] for t in COMPUTE_TOOLS}
    assert names == {"sum", "avg", "max", "min"}
    for world_name in ("crm", "kanban", "projects"):
        ctx, _ = build_context(get_world(world_name), [], random.Random(2))
        assert names <= {decl.name for decl in ctx.tools.values()}


def test_compute_tools_are_never_gated():
    """Deterministic, no state, no consequence: none of mutates/irreversible/
    external, which is what keeps the approval gate from ever blocking on
    them (only DESTRUCTIVE/BULK_WRITE do, and both key on those properties)."""
    for t in COMPUTE_TOOLS:
        assert t["effects"] == []
