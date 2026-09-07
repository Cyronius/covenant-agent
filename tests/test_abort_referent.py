"""ABORT referents (spec §4, 0.3.0): `ABORT reason [sym [sym]]`.

The referent is what makes an abstention checkable and actionable. These
tests cover the grammar (parser), the kind rule (typecheck), the round-trip
to the sandbox, the harness columns, and the founded/unfounded checker.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.parser import parse  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness.abort_check import check_abort  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context  # noqa: E402
from harness.run import run_sandbox, run_task  # noqa: E402
from harness.taskbuild import build_task  # noqa: E402
from harness.run import reference_planner  # noqa: E402
from runtime.worlds import get_world  # noqa: E402


def _codes(text):
    prog, diags = parse(text)
    return prog, [d.code for d in diags]


# -- parser ----------------------------------------------------------------

def test_abort_parses_with_referents():
    prog, diags = parse("ABORT NEEDS_INFO F3\n")
    assert diags == []
    assert prog.body[0].reason == "NEEDS_INFO" and prog.body[0].refs == ["F3"]
    prog, diags = parse("ABORT AMBIGUOUS T2 T5\n")
    assert diags == [] and prog.body[0].refs == ["T2", "T5"]


def test_abort_without_referent_still_parses():
    prog, diags = parse("ABORT UNSUPPORTED\n")
    assert diags == [] and prog.body[0].refs == []


def test_abort_rejects_non_symbol_or_too_many_referents():
    for bad in ("ABORT NEEDS_INFO r0\n", "ABORT NOT_FOUND 7\n",
                "ABORT AMBIGUOUS T1 T2 T3\n", "ABORT NEEDS_INFO title\n"):
        prog, codes = _codes(bad)
        assert prog is None and codes == ["PARSE_ERROR"], bad


# -- typecheck ---------------------------------------------------------------

def _ctx(constants):
    world = get_world("kanban")
    ctx, sctx = build_context(world, constants, random.Random(3))
    return world, ctx, sctx


def _check(src, constants=()):
    world, ctx, _ = _ctx(list(constants))
    res = build(resolve(src, ctx), ctx)
    return [d.code for d in res.diagnostics], res, ctx


def _title_sym(ctx):
    return next(f.sym for f in ctx.fields.values()
                if f.entity == "card" and f.name == "title")


def test_referent_kind_must_match_reason():
    # NEEDS_INFO wants a field; a constant is the wrong kind
    codes, res, _ = _check("ABORT NEEDS_INFO $0\n",
                           [{"type": "STR", "value": "x", "desc": "x"}])
    assert codes == ["TYPE_ERROR"]
    assert res.rendered_diagnostics()[0].startswith(
        "TYPE_ERROR line:1 NEEDS_INFO:F C0")
    # UNSUPPORTED admits none
    codes, _, _ = _check("ABORT UNSUPPORTED @card.title\n")
    assert codes == ["TYPE_ERROR"]


def test_referent_must_be_declared():
    codes, res, _ = _check("ABORT NOT_FOUND C9\n")
    assert codes == ["UNBOUND"]
    codes, res, _ = _check("ABORT NEEDS_INFO F99\n")
    assert codes == ["UNKNOWN_FIELD"]
    assert res.rendered_diagnostics()[0] == "UNKNOWN_FIELD ABORT F99"
    codes, _, _ = _check("ABORT AMBIGUOUS T40\n")
    assert codes == ["UNKNOWN_TOOL"]


def test_valid_referents_compile_to_rt_abort_with_refs():
    codes, res, ctx = _check("ABORT AMBIGUOUS @list_cards @card.title\n")
    assert codes == []
    assert 'rt.abort("AMBIGUOUS", [' in res.js


# -- sandbox + harness -------------------------------------------------------

def test_sandbox_carries_refs():
    world, ctx, sctx = _ctx([{"type": "ID:user", "value": "user_1",
                              "desc": "Bob"}])
    res = build(resolve("ABORT NEEDS_INFO @card.title\n", ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    out = run_sandbox({
        "js": res.js, "state": world["default_state"], "tools": sctx["tools"],
        "fields": sctx["fields"], "constants": sctx["constants"],
        "now": world["now"], "approval": False, "error_injection": [],
        "initial_registers": {}})
    assert out["status"] == "aborted" and out["reason"] == "NEEDS_INFO"
    assert out["refs"] == [_title_sym(ctx)]


def test_task_reference_and_row_carry_referents():
    # build_task takes symbolic segments; resolve against the same context it
    # will use (the harness/curriculum.py pattern)
    constants = [{"type": "ID:user", "value": "user_1", "desc": "Bob"}]
    world = get_world("kanban")
    ctx, sctx = build_context(world, constants, random.Random(1000))
    task = build_task(
        task_id="t_ref", level=11, world_name="kanban",
        request="Create a new card for Bob.", constants=constants,
        segments=[resolve("ABORT NEEDS_INFO @card.title\n", ctx)],
        seed=1000, expected_status="aborted", prebuilt=(ctx, sctx))
    assert task["reference"]["abort_reason"] == "NEEDS_INFO"
    assert len(task["reference"]["abort_refs"]) == 1
    row = run_task(task, reference_planner(task))
    assert row["goal_success"] and row["correct_abstain"]
    assert row["abort_referent_match"] is True
    # right reason, missing referent: still a correct abstain, column says no
    bare = run_task(task, lambda *a: "ABORT NEEDS_INFO\n")
    assert bare["correct_abstain"] and bare["abort_referent_match"] is False


# -- the checker -------------------------------------------------------------

def test_not_found_is_unfounded_when_a_record_matches():
    world, ctx, _ = _ctx([{"type": "STR", "value": "Bob",
                           "desc": "the assignee named"}])
    msg = check_abort(ctx, world["default_state"], "NOT_FOUND", ["C0"])
    assert msg and msg.startswith("ABORT_UNFOUNDED NOT_FOUND C0")
    assert "user_1" in msg


def test_not_found_is_founded_when_nothing_matches():
    world, ctx, _ = _ctx([{"type": "STR", "value": "Cyrus",
                           "desc": "the assignee named"}])
    assert check_abort(ctx, world["default_state"], "NOT_FOUND", ["C0"]) is None


def test_needs_info_is_unfounded_when_a_constant_of_that_type_exists():
    world, ctx, _ = _ctx([{"type": "STR", "value": "Write the report",
                           "desc": "title"}])
    title = _title_sym(ctx)
    msg = check_abort(ctx, world["default_state"], "NEEDS_INFO", [title])
    assert msg == f"ABORT_UNFOUNDED NEEDS_INFO {title} has C0"


def test_needs_info_is_founded_when_no_constant_fits():
    world, ctx, _ = _ctx([{"type": "ID:user", "value": "user_1",
                           "desc": "Bob"}])
    assert check_abort(ctx, world["default_state"], "NEEDS_INFO",
                       [_title_sym(ctx)]) is None


def test_ambiguous_with_one_candidate_is_unfounded():
    world, ctx, _ = _ctx([])
    t = next(iter(ctx.tools))
    msg = check_abort(ctx, world["default_state"], "AMBIGUOUS", [t])
    assert msg and msg.startswith("ABORT_UNFOUNDED AMBIGUOUS")
    a, b = list(ctx.tools)[:2]
    assert check_abort(ctx, world["default_state"], "AMBIGUOUS", [a, b]) is None


def test_no_referent_means_nothing_to_check():
    world, ctx, _ = _ctx([])
    assert check_abort(ctx, world["default_state"], "NEEDS_INFO", []) is None
    assert check_abort(ctx, world["default_state"], "UNSUPPORTED", []) is None
