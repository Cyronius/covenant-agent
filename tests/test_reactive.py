"""Reactive execution (reactive-execution.md §2B, PLAN.md §6 R4): a runtime
error outside TRY returns to the planner as a turn — with the failing line,
the calls that ran, and the registers the sandbox had bound — instead of
ending the task. Covers the sandbox report, the §9 RUNTIME rendering, the
harness loop (react_on_error), its segment bound, and run_a's prompt block."""
import random

from core import diagnostics as dg
from core.pipeline import build
from harness.authoring import resolve
from harness.context import build_context
from harness.run import MAX_SEGMENTS, run_sandbox, run_task, reference_planner
from harness.taskbuild import build_task
from runtime.worlds import get_world


def _kanban(constants):
    world = get_world("kanban")
    ctx, sctx = build_context(world, constants, random.Random(7))
    return world, ctx, sctx


CYRUS = [{"type": "STR", "value": "Cyrus", "desc": "the assignee named"},
         {"type": "ID:card", "value": "card_1", "desc": "card 1"}]

# the Cyrus program (results/S2.md): list, filter on a name nobody has,
# SELECT 0 on the empty match raises INDEX_OUT_OF_RANGE on line 3
LOOKUP = ("CALL @list_users -> r0\n"
          "FILTER r0 @user.name EQ $0 -> r1\n"
          "SELECT r1 0 -> r2\n"
          "GET r2.@user.id -> r3\n"
          "CALL @assign_card $1 r3\n"
          "STOP\n")


def _payload(ctx, sctx, world, text, registers=None):
    res = build(text, ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    return {"js": res.js, "state": world["default_state"],
            "tools": sctx["tools"], "fields": sctx["fields"],
            "constants": sctx["constants"], "now": "2026-01-01T00:00:00Z",
            "initial_registers": registers or {}}, res


# -- sandbox: what a failed segment reports ---------------------------------

def test_sandbox_error_reports_line_and_bound_registers():
    world, ctx, sctx = _kanban(CYRUS)
    payload, _ = _payload(ctx, sctx, world, resolve(LOOKUP, ctx))
    out = run_sandbox(payload)
    assert out["status"] == "error"
    assert out["error"]["code"] == "INDEX_OUT_OF_RANGE"
    assert out["line"] == 3
    # r0 (the list) and r1 (the empty match) were bound; r2+ never were
    assert set(out["registers"]) == {"r0", "r1"}
    assert out["registers"]["r1"] == []
    assert len(out["registers"]["r0"]) > 0
    # the read ran, the write did not
    assert [c["name"] for c in out["calls"]] == ["list_users"]


def test_sandbox_pause_report_unchanged():
    world, ctx, sctx = _kanban(CYRUS)
    payload, _ = _payload(ctx, sctx, world, resolve(
        "CALL @list_users -> r0\nFILTER r0 @user.name EQ $0 -> r1\nPAUSE\n", ctx))
    out = run_sandbox(payload)
    assert out["status"] == "paused" and set(out["registers"]) == {"r0", "r1"}


# -- §9 rendering -----------------------------------------------------------

def test_runtime_diagnostic_render():
    d = dg.runtime_error("INDEX_OUT_OF_RANGE", 3, "index 0 of 0")
    assert d.render() == "RUNTIME INDEX_OUT_OF_RANGE line:3 index 0 of 0"
    assert dg.runtime_error("TIMEOUT", 0, "").render() == "RUNTIME TIMEOUT line:0"


def test_compiled_js_carries_line_markers_and_snapshot():
    _, ctx, _ = _kanban(CYRUS)
    res = build(resolve(LOOKUP, ctx), ctx)
    assert "rt.at(1);" in res.js and "rt.at(5);" in res.js
    assert "rt.snapshot = () => ({r0: r0, r1: r1, r2: r2, r3: r3});" in res.js
    assert res.final_env["r1"].startswith("LIST")


# -- harness: the error becomes a turn --------------------------------------

def _cyrus_task():
    world, ctx, sctx = _kanban(CYRUS)
    return build_task(task_id="cyrus", level=11, world_name="kanban",
                      request="Assign card 1 to Cyrus.", constants=CYRUS,
                      segments=["ABORT NOT_FOUND C0\n"],
                      expected_status="aborted", prebuilt=(ctx, sctx))


def test_error_is_terminal_without_react():
    task = _cyrus_task()
    ctx_text = resolve(LOOKUP, task_context(task))
    row = run_task(task, lambda *a: ctx_text)
    assert row["status"] == "error" and not row["goal_success"]
    assert row["error_turns"] == 0 and row["segments"] == 1
    assert row["diagnostics"][-1].startswith("RUNTIME INDEX_OUT_OF_RANGE line:3")


def task_context(task):
    from core.ir import TaskContext
    return TaskContext.from_json(task["context"])


def test_error_becomes_a_turn_with_react():
    task = _cyrus_task()
    seen = []

    def planner(request, ctx, seg_idx, registers, feedback):
        seen.append((seg_idx, dict(registers), feedback))
        if seg_idx == 0:
            return resolve(LOOKUP, ctx)
        # the continuation sees r1 empty and declines for the right reason
        assert feedback["error"].startswith(
            "RUNTIME INDEX_OUT_OF_RANGE line:3")
        assert [c["name"] for c in feedback["calls"]] == ["list_users"]
        assert registers["r1"] == [] and "r0" in ctx.initial_registers
        return "ABORT NOT_FOUND C0\n"

    row = run_task(task, planner, react_on_error=True)
    assert row["status"] == "aborted" and row["goal_success"]
    assert row["correct_abstain"] and row["abort_referent_match"]
    assert row["segments"] == 2 and row["error_turns"] == 1
    assert row["pauses"] == 0
    assert seen[0][2] is None and seen[1][2] is not None
    # the failed segment's diagnostic is kept, the read is in the call log
    assert any(d.startswith("RUNTIME") for d in row["diagnostics"])
    assert [c["name"] for c in row["calls"]] == ["list_users"]


def test_continuation_may_use_bound_registers():
    """After the failure the continuation is typed from the program's final
    env: r0 (LIST user) is usable without a re-fetch."""
    task = _cyrus_task()
    task["expected_status"] = "ok"

    def planner(request, ctx, seg_idx, registers, feedback):
        if seg_idx == 0:
            return resolve(LOOKUP, ctx)
        assert set(ctx.initial_registers) == {"r0", "r1"}
        return "COUNT r0 -> r4\nRETURN r4\n"

    row = run_task(task, planner, react_on_error=True)
    assert row["status"] == "ok" and row["error_turns"] == 1
    assert [c["name"] for c in row["calls"]] == ["list_users"]


def test_react_is_bounded_by_max_segments():
    task = _cyrus_task()
    calls = []

    def planner(request, ctx, seg_idx, registers, feedback):
        calls.append(seg_idx)
        return resolve(LOOKUP, ctx) if seg_idx == 0 else "SELECT r1 0 -> r5\nSTOP\n"

    row = run_task(task, planner, react_on_error=True)
    assert row["status"] == "segment_limit"
    assert len(calls) == MAX_SEGMENTS
    assert row["error_turns"] == MAX_SEGMENTS and row["segments"] == MAX_SEGMENTS


def test_error_inside_try_is_not_a_turn():
    task = _cyrus_task()
    task["expected_status"] = "ok"
    turns = []

    def planner(request, ctx, seg_idx, registers, feedback):
        turns.append(feedback)
        return resolve("CALL @list_users -> r0\n"
                       "FILTER r0 @user.name EQ $0 -> r1\n"
                       "TRY -> r9\n"
                       "  SELECT r1 0 -> r2\n"
                       "STOP\n", ctx)

    row = run_task(task, planner, react_on_error=True)
    assert row["status"] == "ok" and row["error_turns"] == 0
    assert turns == [None]


def test_reference_planner_unaffected():
    task = _cyrus_task()
    row = run_task(task, reference_planner(task))
    assert row["goal_success"] and row["segments"] == 1 and row["error_turns"] == 0


# -- run_a: the prompt block ------------------------------------------------

def test_build_prompt_failure_block():
    from baselines.qwen.run_a import build_prompt, REACTIVE, SYSTEM
    fb = {"error": "RUNTIME INDEX_OUT_OF_RANGE line:3 index 0 of 0",
          "calls": [{"name": "list_users", "ok": True}],
          "registers": {"r1": []}}
    p = build_prompt("REQUEST: x", {"r1": []}, ["CALL T0 -> r0\nSTOP"], fb)
    assert "FAILED at runtime" in p
    assert "FAILURE: RUNTIME INDEX_OUT_OF_RANGE line:3 index 0 of 0" in p
    assert "CALLS THAT RAN: list_users(ok)" in p
    assert p.endswith("PROGRAM:")
    # no feedback: the PAUSE block, unchanged
    q = build_prompt("REQUEST: x", {"r1": []}, ["CALL T0 -> r0\nPAUSE"])
    assert "ended at PAUSE" in q and "FAILURE" not in q
    assert REACTIVE not in SYSTEM and "PAUSE" in REACTIVE
