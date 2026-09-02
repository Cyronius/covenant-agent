"""Sandbox behaviors: effect gate, error injection, TRY/RETRY, PAUSE
round-trips, and full-task goal_success through the harness."""
import random

from core.pipeline import build
from harness.authoring import resolve
from harness.context import build_context
from harness.run import run_sandbox, run_task, reference_planner
from harness.taskbuild import build_task
from runtime.worlds import get_world


def _build(world_name, constants, src, seed=11):
    world = get_world(world_name)
    ctx, sandbox_ctx = build_context(world, constants, random.Random(seed))
    res = build(resolve(src, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    return world, ctx, sandbox_ctx, res


def _payload(world, sandbox_ctx, res, **kw):
    p = {"js": res.js, "state": world["default_state"],
         "tools": sandbox_ctx["tools"], "fields": sandbox_ctx["fields"],
         "constants": sandbox_ctx["constants"], "now": world["now"],
         "approval": False, "error_injection": [], "initial_registers": {}}
    p.update(kw)
    return p


def test_effect_gate_blocks_delete():
    world, ctx, sctx, res = _build(
        "kanban", [{"type": "ID:card", "value": "card_1", "desc": "c"}],
        "CALL @delete_card $0\nSTOP\n")
    out = run_sandbox(_payload(world, sctx, res, approval=False))
    assert out["status"] == "effect_blocked"
    assert out["error"]["effect"] == "DELETE"
    # state untouched
    assert len(out["state"]["entities"]["card"]) == \
        len(world["default_state"]["entities"]["card"])
    ok = run_sandbox(_payload(world, sctx, res, approval=True))
    assert ok["status"] == "ok"
    assert len(ok["state"]["entities"]["card"]) == \
        len(world["default_state"]["entities"]["card"]) - 1


def test_try_does_not_swallow_gate():
    world, ctx, sctx, res = _build(
        "kanban", [{"type": "ID:card", "value": "card_1", "desc": "c"}],
        "TRY RETRY 2 -> r0\n  CALL @delete_card $0\nSTOP\n")
    out = run_sandbox(_payload(world, sctx, res, approval=False))
    assert out["status"] == "effect_blocked"


def test_error_injection_and_retry():
    world, ctx, sctx, res = _build(
        "kanban", [{"type": "ID:card", "value": "card_1", "desc": "c"}],
        "TRY RETRY 3 -> r0\n  CALL @archive_card $0 -> r1\nSTOP\n")
    out = run_sandbox(_payload(
        world, sctx, res, approval=True,
        error_injection=[{"name": "archive_card", "code": "RATE_LIMITED",
                          "times": 2}]))
    assert out["status"] == "ok"
    assert [c["ok"] for c in out["calls"]] == [False, False, True]
    card = next(c for c in out["state"]["entities"]["card"]
                if c["id"] == "card_1")
    assert card["archived"] is True


def test_uncaught_tool_error_halts():
    world, ctx, sctx, res = _build(
        "kanban", [{"type": "ID:card", "value": "card_99", "desc": "c"}],
        "CALL @archive_card $0 -> r0\nSTOP\n")
    out = run_sandbox(_payload(world, sctx, res, approval=True))
    assert out["status"] == "error"
    assert out["error"]["code"] == "NOT_FOUND"


def test_pause_returns_registers():
    world, ctx, sctx, res = _build(
        "kanban", [{"type": "BOOL", "value": False, "desc": "false"}],
        "CALL @list_cards -> r0\n"
        "FILTER r0 @card.archived EQ $0 -> r1\nPAUSE\n")
    out = run_sandbox(_payload(world, sctx, res))
    assert out["status"] == "paused"
    assert {"r0", "r1"} <= set(out["registers"])
    assert all(not c["archived"] for c in out["registers"]["r1"])


def test_full_pause_task_roundtrip():
    world = get_world("kanban")
    constants = [{"type": "BOOL", "value": False, "desc": "false"}]
    ctx, sctx = build_context(world, constants, random.Random(4))
    segs = [resolve("CALL @list_cards -> r0\n"
                    "FILTER r0 @card.due LT NOW AND @card.archived EQ $0 -> r1\n"
                    "PAUSE\n", ctx),
            resolve("FOREACH r1 -> r2\n"
                    "  CALL @archive_card r2 -> r3\nSTOP\n", ctx)]
    task = build_task(task_id="t", level=10, world_name="kanban",
                      request="req", constants=constants, segments=segs,
                      prebuilt=(ctx, sctx))
    row = run_task(task, reference_planner(task))
    assert row["goal_success"] and row["pauses"] == 1


def test_parallel_executes_and_binds():
    world, ctx, sctx, res = _build(
        "crm", [{"type": "ID:customer", "value": "customer_2", "desc": "c"}],
        "PARALLEL\n"
        "  CALL @get_customer $0 -> r0\n"
        "  CALL @list_invoices -> r1\n"
        "COUNT r1 -> r2\nRETURN r2\n")
    out = run_sandbox(_payload(world, sctx, res))
    assert out["status"] == "ok"
    assert out["return_value"] == len(
        world["default_state"]["entities"]["invoice"])


def test_abort_reports_status_and_reason():
    world, ctx, sctx, res = _build("kanban", [], "ABORT NEEDS_INFO\n")
    out = run_sandbox(_payload(world, sctx, res, approval=True))
    assert out["status"] == "aborted"
    assert out["reason"] == "NEEDS_INFO"
    assert out["calls"] == []


def test_abort_task_scores_only_with_matching_reason():
    from core.ir import TaskContext
    task = build_task(
        task_id="t_abort", level=11, world_name="kanban",
        request="Assign card 4 to Cyrus.",
        constants=[{"type": "ID:card", "value": "card_4", "desc": "card 4"}],
        segments=["ABORT NOT_FOUND\n"], expected_status="aborted")
    assert task["reference"]["abort_reason"] == "NOT_FOUND"
    row = run_task(task, reference_planner(task))
    assert row["goal_success"] and row["correct_abstain"]
    wrong = run_task(task, lambda *a: "ABORT UNSUPPORTED\n")
    assert wrong["status"] == "aborted" and not wrong["goal_success"]
    ctx = TaskContext.from_json(task["context"])
    acted = run_task(task, lambda *a: resolve("CALL @list_cards -> r0\nSTOP\n", ctx))
    assert acted["status"] == "ok" and not acted["goal_success"]
    assert acted["correct_abstain"] is False


def test_format_renders_fields_and_dates():
    consts = [{"type": "ID:card", "value": "card_4", "desc": "card 4"},
              {"type": "STR", "value": "Copy of {0}", "desc": "title template"},
              {"type": "STR", "value": "{0} due {1}", "desc": "two-slot template"}]
    world, ctx, sctx, res = _build(
        "kanban", consts,
        "CALL @get_card $0 -> r0\n"
        "FORMAT $1 r0.@card.title -> r1\n"
        "FORMAT $2 r0.@card.title r0.@card.due -> r3\n"
        "CALL @create_card r1 r0.@card.due r0.@card.assignee -> r2\n"
        "RETURN r3\n")
    out = run_sandbox(_payload(world, sctx, res, approval=True))
    assert out["status"] == "ok", out
    titles = [c["title"] for c in out["state"]["entities"]["card"]]
    original = next(c for c in world["default_state"]["entities"]["card"] if c["id"] == "card_4")
    assert f"Copy of {original['title']}" in titles
    assert out["return_value"].startswith(original["title"] + " due ")
    assert str(original["due"]) not in out["return_value"]  # rendered as a date, not epoch


def test_external_tool_stubs_deterministically_and_gate_reports_args():
    consts = [{"type": "ID:user", "value": "user_1", "desc": "Bob"},
              {"type": "STR", "value": "Tell Bob what is overdue.", "desc": "brief"}]
    world, ctx, sctx, res = _build(
        "kanban", consts,
        "CALL @list_cards -> r0\n"
        "FILTER r0 @card.assignee EQ $0 AND @card.due LT NOW -> r1\n"
        "CALL @write_text $1 r1 -> r2\n"
        "CALL @send_message $0 r2\n"
        "STOP\n")
    blocked = run_sandbox(_payload(world, sctx, res, approval=False))
    assert blocked["status"] == "effect_blocked"
    assert blocked["error"]["args"][0] == "user_1"
    assert blocked["error"]["args"][1] == "[write_text: Tell Bob what is overdue.]"
    assert [c["name"] for c in blocked["calls"]] == ["list_cards", "write_text"]
    ok = run_sandbox(_payload(world, sctx, res, approval=True))
    assert ok["status"] == "ok"
    assert ok["state"]["outbox"][-1]["text"] == "[write_text: Tell Bob what is overdue.]"
