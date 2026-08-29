"""Task construction: run a reference program against a world to derive the
expected final state (backtranslation's execution step, F4), and package a
self-contained task dict for harness.run.

The reference execution always carries the approval token — the expected
state is what the *approved* correct program produces — unless the task
explicitly tests the effect gate (expected_status="effect_blocked").
"""
from __future__ import annotations

import random
from typing import List, Optional

from core.ir import TaskContext, parse_type
from core.pipeline import build
from harness.metrics import program_instructions
from harness.run import run_sandbox
from harness.context import build_context
from runtime.worlds import get_world


class ReferenceError(Exception):
    pass


def execute_segments(segments: List[str], ctx: TaskContext, sandbox_ctx: dict,
                     state: dict, now: int, approval: bool = True,
                     error_injection: Optional[list] = None) -> dict:
    """Run reference segments through build + sandbox, chaining PAUSEs.
    Returns {state, call_log, instructions, calls, pauses, status,
    return_value}."""
    registers: dict = {}
    call_log: list = []
    instructions = 0
    pauses = 0
    status = "error"
    return_value = None
    cur_ctx = ctx
    for i, text in enumerate(segments):
        res = build(text, cur_ctx)
        if not res.compile_ok:
            raise ReferenceError(
                f"segment {i} failed static checks: "
                f"{res.rendered_diagnostics()[:5]}")
        instructions += program_instructions(res.program)
        sres = run_sandbox({
            "js": res.js, "state": state,
            "tools": sandbox_ctx["tools"], "fields": sandbox_ctx["fields"],
            "constants": sandbox_ctx["constants"], "now": now,
            "approval": approval,
            "error_injection": error_injection or [],
            "initial_registers": registers,
        })
        status = sres["status"]
        state = sres.get("state", state)
        call_log += sres.get("calls", [])
        if status == "paused":
            if i == len(segments) - 1:
                raise ReferenceError("reference ended in PAUSE")
            pauses += 1
            registers = sres["registers"]
            pause_env = res.pause_envs[0] if res.pause_envs else {}
            cur_ctx = TaskContext(
                tools=ctx.tools, fields=ctx.fields, constants=ctx.constants,
                initial_registers={r: parse_type(t)
                                   for r, t in pause_env.items()
                                   if r in registers})
            continue
        if status == "return" or status == "ok":
            return_value = sres.get("return_value")
        if i != len(segments) - 1:
            raise ReferenceError(
                f"reference terminated at segment {i} with status {status}")
    return {"state": state, "call_log": call_log,
            "instructions": instructions, "calls": len(call_log),
            "pauses": pauses, "status": status, "return_value": return_value}


def build_task(*, task_id: str, level: int, world_name: str, request: str,
               constants: List[dict], segments: List[str],
               seed: int = 0, approval: bool = True,
               error_injection: Optional[list] = None,
               expected_status: str = "ok",
               tool_subset: Optional[List[str]] = None,
               state: Optional[dict] = None,
               tags: Optional[List[str]] = None,
               provenance: Optional[dict] = None,
               prebuilt: Optional[tuple] = None) -> dict:
    """Build one self-contained task. `segments` use the symbol assignment
    produced by seed (use context_preview() to see it while authoring).
    `prebuilt` optionally supplies (ctx, sandbox_ctx) to skip re-derivation."""
    world = get_world(world_name)
    if prebuilt is not None:
        ctx, sandbox_ctx = prebuilt
    else:
        ctx, sandbox_ctx = build_context(
            world, constants, random.Random(seed), tool_subset)
    init_state = state if state is not None else world["default_state"]
    now = world["now"]

    ref = execute_segments(
        segments, ctx, sandbox_ctx, init_state, now,
        approval=(expected_status != "effect_blocked"),
        error_injection=error_injection)
    # When the task expects the effect gate to block, the reference run above
    # already ran without approval; its state is whatever the gated run left
    # behind (reads may have happened).
    if ref["status"] != expected_status:
        raise ReferenceError(
            f"{task_id}: reference status {ref['status']}, "
            f"expected {expected_status}")
    expected_state = ref["state"]

    return {
        "id": task_id,
        "level": level,
        "world": world_name,
        "request": request,
        "context": ctx.to_json(),
        "state": init_state,
        "now": now,
        "approval": expected_status != "effect_blocked",
        "error_injection": error_injection or [],
        "expected_status": expected_status,
        "expected_state": expected_state,
        "reference": {
            "segments": segments,
            "call_log": ref["call_log"],
            "calls": ref["calls"],
            "instructions": ref["instructions"],
            "pauses": ref["pauses"],
            "return_value": ref["return_value"],
        },
        "tags": tags or [],
        "provenance": provenance or {},
    }
