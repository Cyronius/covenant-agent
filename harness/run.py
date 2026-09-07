"""Task runner (F3): the full pipeline

  request + tool schemas -> planner -> Agent Core text
  -> parse -> typecheck -> effects -> compile -> JS
  -> sandbox execute (with PAUSE round-trips) -> final state -> metrics

A *planner* is any callable (request, ctx, seg_idx, registers) -> Agent Core
text. `reference_planner` replays a task's stored reference segments; model
planners (R2+) plug in the same way.

CLI: python -m harness.run --tasks path.jsonl [--out metrics.jsonl]
     runs every task with its reference program (harness self-check).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import diagnostics as dg  # noqa: E402
from core.ir import TaskContext, parse_type  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness import metrics as M  # noqa: E402
from harness.context import sandbox_from_context  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

SANDBOX = ROOT / "runtime" / "sandbox.js"
MAX_SEGMENTS = 8

Planner = Callable[[str, TaskContext, int, Dict], Optional[str]]


def run_sandbox(payload: dict) -> dict:
    proc = subprocess.run(
        ["node", str(SANDBOX)], input=json.dumps(payload).encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=150 if payload.get("external_url") else 30)
    out = proc.stdout.decode().strip()
    if not out:
        return {"status": "error",
                "error": {"code": "SANDBOX_CRASH",
                          "message": proc.stderr.decode()[-500:]}}
    return json.loads(out)


def reference_planner(task: dict) -> Planner:
    segments = task["reference"]["segments"]

    def plan(request, ctx, seg_idx, registers):
        if seg_idx >= len(segments):
            return None
        return segments[seg_idx]

    return plan


def run_task(task: dict, planner: Planner) -> dict:
    """Execute one task with a planner; return the metrics row (JSONL-ready)."""
    ctx = TaskContext.from_json(task["context"])
    world = get_world(task["world"])
    sandbox_ctx = task.get("sandbox") or sandbox_from_context(ctx, world)

    state = task["state"]
    registers: Dict = {}
    pauses = 0
    call_log: list = []
    n_instructions = 0
    status = "error"
    final_state = state
    parse_ok = compile_ok = tool_valid = arg_valid = exec_ok = False
    diagnostics: list = []
    sres: Optional[dict] = None
    lat = {"generate": 0.0, "validate": 0.0, "compile": 0.0, "execute": 0.0}

    for seg_idx in range(MAX_SEGMENTS):
        t0 = time.perf_counter()
        text = planner(task["request"], ctx, seg_idx, registers)
        lat["generate"] += (time.perf_counter() - t0) * 1000
        if text is None:
            status = "planner_exhausted"
            break

        t0 = time.perf_counter()
        result = build(text, ctx)
        lat["validate"] += (time.perf_counter() - t0) * 1000
        diagnostics += result.rendered_diagnostics()
        parse_ok = result.parse_ok
        codes = {d.code for d in result.diagnostics}
        tool_valid = parse_ok and "UNKNOWN_TOOL" not in codes
        arg_valid = parse_ok and not ({"MISSING_ARG", "TYPE_ERROR"} & codes)
        compile_ok = result.compile_ok
        if not result.compile_ok:
            status = "static_error"
            break
        n_instructions += M.program_instructions(result.program)

        payload = {
            "js": result.js,
            "state": state,
            "tools": sandbox_ctx["tools"],
            "fields": sandbox_ctx["fields"],
            "constants": sandbox_ctx["constants"],
            "now": task["now"],
            "approval": task.get("approval", False),
            "error_injection": task.get("error_injection", []),
            "initial_registers": registers,
        }
        t0 = time.perf_counter()
        sres = run_sandbox(payload)
        lat["execute"] += (time.perf_counter() - t0) * 1000

        status = sres["status"]
        final_state = sres.get("state", state)
        call_log += sres.get("calls", [])
        if status == "paused":
            pauses += 1
            state = final_state
            registers = sres["registers"]
            # seed the next segment's typing environment from this PAUSE
            pause_env = result.pause_envs[0] if result.pause_envs else {}
            ctx = TaskContext.from_json(task["context"])
            ctx.initial_registers = {
                r: parse_type(t) for r, t in pause_env.items()
                if r in registers}
            continue
        break
    else:
        status = "segment_limit"

    exec_ok = status in ("ok", "effect_blocked", "aborted") and compile_ok
    abort_reason = sres.get("reason") if (sres and status == "aborted") else None
    abort_refs = list(sres.get("refs") or []) if (sres and status == "aborted") else []
    if status == "error" and sres is not None:
        err = sres.get("error", {})
        diagnostics.append(f"RUNTIME {err.get('code')} {err.get('message', '')}")

    row = M.metrics_row(
        task,
        parse_ok=parse_ok, compile_ok=compile_ok, tool_valid=tool_valid,
        arg_valid=arg_valid, exec_ok=exec_ok, status=status,
        final_state=final_state, call_log=call_log,
        sandbox_tools=sandbox_ctx["tools"],
        n_instructions=n_instructions or None, pauses=pauses,
        latency=lat, diagnostics=diagnostics, abort_reason=abort_reason,
        abort_refs=abort_refs)
    return row


def run_reference_suite(tasks: list, out_path: Optional[Path] = None) -> list:
    rows = []
    for task in tasks:
        row = run_task(task, reference_planner(task))
        rows.append(row)
    if out_path:
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return rows


def load_tasks(path: Path) -> list:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tasks = load_tasks(Path(args.tasks))
    rows = run_reference_suite(
        tasks, Path(args.out) if args.out else None)
    ok = sum(1 for r in rows if r["goal_success"])
    print(f"{ok}/{len(rows)} goal_success")
    for r in rows:
        if not r["goal_success"]:
            print(f"  FAIL {r['task_id']} status={r['status']} "
                  f"diags={r['diagnostics'][:3]}")
    sys.exit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()
