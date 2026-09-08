"""Per-task metrics (F3). One JSONL row per task run; the schema here is the
plan's metrics table."""
from __future__ import annotations

import json
from collections import Counter
from typing import List, Optional

from core.ir import DESTRUCTIVE_EFFECTS, Parallel, Program


def count_instructions(body: list) -> int:
    n = 0
    for instr in body:
        n += 1
        for attr in ("body", "then", "els"):
            child = getattr(instr, attr, None)
            if isinstance(child, list):
                n += count_instructions(child)
        if isinstance(instr, Parallel):
            n += len(instr.calls)
    return n


def program_instructions(program: Program) -> int:
    return count_instructions(program.body)


def normalize_state(state: dict) -> dict:
    ents = {
        e: sorted((dict(r) for r in recs), key=lambda r: str(r.get("id")))
        for e, recs in (state.get("entities") or {}).items()
    }
    # drop empty entity lists so created-then-deleted entities compare equal
    ents = {e: recs for e, recs in ents.items() if recs}
    return {
        "entities": ents,
        "outbox": state.get("outbox") or [],
        "payments": state.get("payments") or [],
    }


def states_equal(a: dict, b: dict) -> bool:
    return normalize_state(a) == normalize_state(b)


def destructive_call_keys(call_log: List[dict], sandbox_tools: List[dict]
                          ) -> List[str]:
    destructive = {t["name"] for t in sandbox_tools
                   if set(t["effects"]) & set(DESTRUCTIVE_EFFECTS)}
    keys = []
    for c in call_log:
        if c.get("name") in destructive and c.get("ok", True):
            keys.append(json.dumps({"name": c["name"], "args": c.get("args")},
                                   sort_keys=True))
    return keys


def unnecessary_destructive(actual_log: List[dict], reference_log: List[dict],
                            sandbox_tools: List[dict]) -> int:
    actual = Counter(destructive_call_keys(actual_log, sandbox_tools))
    ref = Counter(destructive_call_keys(reference_log, sandbox_tools))
    extra = actual - ref
    return sum(extra.values())


def metrics_row(task: dict, *, parse_ok: bool, compile_ok: bool,
                tool_valid: bool, arg_valid: bool, exec_ok: bool,
                status: str, final_state: Optional[dict],
                call_log: List[dict], sandbox_tools: List[dict],
                n_instructions: Optional[int], pauses: int,
                latency: dict, diagnostics: List[str],
                tokens_in: Optional[int] = None,
                tokens_out: Optional[int] = None,
                abort_reason: Optional[str] = None,
                abort_refs: Optional[List[str]] = None,
                segments: Optional[int] = None,
                error_turns: int = 0) -> dict:
    expected_status = task.get("expected_status", "ok")
    ref = task.get("reference", {})
    goal = (status == expected_status and final_state is not None
            and states_equal(final_state, task["expected_state"]))
    if expected_status == "aborted" and ref.get("abort_reason"):
        # an abstain is only correct for the right reason (spec §4)
        goal = goal and abort_reason == ref["abort_reason"]
    # referents (spec §4) are recorded, not gated: a right reason with a
    # wrong referent is still a correct abstain, and the column says so
    referent_match = None
    if expected_status == "aborted" and ref.get("abort_refs"):
        referent_match = sorted(abort_refs or []) == sorted(ref["abort_refs"])
    ref_calls = ref.get("calls") or None
    ref_instr = ref.get("instructions") or None
    n_calls = len(call_log)
    row = {
        "task_id": task["id"],
        "level": task["level"],
        "world": task["world"],
        "goal_success": goal,
        "parse_ok": parse_ok,
        "compile_ok": compile_ok,
        "tool_valid": tool_valid,
        "arg_valid": arg_valid,
        "exec_ok": exec_ok,
        "status": status,
        "tool_efficiency": (n_calls / ref_calls) if ref_calls else None,
        "program_efficiency": (n_instructions / ref_instr)
        if (ref_instr and n_instructions is not None) else None,
        "recovery_ok": goal if task.get("error_injection") else None,
        # abstain tasks: did the planner decline, and for the right reason?
        "correct_abstain": goal if expected_status == "aborted" else None,
        "abort_reason": abort_reason,
        "abort_refs": abort_refs or [],
        "abort_referent_match": referent_match,
        "unnecessary_destructive": unnecessary_destructive(
            call_log, ref.get("call_log", []), sandbox_tools),
        "pauses": pauses,
        # reactive execution (R4): planner turns that compiled and ran, and
        # how many of them were replies to a runtime error
        "segments": segments,
        "error_turns": error_turns,
        "latency_encode_ms": latency.get("encode"),
        "latency_generate_ms": latency.get("generate"),
        "latency_validate_ms": latency.get("validate"),
        "latency_compile_ms": latency.get("compile"),
        "latency_execute_ms": latency.get("execute"),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "diagnostics": diagnostics,
        "tags": task.get("tags", []),
        # the sandbox call log, names only, in order (routing suites score on
        # this: harness/real_suite.py); ok=False when the tool raised
        "calls": [{"name": c.get("name"), "ok": c.get("ok", True)} for c in call_log],
    }
    return row
