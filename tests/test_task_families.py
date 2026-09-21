"""Families B, D, E and G: the recipes that ride on the existing worlds.

Plan: .claude/plans/archive/task-families.md. Every case builds a real row
through the real generator and, where the row is executable, runs it through
harness.run - a family whose references do not execute is not a corpus.
"""
import json
import random
import re
from pathlib import Path

import pytest

from core.ir import TaskContext
from data.gen import askact, recovery
from harness.decoys import (DECOY_OPS, MUTATING, _bank_key, _return_shape,
                            decoy_world)
from harness.run import reference_planner, run_task
from runtime.worlds import get_world

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def themed():
    from data.gen.domains import register_domains
    register_domains(str(ROOT / "data" / "gen" / "themes"))
    return True


def tool_names(task):
    """The names behind the tool symbols a reference program uses."""
    ctx = TaskContext.from_json(task["context"])
    used = set()
    for segment in task["reference"]["segments"]:
        for token in segment.split():
            if token in ctx.tools:
                used.add(ctx.tools[token].name)
    return used


# ------------------------------------------------------- B: decoy tools

def test_decoys_share_the_signature_and_differ_only_in_description():
    world = get_world("kanban")
    decoyed, added = decoy_world(world, random.Random(4))
    assert added
    by_name = {t["name"]: t for t in decoyed["tools"]}
    real = by_name["archive_card"]
    siblings = [t for t in decoyed["tools"]
                if t["name"] in added
                and [p["type"] for p in t["params"]]
                == [p["type"] for p in real["params"]]
                and t["returns"] == real["returns"]
                and t["effects"] == real["effects"]]
    assert siblings, "no decoy shares archive_card's signature"
    for sibling in siblings:
        assert sibling["desc"] != real["desc"]
        assert sibling["impl"]["op"] == "noop"


def test_a_decoy_never_describes_an_operation_its_effects_deny():
    """A copy operation carrying [SEND] is a tell, and the family is about
    the description being the only signal. Since READ and EXTERNAL joined,
    the return shape is part of it too: "list every" against a `OBJ:x` return
    is the same kind of tell."""
    world = get_world("crm")
    decoyed, added = decoy_world(world, random.Random(11), nonsense=0.0)
    by_name = {t["name"]: t for t in decoyed["tools"]}
    for name in added:
        tool = by_name[name]
        bank = DECOY_OPS[_bank_key(tool)]
        assert any(tool["desc"].startswith(text.split("{")[0])
                   for text in bank.values()), (name, tool["desc"])


def test_a_read_decoy_returns_its_declared_types_empty_value():
    """A getter decoy that hands back the real record *works*, and a decoy
    that works teaches nothing -- it is the type shortcut with extra steps.
    Each READ/EXTERNAL decoy carries the empty value of its return instead
    (harness/decoys.py, runtime/sandbox.js noop)."""
    shapes = {}
    for world_name in ("crm", "kanban"):
        decoyed, added = decoy_world(get_world(world_name), random.Random(5))
        by_name = {t["name"]: t for t in decoyed["tools"]}
        for name in added:
            tool = by_name[name]
            if MUTATING & set(tool["effects"]):
                continue
            shape = _return_shape(tool)
            assert tool["impl"] == {"op": "noop", "empty": shape}, name
            assert "entity" not in tool["impl"], (name, "returns a real record")
            shapes[shape] = shapes.get(shape, 0) + 1
    # the three the worlds actually declare (results/GROUNDING.md section 5)
    assert set(shapes) == {"LIST", "OBJ", "STR"}, shapes


def test_a_decoy_stays_distinguishable_after_prep_compacts_its_description(themed):
    """prep.py trims every schema line's description to its first sentence and
    then to 60 characters, and the description is the ONLY thing separating a
    decoy from its original. A bank entry whose first 60 characters match its
    neighbour's makes the task unanswerable rather than hard -- the reference
    answer would be unreachable from what the model is shown."""
    import sys
    sys.path.insert(0, str(ROOT / "models" / "tiny"))
    from prep import compact_line
    from data.gen.__main__ import gen_one

    checked = 0
    for seed in (101, 202, 303, 404, 505):
        task = gen_one(2, seed, False, "template", decoys=(2, 4))
        lines = {t["sym"]: f"{t['sym']} :: {t['desc']}"
                 for t in task["context"]["tools"]}
        compacted = {s: compact_line(l) for s, l in lines.items()}
        for sym in lines:
            dupes = [o for o, c in compacted.items()
                     if c == compacted[sym] and o != sym]
            assert not dupes, (task["id"], lines[sym], dupes)
            checked += 1
    assert checked > 100, checked


def test_a_decoyed_task_runs_green_and_never_calls_a_decoy(themed):
    from data.gen.__main__ import gen_one

    for seed in (101, 202, 303, 404):
        task = gen_one(2, seed, False, "template", decoys=(2, 4))
        assert "decoyed" in task["tags"]
        assert task["provenance"]["decoys"]
        assert "sandbox" in task, "a decoyed context cannot be rebuilt later"
        assert not (tool_names(task) & set(task["provenance"]["decoys"]))
        row = run_task(task, reference_planner(task))
        assert row["goal_success"], row["diagnostics"][:2]


def _swap_to_a_decoy(task: dict):
    """A planner that runs the reference program with one call redirected to
    a decoy sibling. Returns (planner, decoy name). The sibling shares the
    signature and is a noop, so the program still type-checks and runs."""
    by_sym = {t["sym"]: t for t in task["context"]["tools"]}
    decoys = set(task["provenance"]["decoys"])
    for line in "\n".join(task["reference"]["segments"]).splitlines():
        parts = line.strip().split()
        if len(parts) < 2 or parts[0] != "CALL" or parts[1] not in by_sym:
            continue
        real = by_sym[parts[1]]
        sig = ([p["type"] for p in real["params"]], real["returns"],
               real["effects"])
        for cand in task["context"]["tools"]:
            if cand["name"] not in decoys:
                continue
            if ([p["type"] for p in cand["params"]], cand["returns"],
                    cand["effects"]) != sig:
                continue
            swapped = [re.sub(rf"\bCALL {real['sym']}\b",
                              f"CALL {cand['sym']}", s)
                       for s in task["reference"]["segments"]]

            def plan(request, ctx, seg_idx, registers, _segs=swapped):
                return _segs[seg_idx] if seg_idx < len(_segs) else None

            return plan, cand["name"]
    raise AssertionError("no called tool in this task has a decoy sibling")


def test_decoy_called_records_a_grounding_miss_outcome_scoring_cannot_see(themed):
    """A decoy is never in a reference program, so calling one is a miss
    whatever the task returns -- and on an abstain task the outcome is
    identical either way (harness/metrics.py, results/GROUNDING.md section 5).
    Recorded, not gated: goal_success keeps the meaning it has in results/."""
    from data.gen.__main__ import gen_one

    task = gen_one(2, 101, False, "template", decoys=(2, 4))
    clean = run_task(task, reference_planner(task))
    assert clean["decoy_called"] is False
    assert clean["goal_success"]

    planner, name = _swap_to_a_decoy(task)
    missed = run_task(task, planner)
    assert missed["decoy_called"] is True, name
    assert name in {c["name"] for c in missed["calls"]}

    # None, not False, where there is nothing to miss, so a mean over an
    # undecoyed suite is not diluted by a column that cannot apply
    plain = gen_one(2, 101, False, "template")
    assert run_task(plain, reference_planner(plain))["decoy_called"] is None


def test_the_adversarial_names_turn_up_at_the_rate_asked_for():
    world = get_world("kanban")
    _, added = decoy_world(world, random.Random(7), per_tool=(4, 4),
                           nonsense=1.0)
    from harness.decoys import NONSENSE_NAMES
    assert added and all(n in NONSENSE_NAMES for n in added)


# --------------------------------------------------- D: ask, then act

def test_ask_row_aborts_needs_info_at_the_missing_field(themed):
    ask, _, _ = _askact_pair(themed, 3001)
    ctx = TaskContext.from_json(ask["context"])
    program = ask["reference"]["segments"][0].strip()
    assert program.startswith("ABORT NEEDS_INFO ")
    referent = program.split()[-1]
    assert referent in ctx.fields
    field = ctx.fields[referent]
    assert field.entity is not None, "the referent has to be a real field"
    assert ask["expected_status"] == "aborted"


def test_act_row_carries_the_answer_and_does_the_work(themed):
    _, act, _ = _askact_pair(themed, 3001)
    assert act["expected_status"] == "ok"
    answer = act["request"].split('"')[-2]
    values = {c.value for c in
              TaskContext.from_json(act["context"]).constants.values()}
    assert answer in values, "the answer never became a constant"
    row = run_task(act, reference_planner(act))
    assert row["goal_success"], row["diagnostics"][:2]


def test_the_pair_shares_a_world_and_a_state(themed):
    ask, act, whole = _askact_pair(themed, 4242)
    assert ask["world"] == act["world"] == whole["world"]
    assert ask["state"] == act["state"] == whole["state"]
    assert act["request"].startswith(ask["request"])


def test_the_complete_request_needs_no_question(themed):
    """The counterweight to over-abstention: the same job, stated in full,
    acted on without asking."""
    _, act, whole = _askact_pair(themed, 4242)
    assert whole["expected_status"] == "ok"
    assert "You asked" not in whole["request"]
    assert whole["reference"]["segments"] == act["reference"]["segments"]
    row = run_task(whole, reference_planner(whole))
    assert row["goal_success"], row["diagnostics"][:2]


def _askact_pair(_themed, seed):
    rng = random.Random(seed)
    from data.gen import programs
    pool = sorted(set(programs.PROFILES) - set(askact.RESERVED["worlds"]))
    for _ in range(40):
        world_name = rng.choice(pool)
        world = get_world(world_name)
        state = __import__("data.gen.worldgen", fromlist=["gen_state"]) \
            .gen_state(world_name, rng, world["now"])
        try:
            ask_s, act_s, whole_s = askact.sample_pair(
                world_name, state, world["now"], rng)
            return (askact.build_row(world_name, ask_s, state, seed, "ask",
                                     "classic", False, False),
                    askact.build_row(world_name, act_s, state, seed, "act",
                                     "classic", False, False),
                    askact.build_row(world_name, whole_s, state, seed,
                                     "complete", "classic", False, False))
        except Exception:  # noqa: BLE001 — a theme that cannot carry it
            continue
    pytest.fail("no ask/act pair in 40 attempts")


# ------------------------------------ E: recover from the code you got

@pytest.mark.parametrize("code,shape", [
    ("NOT_FOUND", "look it up by name instead"),
    ("PERMISSION_DENIED", "abort"),
    ("RATE_LIMITED", "retry"),
])
def test_the_continuation_matches_the_code(themed, code, shape):
    row = _recovery_row(themed, code)
    program = row["reference"]["segments"][0]
    if code == "NOT_FOUND":
        assert "FILTER" in program and "FIRST" in program
        assert row["expected_status"] == "ok"
    elif code == "PERMISSION_DENIED":
        assert program.strip() == "ABORT UNSUPPORTED"
        assert row["expected_status"] == "aborted"
    else:
        assert program.startswith("TRY RETRY")
        assert row["error_injection"], "a retry with nothing to retry is decor"


@pytest.mark.parametrize("code", ["NOT_FOUND", "PERMISSION_DENIED",
                                  "RATE_LIMITED"])
def test_the_row_shows_the_real_failure_and_runs_green(themed, code):
    row = _recovery_row(themed, code)
    text = row["input_text"]
    assert "PROGRAM SO FAR (ran until it FAILED at runtime):" in text
    assert f"FAILURE: RUNTIME {code}" in text
    assert not text.rstrip().endswith("PROGRAM:"), \
        "make_sft appends PROGRAM: itself; two of them is a broken prompt"
    metrics = run_task(row, reference_planner(row))
    assert metrics["goal_success"], metrics["diagnostics"][:2]


def _recovery_row(_themed, code):
    from data.gen import programs
    pool = sorted(set(programs.PROFILES) - set(recovery.RESERVED["worlds"]))
    for seed in range(9000, 9060):
        rng = random.Random(seed)
        try:
            return recovery.build_row(rng.choice(pool), seed, rng, code,
                                      "classic", False, False)
        except Exception:  # noqa: BLE001 — a theme that cannot carry it
            continue
    pytest.fail(f"no {code} recovery row in 60 attempts")


# ------------------------------------------- G: the scheduling IR probe

def test_a_filter_clause_compares_two_fields_of_the_element():
    """Spec 0.5.0: the comparison that was legal only in an IF inside a
    FOREACH (both sides operands, `r1.F9` one of them) is now a FILTER
    clause too, so the matching set is a value and not a loop body."""
    from core.pipeline import build
    from harness.authoring import resolve
    from harness.context import build_context
    from harness.run import run_sandbox
    from harness.schedule_probe import CONSTANTS
    from runtime.worlds import scheduling

    ctx, sctx = build_context(scheduling.WORLD, CONSTANTS)

    in_a_loop = """CALL @list_shifts -> r0
FOREACH r0 -> r1
  IF r1.@shift.covered LT r1.@shift.needs
    CALL @list_shifts -> r2
STOP
"""
    assert build(resolve(in_a_loop, ctx), ctx).compile_ok

    as_a_value = """CALL @list_shifts -> r0
FILTER r0 @shift.covered LT @shift.needs -> r1
COUNT r1 -> r2
RETURN r1
"""
    res = build(resolve(as_a_value, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    out = run_sandbox({"js": res.js, "state": scheduling.new_state(),
                       "tools": sctx["tools"], "fields": sctx["fields"],
                       "constants": sctx["constants"],
                       "now": scheduling.WORLD["now"], "approval": True,
                       "error_injection": [], "initial_registers": {}})
    assert out["status"] == "ok"
    assert [s["id"] for s in out["return_value"]] == ["shift_2"]


def test_a_filter_clause_tests_membership_with_in():
    """The other half of the gap, closed in spec 0.6.0: IN puts the list on
    the right, so the intersection of two calendars is a register that SORT
    and FIRST can take. CONTAINS lost its list arm in the same change."""
    from core.pipeline import build
    from harness.authoring import resolve
    from harness.context import build_context
    from harness.schedule_probe import CONSTANTS
    from runtime.worlds import scheduling

    ctx, _ = build_context(scheduling.WORLD, CONSTANTS)
    head = """CALL @list_slots $1 -> r0
FILTER r0 @slot.free EQ $2 -> r1
MAP r1 @slot.start -> r2
CALL @list_slots $0 -> r3
"""
    as_a_value = head + """FILTER r3 @slot.start IN r2 -> r4
RETURN r4
"""
    assert build(resolve(as_a_value, ctx), ctx).compile_ok

    in_a_loop = head + """FOREACH r3 -> r4
  IF r2 CONTAINS r4.@slot.start
    CALL @take_slot r4.@slot.id -> r5
STOP
"""
    res = build(resolve(in_a_loop, ctx), ctx)
    assert not res.compile_ok
    assert [d.code for d in res.diagnostics] == ["TYPE_ERROR"]


def test_the_room_ask_does_fit_and_books_the_cheapest_one():
    from core.pipeline import build
    from harness.authoring import resolve
    from harness.context import build_context
    from harness.run import run_sandbox
    from harness.schedule_probe import CONSTANTS, PROBES
    from runtime.worlds import scheduling

    ctx, sctx = build_context(scheduling.WORLD, CONSTANTS)
    room = next(p for p in PROBES if "room" in p["ask"])
    res = build(resolve(room["program"], ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    out = run_sandbox({"js": res.js, "state": scheduling.new_state(),
                       "tools": sctx["tools"], "fields": sctx["fields"],
                       "constants": sctx["constants"],
                       "now": scheduling.WORLD["now"], "approval": True,
                       "error_injection": [], "initial_registers": {}})
    assert out["status"] == "ok"
    booked = [r for r in out["state"]["entities"]["room"] if not r["free"]]
    # Mill: 8 seats, 35, the cheapest free room that fits eight
    assert {r["name"] for r in booked} == {"Mill", "Quay"}


def test_the_intersection_ask_takes_the_earliest_hour():
    """Family G's IR question, answered: the program the IR allows is now
    the program the request asked for."""
    from core.pipeline import build
    from harness.authoring import resolve
    from harness.context import build_context
    from harness.run import run_sandbox
    from harness.schedule_probe import CONSTANTS, PROBES
    from runtime.worlds import scheduling

    ctx, sctx = build_context(scheduling.WORLD, CONSTANTS)
    both = next(p for p in PROBES if "both calendars" in p["ask"])
    res = build(resolve(both["program"], ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    out = run_sandbox({"js": res.js, "state": scheduling.new_state(),
                       "tools": sctx["tools"], "fields": sctx["fields"],
                       "constants": sctx["constants"],
                       "now": scheduling.WORLD["now"], "approval": True,
                       "error_injection": [], "initial_registers": {}})
    taken = [c for c in out["calls"] if c["name"] == "take_slot"]
    # hours 2 and 5 are free in both calendars; the ask names the earliest
    assert [c["args"][0] for c in taken] == ["slot_person_1_2"]


def test_reserved_worlds_stay_out_of_every_generator():
    reserved = set(json.loads(
        (ROOT / "data" / "holdout" / "reserved.json").read_text())["worlds"])
    from data.gen.__main__ import RESERVED as gen_reserved
    assert reserved <= set(gen_reserved["worlds"])
    assert reserved <= set(askact.RESERVED["worlds"])
    assert reserved <= set(recovery.RESERVED["worlds"])
