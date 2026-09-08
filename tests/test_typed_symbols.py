"""Spec 0.4.0 §2.1–2.3: typed constant letters, string kinds, schema enum
constants, and re-rendering a stored 0.3.x task under the new surface
(harness/retype.py). Scoring must not move: the same reference passes
under either symbol table."""
import json
import random
from pathlib import Path

from core.parser import parse
from core.pipeline import build
from harness.authoring import resolve
from harness.context import (build_context, enum_constants, is_typed,
                             serialize_context)
from harness.retype import infer_kind, retype_task
from harness.run import reference_planner, run_task
from harness.taskbuild import build_task
from harness import task_grammar
from runtime.worlds import get_world

ROOT = Path(__file__).resolve().parent.parent

CONSTS = [{"type": "STR", "value": "Bob", "desc": "the assignee named"},
          {"type": "ID:card", "value": "card_1", "desc": "card 1"},
          {"type": "TIME", "value": 1700000000, "desc": "the cutoff"},
          {"type": "BOOL", "value": True, "desc": "true"},
          {"type": "STR", "value": "done", "desc": "the status"}]


def test_typed_letters_number_per_letter_and_keep_index():
    world = get_world("kanban")
    ctx, sctx = build_context(world, CONSTS, random.Random(1), symbols="typed")
    syms = [c.sym for c in ctx.constants.values()]
    assert syms == ["S0", "I0", "D0", "B0", "S1"]
    assert [c.index for c in ctx.constants.values()] == [0, 1, 2, 3, 4]
    assert sctx["constants"]["S1"] == "done" and is_typed(ctx)
    # authoring $n follows the index, not the letter
    assert resolve("FILTER r0 @user.name EQ $0 -> r1\nLET $4 -> r2\n", ctx).startswith(
        "FILTER r0 ")
    assert " EQ S0 -> r1\nLET S1 -> r2" in resolve(
        "FILTER r0 @user.name EQ $0 -> r1\nLET $4 -> r2\n", ctx)


def test_typed_context_round_trips_through_json():
    world = get_world("kanban")
    ctx, _ = build_context(world, CONSTS, random.Random(1), symbols="typed",
                           enums=True)
    from core.ir import TaskContext
    back = TaskContext.from_json(ctx.to_json())
    assert {c.sym: (c.kind, c.index) for c in back.constants.values()} == \
        {c.sym: (c.kind, c.index) for c in ctx.constants.values()}


def test_enum_constants_come_from_the_schema_without_duplicates():
    world = get_world("kanban")
    ctx, sctx = build_context(world, CONSTS, random.Random(1), symbols="typed",
                              enums=True)
    enums = [c for c in ctx.constants.values() if c.kind.startswith("enum:")]
    values = sorted(c.value for c in enums)
    assert values == ["doing", "done", "todo"]          # "done" not duplicated
    done = next(c for c in enums if c.value == "done")
    assert done.index == 4 and done.desc == "the status"   # the request's own
    todo = next(c for c in enums if c.value == "todo")
    assert todo.index is None and '"todo"' in todo.desc
    assert sctx["constants"][todo.sym] == "todo"


def test_serialized_tool_lines_show_slot_letters():
    world = get_world("kanban")
    ctx, _ = build_context(world, CONSTS, random.Random(1), symbols="typed",
                           enums=True)
    text = serialize_context("req", ctx)
    send = next(t for t in ctx.tools.values() if t.name == "send_message")
    line = next(l for l in text.splitlines() if l.startswith(send.sym + " "))
    assert "(I:user=F" in line and " S=F" in line
    assert "S0 STR :: the assignee named" in text or "S0 STR name" in text
    assert "enum card.status ::" in text
    # classic rendering is byte-for-byte what it was
    cctx, _ = build_context(world, CONSTS, random.Random(1))
    ctext = serialize_context("req", cctx)
    assert "C0 STR :: the assignee named" in ctext and "=F" not in ctext


def test_parser_accepts_both_symbol_forms():
    prog, d = parse("CALL T0 S1 I0 -> r0\nABORT NOT_FOUND S0\n")
    assert d == [] and prog.body[1].refs == ["S0"]
    prog, d = parse("CALL T0 C1 -> r0\nABORT NOT_FOUND C0\n")
    assert d == []
    _, d = parse("CALL T0 X1 -> r0\n")
    assert d and d[0].code == "PARSE_ERROR"


def test_reference_passes_under_either_table():
    world = get_world("kanban")
    consts = CONSTS[:2] + [{"type": "STR", "value": "hi", "desc": "message text"}]
    ctx, sctx = build_context(world, consts, random.Random(5))
    seg = resolve("CALL @list_users -> r0\n"
                  "FILTER r0 @user.name EQ $0 -> r1\n"
                  "FIRST r1 -> r2\n"
                  "CALL @send_message r2.@user.id $2\nSTOP\n", ctx)
    task = build_task(task_id="t", level=3, world_name="kanban",
                      request="message bob about card 1", constants=consts,
                      segments=[seg], prebuilt=(ctx, sctx))
    assert run_task(task, reference_planner(task))["goal_success"]
    for enums, kinds in ((False, False), (True, True)):
        rt = retype_task(task, world, symbols="typed", enums=enums, kinds=kinds)
        assert rt["input_text"] == serialize_context(rt["request"], __import__("core.ir", fromlist=["TaskContext"]).TaskContext.from_json(rt["context"]))
        assert "C0" not in rt["reference"]["segments"][0]
        assert run_task(rt, reference_planner(rt))["goal_success"]
        assert rt["expected_state"] == task["expected_state"]


def test_infer_kind_on_stored_constants():
    assert infer_kind({"type": "STR", "desc": "the assignee named", "value": "Bob"}) == "name"
    assert infer_kind({"type": "STR", "desc": "message text", "value": "hi"}) == "text"
    assert infer_kind({"type": "STR", "desc": "x", "value": "Copy of {0}"}) == "text"
    assert infer_kind({"type": "STR", "desc": "x", "value": "done",
                       "kind": "enum:card.status"}) == "enum:card.status"
    assert infer_kind({"type": "TIME", "desc": "cutoff", "value": 1}) == ""


def test_kind_grammar_restricts_enum_slots():
    from llama_cpp import LlamaGrammar
    world = get_world("kanban")
    ctx, sctx = build_context(world, CONSTS, random.Random(1), symbols="typed",
                              enums=True)
    task = {"context": ctx.to_json()}
    g = task_grammar.grammar_for_task(task, kinds=True)
    assert not task_grammar.check_gbnf_names(g)
    LlamaGrammar.from_string(g, verbose=False)
    status = next(f for f in ctx.fields.values() if f.entity == "card" and f.name == "status")
    name = next(f for f in ctx.fields.values() if f.entity == "user" and f.name == "name")
    enum_syms = sorted(c.sym for c in ctx.constants.values() if c.kind.startswith("enum:"))
    # the status field's operand class admits exactly the enum constants
    for s in enum_syms:
        assert task_grammar.accepts(_cc(g, f"f{status.sym[1:]}"), s)
    assert not task_grammar.accepts(_cc(g, f"f{status.sym[1:]}"), "S0")
    # the name field's class admits S0 (name) and none of the enum values
    assert task_grammar.accepts(_cc(g, f"f{name.sym[1:]}"), "S0")
    for s in enum_syms:
        assert not task_grammar.accepts(_cc(g, f"f{name.sym[1:]}"), s)
    # const rule is per letter
    const_rule = next(l for l in g.splitlines() if l.startswith("const ::="))
    assert '"S" (' in const_rule and '"I" (' in const_rule and '"C" (' not in const_rule


def _cc(grammar: str, tag: str) -> str:
    line = next(l for l in grammar.splitlines() if l.startswith(f"cc-{tag} ::="))
    return line.split("::=", 1)[1]


def test_retype_the_stored_slice():
    """Every task of the 25-task slice re-renders and its reference still
    passes under typed + enums + kinds (the step-0 A/B precondition)."""
    from data.gen.domains import register_domains
    register_domains(str(ROOT / "data" / "gen" / "themes"))
    n = 0
    for name in ("_smoke8b_known", "_smoke8b_demo"):
        for line in open(ROOT / "data" / "holdout" / f"{name}.jsonl", encoding="utf-8"):
            t = json.loads(line)
            rt = retype_task(t, get_world(t["world"]), symbols="typed", enums=True, kinds=True)
            assert run_task(rt, reference_planner(rt))["goal_success"], t["id"]
            n += 1
    assert n == 17
