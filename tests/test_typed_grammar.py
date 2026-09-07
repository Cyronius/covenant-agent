"""Typed-slot CALL rules (harness/task_grammar.py, PLAN.md §5 condition C4).

One rule per tool; each parameter slot admits only registers, field accesses
of a compatible declared type, and constants of a compatible type. These
tests read the emitted grammar text: which constants a slot's class admits,
that required slots are mandatory and optional ones trailing, and that the
whole thing still compiles in llama.cpp.
"""
import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.context import build_context  # noqa: E402
from harness.task_grammar import (accepts, grammar_for_task,  # noqa: E402
                                  load_base, typed_call_rules,
                                  typed_signature)
from runtime.worlds import get_world  # noqa: E402


def _kanban_task(constants):
    world = get_world("kanban")
    ctx, _ = build_context(world, constants, random.Random(5))
    return {"context": ctx.to_json() if hasattr(ctx, "to_json") else json.loads(
        json.dumps({
            "tools": [{"sym": t.sym, "name": t.name, "desc": t.desc,
                       "params": [{"sym": p.sym, "type": _fmt(p.type),
                                   "required": p.required, "desc": p.desc}
                                  for p in t.params],
                       "returns": _fmt(t.returns) if t.returns else None,
                       "effects": t.effects} for t in ctx.tools.values()],
            "fields": [{"sym": f.sym, "entity": f.entity, "name": f.name,
                        "type": _fmt(f.type), "desc": f.desc}
                       for f in ctx.fields.values()],
            "constants": [{"sym": c.sym, "type": _fmt(c.type), "value": c.value,
                           "desc": c.desc} for c in ctx.constants.values()],
        }))}, ctx


def _fmt(t):
    from core.ir import format_type
    return format_type(t)


def _rule(text, name):
    # base rules are column-aligned (`root      ::=`); rewritten ones are not
    m = re.search(r"^%s\s*::=\s*(.*)$" % re.escape(name), text, re.MULTILINE)
    assert m, f"no rule {name}"
    return m.group(1)


def _trie_of(text, name):
    """The digit expression inside a `cc_X`/`cf_X` rule."""
    m = re.match(r'^"[FC]" \((.*)\)$', _rule(text, name))
    assert m, _rule(text, name)
    return m.group(1)


CONSTS = [
    {"type": "ID:user", "value": "user_1", "desc": "Bob"},        # C0
    {"type": "STR", "value": "ship it", "desc": "message text"},  # C1
    {"type": "TIME", "value": 1700000000, "desc": "the cutoff"},  # C2
]


def test_slot_admits_only_compatible_constants():
    task, ctx = _kanban_task(CONSTS)
    _, extra = typed_call_rules(task)
    # send_message(user: ID:user, text: STR)
    assert accepts(_trie_of(extra, "cc_ID_user"), "0")
    assert not accepts(_trie_of(extra, "cc_ID_user"), "1")
    assert not accepts(_trie_of(extra, "cc_ID_user"), "2")
    assert accepts(_trie_of(extra, "cc_STR"), "1")
    assert not accepts(_trie_of(extra, "cc_STR"), "0")
    # a TIME slot admits NOW and the TIME constant, not the user id
    op_time = _rule(extra, "op_TIME")
    assert '"NOW"' in op_time and "cc_TIME" in op_time
    assert accepts(_trie_of(extra, "cc_TIME"), "2")
    assert not accepts(_trie_of(extra, "cc_TIME"), "0")


def test_id_slot_admits_obj_fields_of_that_entity():
    task, ctx = _kanban_task(CONSTS)
    _, extra = typed_call_rules(task)
    # fields of type ID:user (card.assignee, user.id) — and nothing of type STR
    users = {int(f.sym[1:]) for f in ctx.fields.values()
             if f.type in (("ID", "user"), ("OBJ", "user"))}
    strs = {int(f.sym[1:]) for f in ctx.fields.values() if f.type == ("STR",)}
    trie = _trie_of(extra, "cf_ID_user")
    assert all(accepts(trie, str(n)) for n in users) and users
    assert not any(accepts(trie, str(n)) for n in strs)


def test_required_slots_mandatory_and_optional_trailing():
    task, ctx = _kanban_task(CONSTS)
    call_rhs, extra = typed_call_rules(task)
    send = next(t for t in ctx.tools.values() if t.name == "send_message")
    rule = _rule(extra, f"call{send.sym}")
    assert rule.startswith(f'"CALL {send.sym}" " " op_ID_user " " op_STR')
    assert rule.endswith("(arrow)?")
    lst = next(t for t in ctx.tools.values() if t.name == "list_cards")
    assert _rule(extra, f"call{lst.sym}") == f'"CALL {lst.sym}" (arrow)?'
    assert f"call{send.sym}" in call_rhs and f"call{lst.sym}" in call_rhs


def test_registers_stay_open_in_every_slot():
    task, _ = _kanban_task(CONSTS)
    _, extra = typed_call_rules(task)
    for line in extra.splitlines():
        if line.startswith("op_"):
            assert line.split("::=")[1].strip().startswith("reg")


def test_typed_grammar_replaces_call_and_keeps_the_rest():
    task, _ = _kanban_task(CONSTS)
    base = load_base()
    plain = grammar_for_task(task, base)
    typed = grammar_for_task(task, base, typed=True)
    assert _rule(plain, "call") != _rule(typed, "call")
    for name in ("root", "instr", "filterin", "operand", "reg", "field", "const"):
        assert _rule(plain, name) == _rule(typed, name)


def test_typed_signature_distinguishes_types_not_just_symbols():
    a, _ = _kanban_task([{"type": "STR", "value": "x", "desc": "x"}])
    b, _ = _kanban_task([{"type": "TIME", "value": 1, "desc": "x"}])
    assert typed_signature(a) != typed_signature(b)


CROWDED = ROOT / "data" / "holdout" / "e_crowded_v2.jsonl"


@pytest.mark.skipif(not CROWDED.exists(), reason="holdout suite not present")
def test_typed_grammar_compiles_in_llama_cpp_on_a_crowded_task():
    from llama_cpp import LlamaGrammar
    base = load_base()
    with open(CROWDED) as f:
        for i, line in enumerate(f):
            task = json.loads(line)
            LlamaGrammar.from_string(grammar_for_task(task, base, typed=True),
                                     verbose=False)
            if i >= 4:
                break
