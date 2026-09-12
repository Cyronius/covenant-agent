"""Per-task GBNF (harness/task_grammar.py).

The regression under test: `num ::= [0-9] [0-9]?` in agent_core.gbnf made
every symbol above F99 undecodable, so on crowded contexts the correct
program could not be emitted at all. `accepts()` matches a string against
the emitted grammar expression itself, so these assert on the grammar text
rather than on the builder's internals.
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.task_grammar import (accepts, digit_trie_expr,  # noqa: E402
                                  grammar_for_task, load_base, symbol_rules,
                                  symbol_signature)


def _rule(gbnf: str, name: str) -> str:
    m = re.search(r"^%s\s*::=(.*)$" % name, gbnf, re.MULTILINE)
    assert m, "no rule %s in\n%s" % (name, gbnf)
    return m.group(1).strip()


def _expr(rule_rhs: str) -> str:
    """Strip the `"F" (...)` wrapper down to the digit expression."""
    m = re.match(r'^"[TFC]"\s*\((.*)\)$', rule_rhs, re.S)
    assert m, rule_rhs
    return m.group(1)


# -- the trie ------------------------------------------------------------

def test_trie_accepts_exactly_the_given_numbers():
    expr = digit_trie_expr([0, 1, 10, 142, 157])
    for s in ["0", "1", "10", "142", "157"]:
        assert accepts(expr, s), s
    # 14 and 15 are the two-digit prefixes the old grammar forced the sampler
    # to commit to; they are not declared symbols and must be rejected.
    for s in ["14", "15", "2", "11", "1420", "", "01"]:
        assert not accepts(expr, s), s


def test_trie_handles_three_digit_symbols():
    expr = digit_trie_expr(range(200))
    assert accepts(expr, "157")
    assert accepts(expr, "199")
    assert not accepts(expr, "200")


def test_empty_symbol_set_falls_back_to_open_form():
    # a Level 0 task can declare no constants; the rule must still be usable
    assert digit_trie_expr([]) == "[0-9] [0-9]? [0-9]?"


# -- rule assembly -------------------------------------------------------

def test_symbol_rules_cover_each_kind():
    rules = symbol_rules(["T0", "T7"], ["F3", "F157"], ["C0"])
    assert accepts(_expr(rules["tool"]), "7")
    assert not accepts(_expr(rules["tool"]), "8")
    assert accepts(_expr(rules["field"]), "157")
    assert not accepts(_expr(rules["field"]), "15")
    assert accepts(_expr(rules["const"]), "0")
    assert not accepts(_expr(rules["const"]), "1")


def test_grammar_for_task_keeps_the_base_rules():
    base = load_base()
    task = {"context": {"tools": [{"sym": "T0"}], "fields": [{"sym": "F0"}],
                        "constants": [{"sym": "C0"}]}}
    gbnf = grammar_for_task(task, base)
    for rule in ("root", "instr", "call", "operand", "reg", "num"):
        assert _rule(gbnf, rule) == _rule(base, rule)
    assert _rule(gbnf, "field") != _rule(base, "field")


def test_symbol_signature_is_shared_by_identical_symbol_tables():
    a = {"context": {"tools": [{"sym": "T1"}, {"sym": "T0"}],
                     "fields": [{"sym": "F2"}], "constants": []}}
    b = {"context": {"tools": [{"sym": "T0"}, {"sym": "T1"}],
                     "fields": [{"sym": "F2"}], "constants": []}}
    c = {"context": {"tools": [{"sym": "T0"}],
                     "fields": [{"sym": "F2"}], "constants": []}}
    assert symbol_signature(a) == symbol_signature(b)
    assert symbol_signature(a) != symbol_signature(c)


# -- against the suite that exposed the bug ------------------------------

CROWDED = ROOT / "data" / "holdout" / "e_crowded_v2.jsonl"


@pytest.mark.skipif(not CROWDED.exists(), reason="holdout suite not present")
def test_every_reference_program_symbol_is_decodable():
    """The 122 tasks whose reference needs an F>=100 are the whole E-crowded
    gap; under the per-task grammar every reference symbol must be spellable
    and every undeclared one unspellable."""
    base = load_base()
    checked = 0
    with open(CROWDED) as f:
        for line in f:
            task = json.loads(line)
            gbnf = grammar_for_task(task, base)
            exprs = {k: _expr(_rule(gbnf, {"T": "tool", "F": "field",
                                           "C": "const"}[k]))
                     for k in "TFC"}
            declared = {k: set() for k in "TFC"}
            for kind, key in (("T", "tools"), ("F", "fields"),
                              ("C", "constants")):
                for d in task["context"].get(key, []):
                    declared[kind].add(d["sym"][1:])
            program = "".join(task["reference"]["segments"])
            for m in re.finditer(r"\b([TFC])(\d+)\b", program):
                kind, num = m.group(1), m.group(2)
                assert accepts(exprs[kind], num), \
                    "%s%s in %s is not decodable" % (kind, num, task["id"])
                checked += 1
            # an undeclared symbol one past the top must be rejected
            for kind in "TFC":
                if declared[kind]:
                    top = str(max(int(n) for n in declared[kind]) + 1)
                    assert not accepts(exprs[kind], top)
    assert checked > 1000


# -- NULL in call slots (results/RPG.md) ---------------------------------

def _typed_rpg_grammar():
    import random

    from harness.context import build_context
    from runtime.worlds import get_world
    ctx = build_context(get_world("rpg"),
                        [{"type": "ID:item", "value": "item_1",
                          "desc": "a potion"}], random.Random(3))[0]
    tools = {t.name: t.sym for t in ctx.tools.values()}
    return grammar_for_task({"context": ctx.to_json()}, load_base(),
                            typed=True), tools


def test_typed_required_slot_admits_no_null():
    """`use_item` takes a required item and an optional target. NULL belongs
    only in the optional slot: offering it in the required one made NULL the
    single legal operand for a tool whose constant was not in view."""
    gbnf, tools = _typed_rpg_grammar()
    rule = _rule(gbnf, "call%s" % tools["use_item"])
    ops = re.findall(r"op-[A-Za-z0-9-]+", rule)
    assert len(ops) == 2, rule
    assert '"NULL"' not in _rule(gbnf, ops[0])
    assert '"NULL"' in _rule(gbnf, ops[1])


def test_typed_sole_required_slot_admits_no_null():
    gbnf, tools = _typed_rpg_grammar()
    rule = _rule(gbnf, "call%s" % tools["interact"])
    ops = re.findall(r"op-[A-Za-z0-9-]+", rule)
    assert len(ops) == 1, rule
    # no door is in view, so this slot has no constant at all -- it must
    # still not fall back to NULL
    assert '"NULL"' not in _rule(gbnf, ops[0])


# -- a field on the right of a FILTER clause (spec 0.5.0) ------------------

def test_kind_clause_admits_type_compatible_sibling_fields():
    """Under a kinds grammar each field's clause may end in a sibling field
    of the same entity and a compatible type; nothing else is spellable
    there, or the change is invisible to a constrained decode."""
    import random

    from harness.context import build_context
    from runtime.worlds import get_world
    ctx = build_context(get_world("scheduling"), [], random.Random(3))[0]
    syms = {(f.entity, f.name): f.sym for f in ctx.fields.values()
            if f.entity}
    gbnf = grammar_for_task({"context": ctx.to_json()}, load_base(),
                            kinds=True)
    covered, needs = syms[("shift", "covered")], syms[("shift", "needs")]
    seats = syms[("room", "seats")]
    sib = _expr(_rule(gbnf, "sf-f%s" % covered[1:]))
    assert accepts(sib, needs[1:])
    assert not accepts(sib, seats[1:])  # another entity
    assert not accepts(sib, covered[1:])
    assert "sf-f%s" % covered[1:] in _rule(gbnf, "clause")
    # the base grammar carries the open form
    assert '(operand | field)' in _rule(load_base(), "clause")
