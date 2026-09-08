"""Spec 0.4.0 'standard library': MOST / LEAST (with the candidate list that
keeps LEAST honest about zero counts), the EMPTY condition, and normalized
STR comparison. Each named composite earned its place by a failing task
(results/R4.md §3): the argmax pair and the `IF r1 EQ C6` emptiness guess."""
import random

from core.parser import parse
from core.pipeline import build
from harness.authoring import resolve
from harness.context import build_context
from harness.run import run_sandbox
from harness import task_grammar
from runtime.worlds import get_world


def _kanban(constants=()):
    world = get_world("kanban")
    ctx, sctx = build_context(world, list(constants), random.Random(3))
    return world, ctx, sctx


def _run(world, ctx, sctx, text, state=None):
    res = build(text, ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    return run_sandbox({"js": res.js, "state": state or world["default_state"],
                        "tools": sctx["tools"], "fields": sctx["fields"],
                        "constants": sctx["constants"],
                        "now": "2026-01-01T00:00:00Z"})


# -- parser -----------------------------------------------------------------

def test_most_least_parse_and_render():
    prog, diags = parse("MOST r1 F3 -> r2\nLEAST r1 F3 r0 -> r4\nSTOP\n")
    assert diags == []
    a, b = prog.body[0], prog.body[1]
    assert (a.src.n, a.field, a.cands, a.dst.n, a.least) == (1, "F3", None, 2, False)
    assert b.cands.n == 0 and b.least
    _, d = parse("MOST r1 -> r2\n")
    assert d and d[0].code == "PARSE_ERROR"


def test_empty_condition_parses():
    prog, diags = parse("IF EMPTY r1\n  STOP\nIF NOT EMPTY r1 AND r2 EQ C0\n  STOP\nSTOP\n")
    assert diags == []
    c = prog.body[0].cond.clauses[0]
    assert c.cmp == "EMPTY" and c.right is None and str(c) == "EMPTY r1"
    assert str(prog.body[1].cond.clauses[0]) == "NOT EMPTY r1"
    _, d = parse("FILTER r0 EMPTY F1 -> r1\n")   # FILTER clauses stay binary
    assert d and d[0].code == "PARSE_ERROR"


# -- typecheck --------------------------------------------------------------

def test_most_types_as_the_field():
    _, ctx, _ = _kanban()
    res = build(resolve("CALL @list_cards -> r0\nMOST r0 @card.assignee -> r1\n"
                        "CALL @get_user r1 -> r2\nSTOP\n", ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    assert res.final_env["r1"] == "ID:user"


def test_least_candidate_list_must_match_the_field_entity():
    _, ctx, _ = _kanban()
    ok = build(resolve("CALL @list_users -> r0\nCALL @list_cards -> r1\n"
                       "LEAST r1 @card.assignee r0 -> r2\nSTOP\n", ctx), ctx)
    assert ok.compile_ok, ok.rendered_diagnostics()
    bad = build(resolve("CALL @list_cards -> r0\nCALL @list_cards -> r1\n"
                        "LEAST r1 @card.assignee r0 -> r2\nSTOP\n", ctx), ctx)
    assert not bad.compile_ok
    assert any(d.startswith("TYPE_ERROR") and "LIST OBJ:user" in d
               for d in bad.rendered_diagnostics())
    # a candidate list on a non-ID field is a type error too
    bad2 = build(resolve("CALL @list_users -> r0\nCALL @list_cards -> r1\n"
                         "MOST r1 @card.title r0 -> r2\nSTOP\n", ctx), ctx)
    assert not bad2.compile_ok


def test_empty_requires_a_list():
    _, ctx, _ = _kanban()
    ok = build(resolve("CALL @list_cards -> r0\nIF EMPTY r0\n  STOP\nSTOP\n", ctx), ctx)
    assert ok.compile_ok
    bad = build(resolve("CALL @list_cards -> r0\nCOUNT r0 -> r1\nIF EMPTY r1\n  STOP\nSTOP\n", ctx), ctx)
    assert not bad.compile_ok and "TYPE_ERROR" in bad.rendered_diagnostics()[0]


# -- runtime ----------------------------------------------------------------

def _state_with_assignees(world, assignees):
    st = {k: (dict(v) if isinstance(v, dict) else v) for k, v in world["default_state"].items()}
    ents = dict(st["entities"])
    cards = [dict(c) for c in ents["card"]]
    for c, a in zip(cards, assignees):
        c["assignee"] = a
    ents["card"] = cards
    st["entities"] = ents
    return st


def test_most_picks_the_commonest_value():
    world, ctx, sctx = _kanban()
    users = [u["id"] for u in world["default_state"]["entities"]["user"]]
    n = len(world["default_state"]["entities"]["card"])
    # user[1] gets the majority, user[0] the rest
    state = _state_with_assignees(world, [users[1]] * (n - 1) + [users[0]])
    out = _run(world, ctx, sctx,
               resolve("CALL @list_cards -> r0\nMOST r0 @card.assignee -> r1\nRETURN r1\n", ctx),
               state)
    assert out["status"] == "ok" and out["return_value"] == users[1]


def test_least_with_candidates_counts_zero():
    """The trap the candidate list exists for: a user with no cards is the
    true 'fewest' and is absent from the card list."""
    world, ctx, sctx = _kanban()
    users = [u["id"] for u in world["default_state"]["entities"]["user"]]
    n = len(world["default_state"]["entities"]["card"])
    state = _state_with_assignees(world, [users[0]] * n)   # everyone else: 0
    prog = resolve("CALL @list_users -> r0\nCALL @list_cards -> r1\n"
                   "LEAST r1 @card.assignee r0 -> r2\nRETURN r2\n", ctx)
    out = _run(world, ctx, sctx, prog, state)
    assert out["status"] == "ok"
    assert out["return_value"] == users[1]      # first zero-count candidate
    # without candidates, least-among-present is the only user present
    out2 = _run(world, ctx, sctx,
                resolve("CALL @list_cards -> r1\nLEAST r1 @card.assignee -> r2\nRETURN r2\n", ctx),
                state)
    assert out2["return_value"] == users[0]


def test_most_on_empty_list_is_null():
    world, ctx, sctx = _kanban([{"type": "STR", "value": "nobody", "desc": "x"}])
    out = _run(world, ctx, sctx,
               resolve("CALL @list_cards -> r0\nFILTER r0 @card.title EQ $0 -> r1\n"
                       "MOST r1 @card.assignee -> r2\nRETURN r2\n", ctx))
    assert out["status"] == "ok" and out["return_value"] is None


def test_empty_branches_on_the_empty_match():
    world, ctx, sctx = _kanban([{"type": "STR", "value": "Cyrus", "desc": "the name"}])
    prog = resolve("CALL @list_users -> r0\nFILTER r0 @user.name EQ $0 -> r1\n"
                   "IF EMPTY r1\n  ABORT NOT_FOUND $0\nSTOP\n", ctx)
    out = _run(world, ctx, sctx, prog)
    assert out["status"] == "aborted" and out["reason"] == "NOT_FOUND"


def test_str_compare_is_case_folded_and_trimmed():
    world, ctx, sctx = _kanban([{"type": "STR", "value": " bob ", "desc": "the name"}])
    names = [u["name"] for u in world["default_state"]["entities"]["user"]]
    assert "Bob" in names
    prog = resolve("CALL @list_users -> r0\nFILTER r0 @user.name EQ $0 -> r1\n"
                   "COUNT r1 -> r2\nRETURN r2\n", ctx)
    out = _run(world, ctx, sctx, prog)
    assert out["status"] == "ok" and out["return_value"] == 1
    prog = resolve("CALL @list_users -> r0\nFILTER r0 @user.name CONTAINS $0 -> r1\n"
                   "COUNT r1 -> r2\nRETURN r2\n", ctx)
    assert _run(world, ctx, sctx, prog)["return_value"] == 1


# -- grammar ----------------------------------------------------------------

def test_grammar_carries_stdlib_and_the_switch_removes_it():
    from llama_cpp import LlamaGrammar
    _, ctx, _ = _kanban([{"type": "STR", "value": "x", "desc": "x"}])
    task = {"context": ctx.to_json()}
    g = task_grammar.grammar_for_task(task)
    assert "mostin" in g and '"EMPTY " reg' in g
    LlamaGrammar.from_string(g, verbose=False)
    g0 = task_grammar.grammar_for_task(task, stdlib=False)
    assert "| mostin" not in g0 and "EMPTY" not in g0
    LlamaGrammar.from_string(g0, verbose=False)
    assert not task_grammar.check_gbnf_names(g)
