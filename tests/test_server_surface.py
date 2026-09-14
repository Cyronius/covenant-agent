"""The demo serves the surface the checkpoint was trained on.

The browser demo ran classic symbols against a static grammar while every
eval since S3 used typed symbols and a per-request grammar whose slots admit
only type-compatible constants. Same model, same board, same request: the
demo could spell `CALL T3 C0 C15 C16` (four TYPE_ERRORs) and the eval could
not. See .claude/plans/demo-typed-surface.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import dev_server  # noqa: E402
from harness.demo_suite import demo_state  # noqa: E402
from harness.task_grammar import accepts  # noqa: E402
import re  # noqa: E402


def _rule(gbnf: str, name: str) -> str:
    m = re.search(r"^%s\s*::=\s*(.*)$" % re.escape(name), gbnf, re.MULTILINE)
    assert m, f"no rule {name}"
    return m.group(1)


def _prompt(request: str) -> dict:
    return dev_server.handle_kanban_prompt(
        {"request": request, "state": demo_state()})


def test_kanban_prompt_is_typed():
    res = _prompt("add a new issue for recompiling fortran")
    letters = {c["sym"][0] for c in res["context"]["constants"]}
    assert letters <= set("SNBDI"), f"classic symbols leaked: {letters}"
    # the legend has to match the symbols, or the prompt teaches one surface
    # and the table shows another
    assert "F3 EQ S0" in res["system"] and "F3 EQ C0" not in res["system"]


def test_kanban_prompt_carries_a_per_request_grammar():
    """Each CALL slot admits only type-compatible constants, which is what
    makes `CALL T3 <user id> <title> <date>` unspellable rather than merely
    wrong. The fallback grammar has one open `call` rule and cannot say it."""
    res = _prompt("add a new issue for recompiling fortran")
    gbnf, ctx = res["grammar"], res["context"]
    create = next(t for t in ctx["tools"] if t["name"] == "create_card")
    assert [p["type"] for p in create["params"]] == ["STR", "TIME", "ID:user"]
    classes = re.findall(r"op[-A-Za-z0-9_]+", _rule(gbnf, f"call{create['sym']}"))
    assert len(classes) == 3, classes
    title_consts = _rule(gbnf, _rule(gbnf, classes[0]).split("| ")[-1].strip())
    assert '"S"' in title_consts and '"I"' not in title_consts, title_consts
    # the title the serializer extracted is one of the constants it admits
    title = next(c["sym"] for c in ctx["constants"]
                 if c["desc"].startswith("title:"))
    assert accepts(title_consts.split('"S" ')[-1], title[1:])


def test_rpg_prompt_is_typed_too():
    from runtime.worlds import rpg
    res = dev_server.handle_rpg_prompt({"state": rpg.new_state("keep")})
    letters = {c["sym"][0] for c in res["context"]["constants"]}
    assert letters <= set("SNBDI"), f"classic symbols leaked: {letters}"
    assert "callT" in res["grammar"]


def test_plan_compiles_a_supplied_grammar_once():
    """The planner caches by grammar text: a board's symbol table only
    changes when the board does, and compiling GBNF is not free."""
    cache = dev_server.PLANNER._grammar_cache
    cache.clear()
    gbnf = _prompt("assign card 6 to Bob")["grammar"]
    a = dev_server.PLANNER.grammar_for(gbnf)
    b = dev_server.PLANNER.grammar_for(gbnf)
    assert a is b and len(cache) == 1
