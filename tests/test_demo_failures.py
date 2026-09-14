"""The two failures that motivated .claude/plans/demo-typed-surface.md.

Typed surface + per-request grammar, driven through the server's own
handlers. The model is not in the loop here: what is pinned is that the
grammar the demo hands the planner cannot spell the program that failed.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import dev_server  # noqa: E402
from core.pipeline import build  # noqa: E402
from core.ir import TaskContext  # noqa: E402
from harness.demo_suite import demo_state  # noqa: E402


def test_the_wrong_order_create_no_longer_typechecks_and_the_right_one_does():
    """`CALL T3 <user> <title> <date>` was what the demo produced: three
    TYPE_ERRORs, nothing ran. The argument order the types imply compiles."""
    res = dev_server.handle_kanban_prompt(
        {"request": "add a new issue for recompiling fortran",
         "state": demo_state()})
    ctx = TaskContext.from_json(res["context"])
    create = next(t for t in res["context"]["tools"] if t["name"] == "create_card")
    by_desc = {c["desc"]: c["sym"] for c in res["context"]["constants"]}
    title = next(s for d, s in by_desc.items() if d.startswith("title:"))
    due = next(s for d, s in by_desc.items() if "due date" in d)
    user = next(c["sym"] for c in res["context"]["constants"]
                if c["type"] == "ID:user")

    wrong = f"CALL {create['sym']} {user} {title} {due} -> r0\nSTOP\n"
    codes = [d.code for d in build(wrong, ctx).diagnostics]
    assert codes and set(codes) == {"TYPE_ERROR"}

    right = f"CALL {create['sym']} {title} {due} {user} -> r0\nSTOP\n"
    assert build(right, ctx).compile_ok


def test_the_grammar_cannot_spell_the_wrong_order():
    """The point of the per-request grammar: not that the wrong program is
    rejected after the fact, but that the decoder can never emit it."""
    res = dev_server.handle_kanban_prompt(
        {"request": "add a new issue for recompiling fortran",
         "state": demo_state()})
    gbnf = res["grammar"]
    create = next(t for t in res["context"]["tools"] if t["name"] == "create_card")
    rule = re.search(r"^call%s\s*::=\s*(.*)$" % create["sym"], gbnf,
                     re.MULTILINE).group(1)
    classes = re.findall(r"op[-A-Za-z0-9_]+", rule)
    assert classes[0].endswith("STR") and classes[2].endswith("ID-user")
    def rhs(name):
        return re.search(r"^%s\s*::=\s*(.*)$" % re.escape(name), gbnf,
                         re.MULTILINE).group(1)

    # an ID slot's constants are ID constants only, so the title can never
    # land there and the create call cannot be spelled in the wrong order
    id_consts = rhs(next(p for p in rhs(classes[2]).split("| ")
                         if p.strip().startswith("cc-")).strip())
    assert '"I"' in id_consts and '"S"' not in id_consts
