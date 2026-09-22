"""Is a FILTER clause grounded in the request, or is it the corpus's reflex?
(plan step 2a; `.claude/plans/general-agent-plan.md` §2B)

97.2% of training rows that call a list tool then `FILTER` (28,154 of
28,971), so the model writes the slot whether or not the request fills it. A
one-predicate request comes back padded to two — the demo's own
`delinquent EQ true AND delinquent EQ false` — because no generated program
ever had that slot empty.

A clause is counted as padding when **both** halves of its grounding are
missing:

  - the request never names its field (`card.archived` needs "archive"
    somewhere in the request, in some form), and
  - the request never spells the value it compares against — neither the
    constant's value ("done", "Bob") nor the description the serializer
    gave it appears in the request.

Both halves matter. A clause on a field the request does name is the model
reading the request; a clause whose value came out of the request is too,
even where the field name never appears in so many words ("archive the done
ones" names no field but spells the status). What this flags is the clause
with neither: `delinquent EQ false` on "list every customer".

Deliberately *not* the test: whether the constant is request-derived by its
`index`. Every constant in both demo suites carries an integer index
(measured, 2026-09-21: 1,175 and 980 of them, none `None`), so that test
would make this column read 0% everywhere — the same defect
`abort_check.py:66` names: "a check with no discriminative power is not a
check."

**This column has a floor, and it is not zero.** Some correct clauses are
semantic inference the wording never spells: "show me the overdue invoices"
filters `NOT paid EQ true`, "whoever has the most unfinished cards on the
board" filters `archived EQ false`, and `L9_projects_cleanup`'s "clean up
the old projects" is *tagged* ambiguous precisely because "old" names no
cutoff. Measured on the reference programs, which are correct by
construction (2026-09-21):

    e_demo_requests     0 of 15 refs with a FILTER    0.0%
    e_db_requests       5 of 40                      12.5%
    curriculum_tasks    4 of 24                      16.7%

That is the null. A model's rate means something only against it — the same
lesson `chance_tool_sig` taught the grounding numbers (`results/R9.md` §3).

Recorded, not gated, like `decoy_called` and `abort_referent_match`: this
adds a column, it never changes `goal_success`, so no number already in
`results/` moves.

  from harness.filter_check import padded_clauses
  pads = padded_clauses(program, ctx, task["request"])   # [str], possibly []
"""
from __future__ import annotations

import re
from typing import List

from core.ir import (Clause, Const, ElemField, Filter, Foreach, If, Parallel,
                     Program, TaskContext, Try)

_WORD = re.compile(r"[a-z0-9]+")
# Field names are snake_case and often compounds of a word the request does
# use ("last_activity" against "inactive since"), so a field counts as named
# when any of its parts is a request word, and short parts are dropped
# rather than matching everything ("id", "at", "by").
_MIN_PART = 3
# Function words carry no grounding and, under the substring rule below,
# match far too much: a constant described as "the completed status" read as
# grounded by a request saying "what cards are there", because "the" is
# inside "there".
_STOP = frozenset("""
the and for are was its with that this from have has all any not but you
your our their there here when what which who how why one two out into
each per its it's about than then them they some more most other another
""".split())


def _request_words(request: str) -> set:
    return {w for w in _WORD.findall(request.casefold()) if w not in _STOP}


def _matches_a_request_word(text: str, words: set) -> bool:
    """Any word of `text`, at least `_MIN_PART` long, against any request
    word, either containing the other: "archive" against "archived", "due"
    against "overdue", "Bob" against "Bob's". Deliberately lenient — this
    column counts a defect, so a false *positive* (calling a grounded clause
    padding) is the costlier error."""
    for part in _WORD.findall(text.casefold()):
        if len(part) < _MIN_PART or part in _STOP:
            continue
        for w in words:
            if part == w or part in w or w in part:
                return True
    return False


def _named_by_request(name: str, words: set) -> bool:
    if not [p for p in _WORD.findall(name.casefold()) if len(p) >= _MIN_PART]:
        return True      # nothing checkable in the name: do not call it padding
    return _matches_a_request_word(name, words)


def _spelled_by_request(operand, ctx: TaskContext, words: set) -> bool:
    """True when the request spells the value this clause compares against.

    `NOW` counts: "overdue" and "due this week" are the request asking for a
    comparison against the clock, and no constant carries that.
    """
    if not isinstance(operand, Const):
        # NOW, an int literal, a register, a field of the element: none of
        # these is an unrequested enum value reached for to fill the slot.
        return True
    c = ctx.constants.get(operand.sym)
    if c is None:
        return False
    return any(_matches_a_request_word(t, words)
               for t in (str(c.value), c.desc or ""))


def _walk_filters(body: list):
    for instr in body:
        if isinstance(instr, Filter):
            yield instr
        elif isinstance(instr, Foreach):
            yield from _walk_filters(instr.body)
        elif isinstance(instr, If):
            yield from _walk_filters(instr.then)
            if instr.els is not None:
                yield from _walk_filters(instr.els)
        elif isinstance(instr, Try):
            yield from _walk_filters(instr.body)
        elif isinstance(instr, Parallel):
            continue      # CALL lines only (spec §4)


def _clause_padded(cl: Clause, ctx: TaskContext, words: set) -> bool:
    if not isinstance(cl.left, str):
        return False      # an IF-shaped clause, not a FILTER field clause
    field = ctx.fields.get(cl.left)
    if field is None:
        return False      # UNKNOWN_FIELD is the typechecker's to report
    if _named_by_request(field.name, words):
        return False
    if isinstance(cl.right, ElemField):
        # spec 0.5.0: a second field of the same element. Grounded when
        # either side is named.
        other = ctx.fields.get(cl.right.sym)
        return not (other is not None and _named_by_request(other.name, words))
    return not _spelled_by_request(cl.right, ctx, words)


def padded_clauses(program: Program, ctx: TaskContext,
                   request: str) -> List[str]:
    """Every FILTER clause that names neither a field the request names nor a
    value it supplied, rendered for the metrics row."""
    words = _request_words(request or "")
    out: List[str] = []
    for instr in _walk_filters(program.body):
        for cl in instr.pred.clauses:
            if _clause_padded(cl, ctx, words):
                out.append(str(cl))
    return out
