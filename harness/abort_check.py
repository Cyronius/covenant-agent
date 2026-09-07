"""Is an abort founded? (spec §4 referents, §9 `ABORT_UNFOUNDED`)

An `ABORT` with a referent is a claim about the task the harness can test
without consulting the reference: `NOT_FOUND C2` says nothing matches C2's
value, `NEEDS_INFO F3` says no constant supplies F3, `AMBIGUOUS a b` says
there are several candidates. When the claim is false the checker renders an
`ABORT_UNFOUNDED` line in the §9 vocabulary, and a repair round can consume
it exactly like a `TYPE_ERROR`. It confirms or denies what the program
asserted and reveals nothing about the reference.

A referent-less abort has nothing to check and returns None: this module can
never make a bare abort look wrong, only a specific one.

  from harness.abort_check import check_abort
  msg = check_abort(ctx, task["state"], "NOT_FOUND", ["C2"])   # str | None
"""
from __future__ import annotations

from typing import List, Optional

from core.ir import TaskContext, format_type

NUMERIC = {"INT", "FLOAT"}


def _compatible(want, have) -> bool:
    if want[0] == "ID" and have[0] == "ID":
        return want[1] == have[1]
    if want[0] in NUMERIC and have[0] in NUMERIC:
        return True
    return format_type(want) == format_type(have)


def _records(state: dict):
    for entity, recs in (state.get("entities") or {}).items():
        for r in recs:
            yield entity, r


def _not_found(ctx: TaskContext, state: dict, sym: str) -> Optional[str]:
    c = ctx.constants.get(sym)
    if c is None:
        return None
    val = c.value
    if c.type[0] == "ID":
        for entity, r in _records(state):
            if entity == c.type[1] and r.get("id") == val:
                return f"ABORT_UNFOUNDED NOT_FOUND {sym} matches {entity} {r['id']}"
        return None
    if isinstance(val, str):
        needle = val.strip().casefold()
        for entity, r in _records(state):
            for k, v in r.items():
                if isinstance(v, str) and v.strip().casefold() == needle:
                    return (f"ABORT_UNFOUNDED NOT_FOUND {sym} matches "
                            f"{entity} {r.get('id', '?')} {k}")
    return None


def _needs_info(ctx: TaskContext, sym: str) -> Optional[str]:
    f = ctx.fields.get(sym)
    if f is None:
        return None
    for c in sorted(ctx.constants.values(), key=lambda c: int(c.sym[1:])):
        if _compatible(f.type, c.type):
            return f"ABORT_UNFOUNDED NEEDS_INFO {sym} has {c.sym}"
    return None


def _ambiguous(refs: List[str]) -> Optional[str]:
    # one T or C names a single candidate, which is not an ambiguity; a lone
    # F is the field that would disambiguate among records, which is fine
    if len(refs) == 1 and refs[0][0] in "TC":
        return f"ABORT_UNFOUNDED AMBIGUOUS {refs[0]} names one candidate"
    return None


def check_abort(ctx: TaskContext, state: dict, reason: str,
                refs: List[str]) -> Optional[str]:
    """None when the abort is founded (or unverifiable); else the rendered
    `ABORT_UNFOUNDED` diagnostic."""
    if not refs:
        return None
    if reason == "NOT_FOUND":
        return _not_found(ctx, state, refs[0])
    if reason == "NEEDS_INFO":
        return _needs_info(ctx, refs[0])
    if reason == "AMBIGUOUS":
        return _ambiguous(refs)
    return None
