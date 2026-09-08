"""Re-render a stored task under spec 0.4.0 symbols without regenerating it.

Stored suites carry the 0.3.x form: `C<n>` constants, no kinds, no schema
enums, `input_text` = serialize_context(request, ctx) (verified for every
suite, 2026-09-08). `retype_task` rebuilds the constant table from the
stored context — typed letters, inferred kinds, enum constants for the
touched entities — and rewrites everything that names a constant: the
context, the input text, the reference segments and abort referents, and
the sandbox constant table when the task carries one. Tools, fields,
state and expected state are untouched, so scoring is unchanged and the
same 25 tasks can be run under either surface (`run_a.py --symbols typed
--enums --kinds`), which is the step-0 A/B of `.claude/plans/spec-0.4.0.md`.

Kind inference is for stored tasks only; the generator and the real-suite
builder set kinds explicitly when they emit 0.4.0 suites.
"""
from __future__ import annotations

import re
from typing import Dict, List

from core.ir import TaskContext, format_type
from harness.context import (assign_constants, enum_constants, _kinded,
                             serialize_context)

_NAME_HINT = re.compile(r"\b(name|named|called|titled|title|assignee|owner|"
                        r"customer|user|person|who)\b", re.I)


def infer_kind(c: dict) -> str:
    """name | text for a STR constant without a kind (enum kinds come from
    the schema match in harness.context._kinded)."""
    if c.get("kind"):
        return c["kind"]
    if c.get("type") != "STR":
        return ""
    desc = str(c.get("desc", ""))
    if "{0}" in str(c.get("value", "")):
        return "text"                      # a FORMAT template
    return "name" if _NAME_HINT.search(desc) else "text"


def _world_tools(world: dict, ctx: TaskContext) -> List[dict]:
    by_name = {t["name"]: t for t in world["tools"]}
    return [by_name[t.name] for t in ctx.tools.values() if t.name in by_name]


def retype_task(task: dict, world: dict, symbols: str = "typed",
                enums: bool = True, kinds: bool = True) -> dict:
    """A copy of `task` re-rendered under the requested surface."""
    ctx = TaskContext.from_json(task["context"])
    old = sorted(ctx.constants.values(),
                 key=lambda c: (c.index if c.index is not None else 10**6))
    constants = []
    for c in old:
        d = {"type": format_type(c.type), "value": c.value, "desc": c.desc,
             "index": c.index}
        if c.kind:
            d["kind"] = c.kind
        constants.append(d)
    tools = _world_tools(world, ctx)
    if enums:
        constants = _kinded(world, tools, constants)
        constants += [dict(e, index=None)
                      for e in enum_constants(world, tools, constants)]
    if kinds:
        for d in constants:
            d["kind"] = infer_kind(d)
    else:
        for d in constants:
            if not d.get("kind", "").startswith("enum:"):
                d["kind"] = ""
    decls, values = assign_constants(constants, symbols)

    # old sym -> new sym, by constants-list position
    mapping: Dict[str, str] = {}
    for c in old:
        new = next(n for n in decls.values() if n.index == c.index)
        mapping[c.sym] = new.sym
    sym_re = re.compile(r"\bC(\d+)\b")

    def remap(text: str) -> str:
        return sym_re.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)

    new_ctx = TaskContext(tools=ctx.tools, fields=ctx.fields, constants=decls,
                          initial_registers=ctx.initial_registers)
    out = dict(task)
    out["context"] = new_ctx.to_json()
    out["input_text"] = serialize_context(task["request"], new_ctx)
    ref = dict(task.get("reference", {}))
    if "segments" in ref:
        ref["segments"] = [remap(s) for s in ref["segments"]]
    if ref.get("abort_refs"):
        ref["abort_refs"] = [mapping.get(r, r) for r in ref["abort_refs"]]
    out["reference"] = ref
    if task.get("sandbox"):
        sb = dict(task["sandbox"])
        sb["constants"] = values
        out["sandbox"] = sb
    out["spec_version"] = "0.4.0"
    out["symbols"] = symbols
    return out
