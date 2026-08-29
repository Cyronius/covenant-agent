"""Task-context construction: per-request dynamic symbol assignment.

Tools and fields are dynamic runtime symbols (T0…, F0…) assigned fresh for
every request (spec §1). Given a world, a constants list, and a seeded RNG,
build_context() assigns symbols (shuffled, so nothing about a symbol's number
is learnable) and returns:

  - a core TaskContext (for parse/typecheck/compile),
  - the sandbox payload (tools with impls + field map + constant values).

serialize_context() renders the model-facing input text: request + tool
schemas + constants, all by symbol, with descriptions carrying the meaning.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from core.ir import (ConstDecl, FieldDecl, TaskContext, ToolDecl, ToolParam,
                     format_type, parse_type)


def build_context(world: dict, constants: List[dict],
                  rng: Optional[random.Random] = None,
                  tool_subset: Optional[List[str]] = None
                  ) -> Tuple[TaskContext, dict]:
    """constants: [{type, value, desc}] in a stable order.
    tool_subset: restrict the visible tool set (by internal name)."""
    rng = rng or random.Random(0)

    tools = [t for t in world["tools"]
             if tool_subset is None or t["name"] in tool_subset]
    tools = list(tools)
    rng.shuffle(tools)

    # field slots: every (entity, field) pair, plus unlinked tool params
    slots: List[tuple] = []
    for entity in sorted(world["entities"]):
        for fname in world["entities"][entity]:
            slots.append(("field", entity, fname))
    for t in sorted(tools, key=lambda t: t["name"]):
        for p in t["params"]:
            if not p.get("field"):
                slots.append(("param", t["name"], p["name"]))
    rng.shuffle(slots)

    field_syms = {}   # ('field', entity, fname) / ('param', tool, pname) -> sym
    field_decls = {}
    for i, slot in enumerate(slots):
        sym = f"F{i}"
        field_syms[slot] = sym
        if slot[0] == "field":
            _, entity, fname = slot
            field_decls[sym] = FieldDecl(
                sym=sym, entity=entity, name=fname,
                type=parse_type(world["entities"][entity][fname]),
                desc=f"{entity}.{fname}")
        else:
            _, tname, pname = slot
            tool = next(t for t in tools if t["name"] == tname)
            p = next(p for p in tool["params"] if p["name"] == pname)
            field_decls[sym] = FieldDecl(
                sym=sym, entity=None, name=pname,
                type=parse_type(p["type"]), desc=p.get("desc", pname))

    tool_decls = {}
    sandbox_tools = []
    for i, t in enumerate(tools):
        sym = f"T{i}"
        params = []
        for p in t["params"]:
            if p.get("field"):
                psym = field_syms[("field", p["field"][0], p["field"][1])]
            else:
                psym = field_syms[("param", t["name"], p["name"])]
            params.append(ToolParam(
                sym=psym, type=parse_type(p["type"]),
                required=p.get("required", True), desc=p.get("desc", "")))
        tool_decls[sym] = ToolDecl(
            sym=sym, name=t["name"], desc=t["desc"], params=params,
            returns=parse_type(t["returns"]) if t.get("returns") else None,
            effects=list(t["effects"]))
        sandbox_tools.append({
            "sym": sym, "name": t["name"],
            "params": [{"name": p["name"], "type": p["type"],
                        "required": p.get("required", True)}
                       for p in t["params"]],
            "effects": list(t["effects"]),
            "impl": t["impl"],
        })

    const_decls = {}
    sandbox_constants = {}
    for i, c in enumerate(constants):
        sym = f"C{i}"
        const_decls[sym] = ConstDecl(sym=sym, type=parse_type(c["type"]),
                                     value=c["value"], desc=c.get("desc", ""))
        sandbox_constants[sym] = c["value"]

    ctx = TaskContext(tools=tool_decls, fields=field_decls,
                      constants=const_decls)
    sandbox = {
        "tools": sandbox_tools,
        "fields": [{"sym": f.sym, "entity": f.entity, "name": f.name}
                   for f in field_decls.values()],
        "constants": sandbox_constants,
    }
    return ctx, sandbox


def sandbox_from_context(ctx: TaskContext, world: dict) -> dict:
    """Rebuild the sandbox payload for a TaskContext restored from JSON."""
    by_name = {t["name"]: t for t in world["tools"]}
    tools = []
    for t in ctx.tools.values():
        w = by_name[t.name]
        tools.append({
            "sym": t.sym, "name": t.name,
            "params": [{"name": p["name"], "type": p["type"],
                        "required": p.get("required", True)}
                       for p in w["params"]],
            "effects": list(w["effects"]),
            "impl": w["impl"],
        })
    return {
        "tools": tools,
        "fields": [{"sym": f.sym, "entity": f.entity, "name": f.name}
                   for f in ctx.fields.values()],
        "constants": {c.sym: c.value for c in ctx.constants.values()},
    }


def serialize_context(request: str, ctx: TaskContext) -> str:
    """The model-facing input: request, then schemas, all symbolic."""
    lines = [f"REQUEST: {request}", "TOOLS:"]
    for t in sorted(ctx.tools.values(), key=lambda t: int(t.sym[1:])):
        ps = " ".join(
            f"{p.sym}:{format_type(p.type)}{'' if p.required else '?'}"
            for p in t.params)
        ret = format_type(t.returns) if t.returns else "-"
        eff = ",".join(t.effects)
        lines.append(f"{t.sym} ({ps}) -> {ret} [{eff}] :: {t.desc}")
    lines.append("FIELDS:")
    for f in sorted(ctx.fields.values(), key=lambda f: int(f.sym[1:])):
        ent = f.entity or "-"
        lines.append(f"{f.sym} {ent} {format_type(f.type)} :: {f.desc}")
    lines.append("CONSTANTS:")
    for c in sorted(ctx.constants.values(), key=lambda c: int(c.sym[1:])):
        lines.append(f"{c.sym} {format_type(c.type)} :: {c.desc}")
    if ctx.initial_registers:
        lines.append("REGISTERS:")
        for r in sorted(ctx.initial_registers, key=lambda r: int(r[1:])):
            lines.append(f"{r} {format_type(ctx.initial_registers[r])}")
    return "\n".join(lines) + "\n"
