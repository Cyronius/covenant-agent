"""Task-context construction: per-request dynamic symbol assignment.

Tools and fields are dynamic runtime symbols (T0…, F0…) assigned fresh for
every request (spec §1). Given a world, a constants list, and a seeded RNG,
build_context() assigns symbols (shuffled, so nothing about a symbol's number
is learnable) and returns:

  - a core TaskContext (for parse/typecheck/compile),
  - the sandbox payload (tools with impls + field map + constant values).

serialize_context() renders the model-facing input text: request + tool
schemas + constants, all by symbol, with descriptions carrying the meaning.

Spec 0.4.0 (`.claude/plans/spec-0.4.0.md`), behind `symbols=`/`enums=`:

  symbols="classic"   C0 C1 …             the 0.3.x form; S3 and every
                                          stored suite
  symbols="typed"     S0 N0 B0 D0 I0 …    the letter carries the base type
                                          (§2.1); tool lines show slot
                                          letters, `T5 (I:user=F11 S=F0)`
  enums=True          every enum field of an entity the visible tools
                      touch contributes its values as STR constants of kind
                      `enum:<entity>.<field>` (§2.3), so "set status to
                      cancelled" has a symbol whether or not the request
                      spelled the value
  kinds               a constant dict may carry "kind": name | text |
                      enum:<entity>.<field> (§2.2); rendered after the type
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from core.ir import (ConstDecl, FieldDecl, TaskContext, ToolDecl, ToolParam,
                     format_type, letter_for_type, parse_type)


def _touched_entities(world: dict, tools: List[dict]) -> set:
    """Entities a tool set reads or writes: linked params and return types."""
    ents = set()
    for t in tools:
        for p in t["params"]:
            if p.get("field"):
                ents.add(p["field"][0])
            pt = parse_type(p["type"])
            while pt[0] == "LIST":
                pt = pt[1]
            if pt[0] in ("ID", "OBJ"):
                ents.add(pt[1])
        if t.get("returns"):
            rt = parse_type(t["returns"])
            while rt[0] == "LIST":
                rt = rt[1]
            if rt[0] in ("ID", "OBJ"):
                ents.add(rt[1])
    return {e for e in ents if e in world.get("entities", {})}


def enum_constants(world: dict, tools: List[dict],
                   existing: List[dict]) -> List[dict]:
    """The schema's enum values as constants (spec 0.4.0 §2.3), for enum
    fields of entities the tools touch. A request constant that already
    carries one of the values is left in place and given the kind instead
    of being duplicated (the caller applies that: see `_kinded`)."""
    touched = _touched_entities(world, tools)
    out = []
    have = {(str(c["value"]).strip().lower()) for c in existing
            if c.get("type") == "STR"}
    for key, values in sorted(world.get("enums", {}).items(),
                              key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        entity, fname = key
        if entity not in touched:
            continue
        for v in values:
            if str(v).strip().lower() in have:
                continue
            # the value is in the description: an enum symbol is only
            # useful if the model can tell which value it is
            out.append({"type": "STR", "value": v,
                        "desc": f'{entity}.{fname} "{v}"',
                        "kind": f"enum:{entity}.{fname}"})
    return out


def _kinded(world: dict, tools: List[dict], constants: List[dict]) -> List[dict]:
    """Give request constants that spell an enum value the enum kind, when
    they carry no kind of their own."""
    touched = _touched_entities(world, tools)
    by_value: Dict[str, str] = {}
    for (entity, fname), values in world.get("enums", {}).items():
        if entity in touched:
            for v in values:
                by_value.setdefault(str(v).strip().lower(), f"enum:{entity}.{fname}")
    out = []
    for c in constants:
        c = dict(c)
        if c.get("type") == "STR" and not c.get("kind"):
            k = by_value.get(str(c["value"]).strip().lower())
            if k:
                c["kind"] = k
        out.append(c)
    return out


def assign_constants(constants: List[dict], symbols: str = "classic"
                     ) -> Tuple[dict, dict]:
    """[{type, value, desc[, kind][, index]}] -> ({sym: ConstDecl}, {sym: value}).
    Classic: C<i> positional. Typed: one counter per letter, in list order —
    positional within the letter, so the number still carries nothing."""
    decls, values = {}, {}
    counters: Dict[str, int] = {}
    for i, c in enumerate(constants):
        t = parse_type(c["type"])
        if symbols == "typed":
            letter = letter_for_type(t)
            n = counters.get(letter, 0)
            counters[letter] = n + 1
            sym = f"{letter}{n}"
        else:
            sym = f"C{i}"
        index = c.get("index", i)
        decls[sym] = ConstDecl(sym=sym, type=t, value=c["value"],
                               desc=c.get("desc", ""), kind=c.get("kind", ""),
                               index=index)
        values[sym] = c["value"]
    return decls, values


def build_context(world: dict, constants: List[dict],
                  rng: Optional[random.Random] = None,
                  tool_subset: Optional[List[str]] = None,
                  symbols: str = "classic", enums: bool = False
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

    consts = _kinded(world, tools, list(constants)) if enums else list(constants)
    if enums:
        extra = enum_constants(world, tools, consts)
        consts += [dict(e, index=None) for e in extra]
    const_decls, sandbox_constants = assign_constants(consts, symbols)

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


def is_typed(ctx: TaskContext) -> bool:
    """True when the context uses the 0.4.0 constant letters."""
    return any(c.sym[0] != "C" for c in ctx.constants.values())


def _slot(p: ToolParam) -> str:
    """The slot form on a typed tool line: letter (with entity for I) for
    constant-shaped types, the full type otherwise, then `=F<n>`."""
    t = p.type
    if t[0] in ("STR", "INT", "FLOAT", "BOOL", "TIME"):
        head = letter_for_type(t)
    elif t[0] == "ID":
        head = f"I:{t[1]}"
    else:
        head = format_type(t)
    return f"{head}={p.sym}{'' if p.required else '?'}"


def _sym_key(sym: str) -> tuple:
    return (sym[0], int(sym[1:]))


def render_kind(kind: str) -> str:
    if kind.startswith("enum:"):
        return "enum " + kind[5:]
    return kind


def serialize_context(request: str, ctx: TaskContext) -> str:
    """The model-facing input: request, then schemas, all symbolic."""
    typed = is_typed(ctx)
    lines = [f"REQUEST: {request}", "TOOLS:"]
    for t in sorted(ctx.tools.values(), key=lambda t: int(t.sym[1:])):
        if typed:
            ps = " ".join(_slot(p) for p in t.params)
        else:
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
    for c in sorted(ctx.constants.values(), key=lambda c: _sym_key(c.sym)):
        kind = f" {render_kind(c.kind)}" if c.kind else ""
        lines.append(f"{c.sym} {format_type(c.type)}{kind} :: {c.desc}")
    if ctx.initial_registers:
        lines.append("REGISTERS:")
        for r in sorted(ctx.initial_registers, key=lambda r: int(r[1:])):
            lines.append(f"{r} {format_type(ctx.initial_registers[r])}")
    return "\n".join(lines) + "\n"
