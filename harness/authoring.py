"""Authoring form for reference programs and generator templates.

Programs are written with stable names and resolved to a task's per-request
symbols (fresh T/F/C assignment) before parsing:

  @list_cards            -> the tool's T symbol
  @card.due              -> the F symbol of (entity card, field due)
  @send_message.text     -> the F symbol of tool send_message's unlinked
                            param `text`
  $2                     -> the symbol of the task's third constant: C2
                            under 0.3.x symbols, S1 / I0 / ... under the
                            0.4.0 typed letters (ConstDecl.index)

Only symbol-form programs ever reach the model or the corpus; the authoring
form exists so humans and templates don't hand-track random assignments.
"""
from __future__ import annotations

import re

from core.ir import TaskContext

_NAME_RE = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?")
_CONST_RE = re.compile(r"\$(\d+)")


class ResolveError(Exception):
    pass


def resolve(text: str, ctx: TaskContext) -> str:
    tools_by_name = {t.name: t for t in ctx.tools.values()}
    entity_fields = {}
    param_fields = {}
    for f in ctx.fields.values():
        if f.entity is not None:
            entity_fields[(f.entity, f.name)] = f.sym

    # unlinked params: find them through the tool decls (their param syms
    # point at FieldDecls with entity=None)
    for t in ctx.tools.values():
        for p in t.params:
            fd = ctx.fields.get(p.sym)
            if fd is not None and fd.entity is None:
                param_fields[(t.name, fd.name)] = p.sym

    def name_sub(m: re.Match) -> str:
        head, tail = m.group(1), m.group(2)
        if tail is None:
            tool = tools_by_name.get(head)
            if tool is None:
                raise ResolveError(f"unknown tool @{head}")
            return tool.sym
        if (head, tail) in entity_fields:
            return entity_fields[(head, tail)]
        if (head, tail) in param_fields:
            return param_fields[(head, tail)]
        raise ResolveError(f"unknown field @{head}.{tail}")

    by_index = {c.index: c.sym for c in ctx.constants.values()
                if c.index is not None}

    def const_sub(m: re.Match) -> str:
        i = int(m.group(1))
        return by_index.get(i, f"C{i}")

    out = _NAME_RE.sub(name_sub, text)
    out = _CONST_RE.sub(const_sub, out)
    return out
