"""Static effect analysis (spec §7).

program_effects(program, ctx) -> set of effect names: the union of declared
effects of every tool appearing in a CALL, reachable or not.

check_effects(program, ctx) -> [Diagnostic]: EFFECT_UNDECLARED for each
computed effect missing from an EFFECTS header (no header = no check).
"""
from __future__ import annotations

from typing import List, Set

from . import diagnostics as dg
from .ir import (DESTRUCTIVE_EFFECTS, Call, Foreach, If, Parallel, Program,
                 TaskContext, Try)


def _walk_calls(body: list):
    for instr in body:
        if isinstance(instr, Call):
            yield instr
        elif isinstance(instr, Foreach):
            yield from _walk_calls(instr.body)
        elif isinstance(instr, If):
            yield from _walk_calls(instr.then)
            if instr.els is not None:
                yield from _walk_calls(instr.els)
        elif isinstance(instr, Parallel):
            yield from instr.calls
        elif isinstance(instr, Try):
            yield from _walk_calls(instr.body)


def program_effects(program: Program, ctx: TaskContext) -> Set[str]:
    effects: Set[str] = set()
    for call in _walk_calls(program.body):
        tool = ctx.tools.get(call.tool)
        if tool is not None:
            effects.update(tool.effects)
    return effects


def destructive_effects(program: Program, ctx: TaskContext) -> Set[str]:
    return program_effects(program, ctx) & set(DESTRUCTIVE_EFFECTS)


def check_effects(program: Program, ctx: TaskContext) -> List[dg.Diagnostic]:
    if program.effects_decl is None:
        return []
    computed = program_effects(program, ctx)
    declared = set(program.effects_decl)
    return [dg.effect_undeclared(e) for e in sorted(computed - declared)]
