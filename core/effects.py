"""Static effect analysis (spec §7).

program_effects(program, ctx) -> set of consequence properties (spec 0.7.0:
a subset of "mutates"/"irreversible"/"external"): the union of declared
properties of every tool appearing in a CALL, reachable or not.
"""
from __future__ import annotations

from typing import Set

from .ir import Call, Foreach, If, Parallel, Program, TaskContext, Try


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
