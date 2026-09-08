"""Front-to-back static pipeline: text -> parse -> typecheck -> effects -> JS.

This is the single entry point the harness, tests, and data generator use;
nothing else should chain the stages by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import diagnostics as dg
from .compile import compile_program
from .effects import check_effects, program_effects
from .ir import Program, TaskContext
from .parser import parse
from .typecheck import check


@dataclass
class BuildResult:
    parse_ok: bool
    compile_ok: bool           # parse + typecheck + effect check all clean
    diagnostics: List[dg.Diagnostic] = field(default_factory=list)
    program: Optional[Program] = None
    js: Optional[str] = None
    static_effects: List[str] = field(default_factory=list)
    pause_envs: List[dict] = field(default_factory=list)
    # register types at the program's end: after a runtime error the
    # reactive harness types the sandbox's bound registers from this
    final_env: dict = field(default_factory=dict)

    def rendered_diagnostics(self) -> List[str]:
        return [d.render() for d in self.diagnostics]


def build(text: str, ctx: TaskContext) -> BuildResult:
    program, pdiags = parse(text)
    if program is None:
        return BuildResult(parse_ok=False, compile_ok=False, diagnostics=pdiags)
    tc = check(program, ctx)
    ediags = check_effects(program, ctx)
    diags = list(tc.diagnostics) + ediags
    effects = sorted(program_effects(program, ctx))
    if diags:
        return BuildResult(parse_ok=True, compile_ok=False, diagnostics=diags,
                           program=program, static_effects=effects,
                           pause_envs=tc.pause_envs, final_env=tc.final_env)
    return BuildResult(parse_ok=True, compile_ok=True, program=program,
                       js=compile_program(program, ctx),
                       static_effects=effects, pause_envs=tc.pause_envs,
                       final_env=tc.final_env)
