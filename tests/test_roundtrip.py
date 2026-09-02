"""F2 round-trip requirements:
  - every spec/examples (curriculum) program parses, typechecks, compiles;
  - property test: random valid programs from the F4 generator always parse,
    typecheck, and compile;
  - the compiler is deterministic (same AST -> byte-identical JS).
"""
import random

import pytest

from core.ir import TaskContext, parse_type
from core.compile import compile_program
from core.parser import parse
from core.pipeline import build
from data.gen import programs
from data.gen.worldgen import gen_state
from harness.authoring import resolve
from harness.context import build_context
from harness.curriculum import CURRICULUM
from runtime.worlds import get_world


@pytest.mark.parametrize("spec", CURRICULUM, ids=lambda s: s["id"])
def test_curriculum_examples_roundtrip(spec):
    world = get_world(spec["world"])
    ctx, _ = build_context(world, spec["constants"], random.Random(3))
    cur = ctx
    for seg in spec["segments"]:
        res = build(resolve(seg, ctx), cur)
        assert res.compile_ok, res.rendered_diagnostics()
        if res.pause_envs:
            cur = TaskContext(
                tools=ctx.tools, fields=ctx.fields, constants=ctx.constants,
                initial_registers={r: parse_type(t)
                                   for r, t in res.pause_envs[0].items()})


@pytest.mark.parametrize("level", range(12))
def test_property_random_programs_compile(level):
    """20 random samples per level: always parse + typecheck + compile."""
    count = 0
    for seed in range(1000):
        if count >= 20:
            break
        rng = random.Random(seed * 13 + level)
        world_name = rng.choice(["kanban", "crm", "projects"])
        world = get_world(world_name)
        state = gen_state(world_name, rng, world["now"])
        try:
            sample = programs.sample_level(
                level, world_name, state, world["now"], rng, set())
        except programs.SampleError:
            continue
        ctx, _ = build_context(world, sample.constants,
                               random.Random(seed), None)
        cur = ctx
        for seg in sample.segments:
            res = build(resolve(seg, ctx), cur)
            assert res.compile_ok, (world_name, seg,
                                    res.rendered_diagnostics())
            if res.pause_envs:
                cur = TaskContext(
                    tools=ctx.tools, fields=ctx.fields,
                    constants=ctx.constants,
                    initial_registers={r: parse_type(t)
                                       for r, t in res.pause_envs[0].items()})
        count += 1
    assert count == 20


def test_compile_deterministic():
    world = get_world("kanban")
    constants = [{"type": "STR", "value": "done", "desc": "done"},
                 {"type": "BOOL", "value": False, "desc": "false"}]
    ctx, _ = build_context(world, constants, random.Random(1))
    src = resolve(
        "CALL @list_cards -> r0\n"
        "FILTER r0 @card.status EQ $0 AND @card.archived EQ $1 -> r1\n"
        "FOREACH r1 -> r2\n"
        "  CALL @archive_card r2 -> r3\n"
        "STOP\n", ctx)
    p1, d1 = parse(src)
    p2, d2 = parse(src)
    assert d1 == [] and d2 == []
    assert compile_program(p1, ctx) == compile_program(p2, ctx)
