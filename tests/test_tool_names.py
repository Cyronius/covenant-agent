"""Spec 0.8.0: a tool's declared name is input text, never a symbol (§1/§6).

The serializer may render it between the symbol and the parameters; nothing
that reads a program or a context by symbol may be moved by it -- not the
parser, not the model-side line readers, and not the program itself.
"""
import json
import random
import sys
from pathlib import Path

from core.ir import TaskContext
from core.pipeline import build
from data.gen import programs
from data.gen.worldgen import gen_state
from harness.authoring import resolve
from harness.context import build_context, render_name, serialize_context
from runtime.worlds import get_world

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "tiny"))


def _task(level, seed, world="kanban"):
    rng = random.Random(seed)
    w = get_world(world)
    state = gen_state(world, rng, w["now"])
    s = programs.sample_level(level, world, state, w["now"], rng, set())
    ctx, _ = build_context(w, s.constants, random.Random(seed ^ 0x5EED),
                           symbols="typed", enums=True)
    return ctx, [resolve(seg, ctx) for seg in s.segments], s


def test_names_render_between_symbol_and_params():
    ctx, _, _ = _task(3, 7)
    on = serialize_context("do it", ctx, names=True).splitlines()
    off = serialize_context("do it", ctx).splitlines()
    tools = sorted(ctx.tools.values(), key=lambda t: int(t.sym[1:]))
    tool_lines_on = [l for l in on if l.split(" ", 1)[0] in ctx.tools]
    tool_lines_off = [l for l in off if l.split(" ", 1)[0] in ctx.tools]
    for t, a, b in zip(tools, tool_lines_on, tool_lines_off):
        assert a == b.replace(f"{t.sym} (", f"{t.sym} {render_name(t.name)} (", 1)
    # everything but the tool lines is untouched
    assert [l for l in on if l not in tool_lines_on] == \
        [l for l in off if l not in tool_lines_off]


def test_default_is_the_pre_080_line():
    ctx, _, _ = _task(3, 7)
    for line in serialize_context("x", ctx).splitlines():
        if line.split(" ", 1)[0] in ctx.tools:
            assert line.split(" ", 2)[1].startswith("(")


def test_whitespace_in_a_name_renders_as_underscore():
    assert render_name("archive  the card") == "archive_the_card"


def test_a_symbol_shaped_name_never_binds():
    """Rename every tool to another tool's symbol: the program still binds
    to the symbols, and the line readers still key lines by symbol."""
    from diagnose import parse as parse_lines
    from prep import Lines

    ctx, segs, _ = _task(5, 11)
    syms = sorted(ctx.tools, key=lambda s: int(s[1:]))
    raw = json.loads(json.dumps(ctx.to_json()))
    for t, other in zip(raw["tools"], reversed(syms)):
        t["name"] = other
    renamed = TaskContext.from_json(raw)
    plain = build(segs[0], ctx)
    named = build(segs[0], renamed)
    assert plain.compile_ok and named.compile_ok
    assert named.js == plain.js
    src = serialize_context("req", renamed, names=True)
    ln = Lines(src)
    assert [s for s, _ in ln.tools] == syms
    tools, _ = parse_lines(src)
    assert sorted(tools, key=lambda s: int(s[1:])) == syms


def test_programs_round_trip_unchanged_with_names_on():
    """The target a model is trained on never depends on the serializer."""
    from corpus import program_tokens, detokenize
    for level in (1, 3, 8, 11):
        ctx, segs, _ = _task(level, 100 + level)
        for seg in segs:
            assert detokenize(program_tokens(seg)).strip() == seg.strip()
            assert build(seg, ctx).compile_ok
        # the input changes, the target does not
        assert serialize_context("r", ctx, names=True) != serialize_context("r", ctx)
