"""Theme surface v2 + recipes L12-L18 (plan lane-c-retrain §C1/§C2): every
new level samples, renders, compiles, and executes to the expected status
on a registered theme; the new tools carry the shapes the real sessions
need (two-id signatures, optional trailing params, EXTERNAL content tools)."""
import random
from pathlib import Path

import pytest

from data.gen import english, programs
from data.gen.domains import load_theme, register_theme
from data.gen.worldgen import gen_state
from harness.authoring import resolve
from harness.context import build_context
from harness.run import reference_planner, run_task
from harness.taskbuild import build_task
from runtime.worlds import get_world

THEME = Path(__file__).resolve().parent.parent / "data" / "gen" / "themes" / "gen_airline_ops.json"


@pytest.fixture(scope="module")
def theme_name():
    return register_theme(load_theme(THEME))


def test_v2_tools_have_the_real_shapes(theme_name):
    world = get_world(theme_name)
    tools = {t["name"]: t for t in world["tools"]}
    assert len(tools) == 18
    upd = tools["update_segment_note"]
    assert [p["type"] for p in upd["params"]][:2] == ["ID:segment", "ID:segment_note"]
    assert upd["params"][-1]["required"] is False            # optional trailing title
    assert tools["create_segment"]["params"][-1]["required"] is False
    ext = [t for t in tools.values() if t["effects"] == ["EXTERNAL"]]
    assert {t["returns"] for t in ext} == {"STR"} and len(ext) == 2
    assert "segment_note" in world["entities"] and world["entities"]["segment"]["image"] == "STR"


@pytest.mark.parametrize("level", range(12, 19))
def test_each_v2_level_roundtrips(theme_name, level):
    world = get_world(theme_name)
    seen = set()
    done = 0
    for attempt in range(60):
        seed = level * 100 + attempt
        rng = random.Random(seed)
        state = gen_state(theme_name, rng, world["now"])
        try:
            sample = programs.sample_level(level, theme_name, state, world["now"], rng, set())
        except programs.SampleError:
            continue
        request, _ = english.render(sample.frame, rng)
        assert request and "{" not in request.replace("{0}", "").replace("{1}", "")
        ctx, sctx = build_context(world, sample.constants, random.Random(seed))
        segments = [resolve(s, ctx) for s in sample.segments]
        task = build_task(task_id=f"t{seed}", level=level, world_name=theme_name,
                          request=request, constants=sample.constants, segments=segments,
                          seed=seed, state=state, tags=sample.tags,
                          expected_status="aborted" if "abort" in sample.tags else "ok",
                          prebuilt=(ctx, sctx))
        row = run_task(task, reference_planner(task))
        assert row["goal_success"], (request, row["status"], row["diagnostics"][:2])
        seen.add(sample.frame["recipe"])
        done += 1
        if done >= 6:
            break
    assert done >= 3, f"level {level} rarely samples"
    if level == 16:
        assert len(seen) >= 2       # more than one abort flavour appears


def test_report_returns_a_value_and_search_hits_the_stub(theme_name):
    world = get_world(theme_name)
    for level, key in ((14, "report"), (17, "search")):
        for attempt in range(40):
            rng = random.Random(7 + attempt)
            state = gen_state(theme_name, rng, world["now"])
            try:
                sample = programs.sample_level(level, theme_name, state, world["now"], rng, set())
            except programs.SampleError:
                continue
            if "RETURN" not in sample.segments[0]:
                continue
            ctx, sctx = build_context(world, sample.constants, random.Random(1))
            segs = [resolve(s, ctx) for s in sample.segments]
            task = build_task(task_id="r", level=level, world_name=theme_name, request="x",
                              constants=sample.constants, segments=segs, state=state,
                              prebuilt=(ctx, sctx))
            assert task["reference"]["return_value"] is not None
            break
        else:
            pytest.fail(f"no RETURN sample for level {level}")
