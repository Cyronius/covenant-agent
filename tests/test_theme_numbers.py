"""One numeric field per theme, so the compute tools have something to add.

`sum`/`avg`/`max`/`min` arrived as tools in step 1g, and L21's aggregate arm
needs a `LIST INT` to call them on. Every themed field was STR/BOOL/TIME/ID,
so that arm fired on 1.7% of themed rows (`results/REFLEX.md` §6) — the
worldgen gap this closes. A theme now declares `child.number`
(`data/gen/THEME_SCHEMA.md`), the compiler types it `INT`, the state
generator fills it, and the profile carries the words a request uses for it.
"""
import random

import pytest

import runtime.worlds
from core.pipeline import build
from data.gen import english, programs
from data.gen.domains import (build_profile, build_world, load_theme,
                              make_state_gen, register_domains,
                              validate_theme)
from data.gen.worldgen import gen_state
from harness.authoring import resolve
from harness.context import build_context
from harness.run import reference_planner, run_task
from harness.taskbuild import build_task
from pathlib import Path
from runtime.worlds import get_world

THEMES = Path("data/gen/themes")


@pytest.fixture(scope="module", autouse=True)
def themes():
    """Register the generated domains for this module only — the theme set
    shadows the real-sessions `coursebuilder` world (see
    tests/test_gen_stdlib.py)."""
    worlds = dict(runtime.worlds.WORLDS)
    profiles = dict(programs.PROFILES)
    register_domains(str(THEMES))
    yield
    runtime.worlds.WORLDS.clear()
    runtime.worlds.WORLDS.update(worlds)
    programs.PROFILES.clear()
    programs.PROFILES.update(profiles)


def _theme_paths():
    """{domain -> file}. Most themes are named `gen_<domain>.json`, the two
    oldest are not."""
    return {load_theme(p)["domain"]: p for p in sorted(THEMES.glob("*.json"))}


def _theme(domain):
    return load_theme(_theme_paths()[domain])


def test_a_declared_number_becomes_the_child_entity_s_only_int_field():
    for path in sorted(THEMES.glob("*.json")):
        theme = load_theme(path)
        num = theme["child"].get("number")
        fields = build_world(theme)["entities"][theme["child"]["entity"]]
        ints = [f for f, t in fields.items() if t == "INT"]
        if num is None:
            assert ints == [], f"{theme['domain']} has an unexpected INT field"
        else:
            assert ints == [num["field"]], theme["domain"]


def test_every_generated_record_carries_a_value_inside_the_declared_range():
    theme = _theme("vetclinic")
    num = theme["child"]["number"]
    gen = make_state_gen(theme)
    for seed in range(8):
        state = gen(random.Random(seed), 1_760_000_000)
        records = state["entities"][theme["child"]["entity"]]
        assert records
        for rec in records:
            assert num["min"] <= rec[num["field"]] <= num["max"], rec


def test_the_profile_carries_the_words_a_request_uses_not_the_field_name():
    theme = _theme("dental")
    child = theme["child"]["entity"]
    measures = build_profile(theme)["entities"][child]["measures"]
    assert measures == {"chair_minutes": "chair time in minutes"}


def test_a_theme_without_a_number_still_compiles():
    theme = _theme("vetclinic")
    theme["child"].pop("number")
    validate_theme(theme)
    child = theme["child"]["entity"]
    assert build_profile(theme)["entities"][child]["measures"] == {}
    assert "INT" not in build_world(theme)["entities"][child].values()


@pytest.mark.parametrize("bad,reason", [
    ({"field": "status", "noun": "status", "min": 1, "max": 5}, "collides"),
    ({"field": "n", "noun": "n", "min": 9, "max": 9}, "min < max"),
    ({"field": "n", "noun": "n", "min": 1, "max": 5, "step": 2}, "only"),
    ({"field": "n", "noun": "", "min": 1, "max": 5}, "noun"),
])
def test_validate_theme_rejects_a_malformed_number(bad, reason):
    theme = _theme("vetclinic")
    theme["child"]["number"] = bad
    with pytest.raises(AssertionError) as e:
        validate_theme(theme)
    assert reason in str(e.value)


def _themed_lookup_samples(n=90):
    """(world, state, sample) triples from L21 on generated worlds."""
    names = sorted(set(_theme_paths()) & set(programs.PROFILES))
    out = []
    for seed in range(n * 3):
        if len(out) >= n:
            break
        rng = random.Random(seed)
        world_name = rng.choice(names)
        world = get_world(world_name)
        state = gen_state(world_name, rng, world["now"])
        try:
            s = programs.sample_level(21, world_name, state, world["now"],
                                      rng, set())
        except programs.SampleError:
            continue
        out.append((world_name, state, s))
    return out


def test_the_aggregate_arm_fires_on_themed_worlds_and_stays_a_minority():
    rows = _themed_lookup_samples()
    compute = [r for r in rows if "compute" in r[2].tags]
    share = len(compute) / len(rows)
    # 3 of 11 by weight (programs.sample_lookup); it was 1.7% before a theme
    # could declare a number, and a uniform pick over the six arms would
    # make it half the family
    assert 0.1 < share < 0.5, f"{len(compute)}/{len(rows)} rows aggregate"


def test_the_aggregate_rows_map_a_field_call_a_compute_tool_and_return():
    rows = [r for r in _themed_lookup_samples() if "compute" in r[2].tags]
    assert rows, "no aggregate rows drawn"
    for world_name, state, s in rows[:12]:
        lines = [l.strip() for l in s.segments[0].strip().splitlines()]
        assert lines[-1].startswith("RETURN"), lines
        assert any(l.startswith("MAP ") for l in lines), lines
        world = get_world(world_name)
        ctx, sandbox_ctx = build_context(world, s.constants, random.Random(3))
        segments = [resolve(seg, ctx) for seg in s.segments]
        res = build(segments[0], ctx)
        assert res.compile_ok, res.rendered_diagnostics()
        task = build_task(
            task_id="aggregate", level=21, world_name=world_name,
            request="irrelevant", constants=s.constants, segments=segments,
            seed=3, state=state, prebuilt=(ctx, sandbox_ctx))
        row = run_task(task, reference_planner(task))
        assert row["goal_success"], row.get("diagnostics")
        assert row["return_match"] is True


def test_the_question_says_the_measure_in_the_theme_s_own_words():
    rows = [r for r in _themed_lookup_samples() if "compute" in r[2].tags]
    assert rows, "no aggregate rows drawn"
    for _world, _state, s in rows[:20]:
        request, _style = english.render(s.frame, random.Random(1))
        assert s.frame["measure"] in request, request
        assert "_" not in request, f"raw field name leaked: {request}"
