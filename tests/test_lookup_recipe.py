"""L21, answering rather than acting (plan step 2b).

79% of training rows end in `STOP` and 7.6% in `RETURN`, and no recipe
listed things and handed them back, so the planner answered questions by
acting on something (`.claude/plans/general-agent-plan.md` Tier 1 item 2).
Every row this family writes ends in `RETURN`, a third of them never
filter, and the references execute — which is the part that makes the rows
worth training on rather than merely present.
"""
import random

import pytest

from data.gen import programs
from data.gen.worldgen import gen_state
from harness.authoring import resolve
from harness.context import build_context
from harness.filter_check import padded_clauses
from harness.run import reference_planner, run_task
from harness.taskbuild import build_task
from core.pipeline import build
from runtime.worlds import get_world

WORLDS = ("crm", "kanban", "projects")


def _samples(n=40, world_names=WORLDS):
    """(world, sample) pairs from the L21 recipe, skipping draws a world
    cannot carry (too few records to ask a question about)."""
    out = []
    for seed in range(n * 4):
        if len(out) >= n:
            break
        rng = random.Random(seed)
        world_name = rng.choice(list(world_names))
        world = get_world(world_name)
        state = gen_state(world_name, rng, world["now"])
        try:
            s = programs.sample_level(21, world_name, state, world["now"],
                                      rng, set())
        except programs.SampleError:
            continue
        out.append((world_name, state, s))
    assert len(out) >= n // 2, f"only {len(out)} L21 samples in {n * 4} draws"
    return out


def test_every_row_ends_in_return():
    """The whole point of the family: the corpus had 7.6% of rows ending in
    RETURN and no recipe that answers a question by returning a value."""
    for _, _, s in _samples():
        last = [l for l in s.segments[-1].strip().splitlines() if l.strip()][-1]
        assert last.strip().startswith("RETURN"), s.segments[-1]


def test_a_third_of_them_never_filter():
    """"List Bob's cards" has an empty FILTER slot and the corpus has never
    shown one — 2.8% of rows that call a list tool skip it. That empty slot
    is what the padding reflex is made of (results/REFLEX.md)."""
    rows = _samples(60)
    bare = sum(1 for _, _, s in rows if "FILTER" not in s.segments[0])
    assert bare / len(rows) > 0.2, f"only {bare}/{len(rows)} skip the filter"


def test_the_references_execute_and_return_what_they_claim():
    for world_name, state, s in _samples(24):
        world = get_world(world_name)
        ctx, sandbox_ctx = build_context(world, s.constants, random.Random(3))
        segments = [resolve(seg, ctx) for seg in s.segments]
        task = build_task(
            task_id="lookup", level=21, world_name=world_name,
            request="irrelevant", constants=s.constants, segments=segments,
            seed=3, state=state, prebuilt=(ctx, sandbox_ctx))
        row = run_task(task, reference_planner(task))
        assert row["goal_success"], row.get("diagnostics")
        assert row["return_match"] is True


def test_the_questions_name_what_the_program_filters_on():
    """2b's own gate: the padding share on the new rows is near zero. The
    first version failed this at 23.1%, because the `first` arm sampled a
    clause its question never mentioned."""
    from data.gen import english
    flagged = withf = 0
    for world_name, state, s in _samples(60):
        world = get_world(world_name)
        ctx, _ = build_context(world, s.constants, random.Random(3))
        res = build(resolve(s.segments[0], ctx), ctx)
        assert res.compile_ok, res.rendered_diagnostics()
        if "FILTER" not in s.segments[0]:
            continue
        withf += 1
        request, _style = english.render(s.frame, random.Random(1))
        if padded_clauses(res.program, ctx, request):
            flagged += 1
    # the reference floor across the tree is 0-17% (results/REFLEX.md §3):
    # synonyms the wording never spells ("completed" for "done") land here
    # too, so this is bounded rather than zeroed.
    assert withf, "no filtered rows drawn"
    assert flagged / withf <= 0.2, f"{flagged}/{withf} rows filter unrequested"


@pytest.mark.parametrize("arm", ["count", "list"])
def test_both_universal_arms_are_reachable(arm):
    """`first` needs a sort spec and the compute arms need an INT field, but
    count and list must work in every world."""
    seen = False
    for _, _, s in _samples(60):
        if f"answering:{arm}" in s.tags:
            seen = True
            break
    assert seen, f"the {arm} arm never fired in 60 draws"
