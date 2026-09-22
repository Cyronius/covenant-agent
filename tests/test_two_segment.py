"""Observe, then decide (plan step 3b).

One recipe produced `PAUSE` before this: a filter-then-act program cut in
half, which teaches "pause when the request says report back". It does not
teach "pause because you cannot know yet" — and that is the habit the loop
needs, because `results/R4.md` measured what happens without it: told in the
prompt to pause and look, the 27B wrote guesses about the data into a
one-shot program instead.

The three levels with a real data-dependent decision now have a two-segment
form: 5 (which branch), 7's notify-if-any, 11 (found or not). What has to
hold is that the second segment is the ending the *state* calls for, and
that the pause is not a tell — a paused first segment must sometimes end in
an action and sometimes in a decline.
"""
import random

import pytest

from data.gen import programs
from data.gen.worldgen import gen_state
from harness.authoring import resolve
from harness.context import build_context
from harness.run import reference_planner, run_task
from harness.taskbuild import build_task
from runtime.worlds import get_world

WORLDS = ("crm", "kanban", "projects")


def _draws(level, n=60, worlds=WORLDS):
    """(world, state, sample) triples for one level."""
    out = []
    for seed in range(n * 6):
        if len(out) >= n:
            break
        rng = random.Random(seed)
        world_name = rng.choice(list(worlds))
        world = get_world(world_name)
        state = gen_state(world_name, rng, world["now"])
        try:
            s = programs.sample_level(level, world_name, state, world["now"],
                                      rng, set())
        except programs.SampleError:
            continue
        out.append((world_name, state, s))
    return out


def _two(level, **kw):
    rows = [r for r in _draws(level, **kw) if len(r[2].segments) > 1]
    assert rows, f"level {level} never drew a two-segment sample"
    return rows


@pytest.mark.parametrize("level", [5, 7, 11])
def test_the_first_segment_ends_by_pausing(level):
    for _w, _s, sample in _two(level):
        lines = [l.strip() for l in sample.segments[0].strip().splitlines()]
        assert lines[-1] == "PAUSE", sample.segments[0]
        assert "two_segment" in sample.tags


def test_the_branch_that_survives_is_the_one_the_record_calls_for():
    """L5's second segment is a single action, not an IF: the program cannot
    be written before the first segment has run."""
    for world_name, state, sample in _two(5):
        seg2 = sample.segments[1]
        assert "IF" not in seg2 and "ELSE" not in seg2, seg2
        assert sample.frame["observed"] in (True, False)
        # the action in segment 2 is the frame's then-branch exactly when the
        # observed value matched the condition
        wanted = sample.frame["then" if sample.frame["observed"] else "else"]
        assert wanted["tool"] in seg2, (wanted["tool"], seg2)


def test_a_pause_is_not_a_tell_that_the_answer_is_not_found():
    """Half of L11's paused rows end in an action, so `PAUSE` cannot be read
    as "this one aborts" from the shape of the first segment alone."""
    rows = _two(11, n=120)
    declined = [r for r in rows if "ABORT" in r[2].segments[1]]
    acted = [r for r in rows if "ABORT" not in r[2].segments[1]]
    assert declined and acted, \
        f"{len(declined)} declines, {len(acted)} actions - need both"
    for _w, _s, sample in declined:
        assert "abort" in sample.tags
    for _w, _s, sample in acted:
        assert "abort" not in sample.tags, \
            "an acting row tagged abort would be scored as expected_status " \
            "aborted"


def test_the_notify_row_that_found_nothing_does_nothing():
    """L7's count variant: when the filter matched nothing the honest second
    segment is a bare STOP, a shape no one-shot row in the corpus has."""
    rows = _two(7, n=120)
    empties = [r for r in rows if r[2].frame.get("observed") is False]
    if not empties:
        pytest.skip("no empty-match draw in this sample")
    for _w, _s, sample in empties:
        assert sample.segments[1].strip() == "STOP", sample.segments[1]


@pytest.mark.parametrize("level", [5, 7, 11])
def test_the_two_segment_references_still_execute(level):
    for world_name, state, sample in _two(level)[:8]:
        world = get_world(world_name)
        ctx, sandbox_ctx = build_context(world, sample.constants,
                                        random.Random(3))
        segments = [resolve(seg, ctx) for seg in sample.segments]
        task = build_task(
            task_id=f"two_{level}", level=level, world_name=world_name,
            request="irrelevant", constants=sample.constants,
            segments=segments, seed=3, state=state,
            expected_status="aborted" if "abort" in sample.tags else "ok",
            prebuilt=(ctx, sandbox_ctx))
        row = run_task(task, reference_planner(task))
        assert row["goal_success"], (sample.segments, row.get("diagnostics"))
        assert row["segments"] == 2, row["segments"]


def test_report_segments_counts_the_paused_rows_per_level(tmp_path, capsys):
    import json

    from data.gen.__main__ import report_segments
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"level": 5, "reference": {"segments": ["CALL @a -> r0\nPAUSE\n",
                                               "STOP\n"]}},
        {"level": 5, "reference": {"segments": ["STOP\n"]}},
        {"level": 2, "reference": {"segments": ["STOP\n"]}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report_segments(path)
    out = capsys.readouterr().out
    assert "1/3 rows (33.3%)" in out
    assert "L5 1/2" in out
