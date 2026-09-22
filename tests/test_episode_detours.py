"""The aimed wrong move, and the reaction it buys (plan step 3a).

The dungeon exam has the model repeat a move it was just told failed, 103
turns of 105 (`.claude/plans/general-agent-plan.md` Tier 1 item 1). The
cause is in the corpus: a turn that opens with `Last turn: failed: ...` was
labelled with the oracle's plan for a board the failure did not change, and
the failure itself was a *random* illegal move, so the label had no reason
to answer it. An aimed wrong move — the right object with a verb that does
not apply, or the right verb aimed at something that refuses it — produces
a failure the next label does answer.
"""
import json
import random

import pytest

from data.gen.episodes import (call_parts, pick_detour, predictable_wrong,
                               report_reaction)
from harness import decision
from runtime.worlds import get_world


def _turn(world, seed=0):
    module = decision.world_module(world)
    oracle = decision.oracle_module(world)
    state = decision.sample_state(world, random.Random(seed))
    obs = module.observe(state)
    want = oracle.plan_turn(state, budget=state.get("turn_budget", 3))
    return module, get_world(world), state, obs.constants, want


def test_call_parts_reads_the_first_call():
    assert call_parts("CALL @go $1\nSTOP\n") == ("go", ("$1",))
    assert call_parts("FILTER r0 F1 EQ C0 -> r1\nSTOP\n") == ("", ())


@pytest.mark.parametrize("world", ["house", "warehouse_robot", "app_ticket"])
def test_the_aimed_move_is_illegal_and_aimed(world):
    module, world_dict, state, constants, want = _turn(world)
    found = predictable_wrong(module, world_dict, state, want, constants,
                              random.Random(1))
    assert found, f"no aimed move on {world}"
    aimed, branch = found
    assert branch in ("same_obj", "same_verb", "on_board")
    assert aimed not in set(module.legal_actions(state)), \
        "an aimed move that is legal would not fail, and teaches nothing"
    verb, args = call_parts(aimed)
    want_verb, want_args = call_parts(want)
    on_board = {a for action in module.legal_actions(state)
                for a in call_parts(action)[1]}
    # the right object with the wrong verb, the right verb at the wrong
    # object, or at least a verb error on something the board really has
    assert (set(args) & set(want_args)) or verb == want_verb or \
        (set(args) & on_board), (want, aimed)


def test_a_world_where_nothing_can_fail_gets_no_aimed_move():
    """Blackjack: hit and stand are always legal, so there is no plausible
    wrong move to play and the caller falls back to a random one."""
    module, world_dict, state, constants, want = _turn("cards")
    assert predictable_wrong(module, world_dict, state, want, constants,
                             random.Random(2)) is None


def test_pick_detour_reports_which_kind_it_played():
    module, world_dict, state, constants, want = _turn("house")
    aimed, kind = pick_detour(module, world_dict, state, want, constants,
                              random.Random(3), illegal_share=1.0,
                              predictable_share=1.0)
    assert aimed and kind.startswith("aimed:"), kind
    _, kind = pick_detour(module, world_dict, state, want, constants,
                          random.Random(3), illegal_share=1.0,
                          predictable_share=0.0)
    assert kind == "random_illegal"
    _, kind = pick_detour(module, world_dict, state, want, constants,
                          random.Random(3), illegal_share=0.0,
                          predictable_share=1.0)
    assert kind == "legal"


def _row(turn, oracle_tool, played=None, failed=False, kind="on_path",
         failure=None):
    """One written row, as `report_reaction` reads it: the label lives in
    provenance and the failure the turn opens with lives in the request,
    because that is where the model sees it."""
    request = "You are in the hall.\n"
    request += f"Last turn: failed: {failure}.\n" if failure \
        else "Last turn: nothing yet.\n"
    return {"request": request,
            "provenance": {"world": "house", "seed": 1, "turn": turn,
                           "oracle_tool": oracle_tool, "played_move": played,
                           "played_failed": failed, "detour_kind": kind}}


def test_report_reaction_counts_repeats_and_answers(tmp_path, capsys):
    path = tmp_path / "ep.jsonl"
    rows = [
        # the oracle wanted to open the shut way; the aimed wrong move walked
        # into it instead
        _row(0, "CALL @open $1", played="CALL @go $1", failed=True,
             kind="aimed:same_obj"),
        # answers it: same object, another verb, and the reason names it
        _row(1, "CALL @open $1", failure="the north way is shut - open it"),
        _row(2, "CALL @go $4", played="CALL @take $9", failed=True,
             kind="random_illegal"),
        # repeats the move that just failed
        _row(3, "CALL @take $9", failure="the lamp is not here"),
        # no failure to open with, so not counted at all
        _row(4, "CALL @go $2", played="CALL @go $2"),
        _row(5, "CALL @go $3"),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    stats = report_reaction(path)
    assert stats["after_failure"] == 2
    assert stats["repeated"] == 1
    assert stats["same_object"] == 1
    assert stats["answered"] == 1
    assert stats["by_branch"]["aimed:same_obj"] == {"n": 1, "answered": 1}
    assert stats["by_branch"]["random_illegal"] == {"n": 1, "answered": 0}
    assert "of 2 turns that open with a failure" in capsys.readouterr().out


def test_report_reaction_is_silent_when_nothing_failed(tmp_path):
    path = tmp_path / "ep.jsonl"
    path.write_text(json.dumps(_row(0, "CALL @go $1")), encoding="utf-8")
    assert report_reaction(path) is None
