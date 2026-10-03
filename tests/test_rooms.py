"""The grid rooms world (BabyAI missions): engine rules, oracle, perception.

Every case goes through the real pipeline (resolve -> build -> node
sandbox), as tests/test_decision_worlds.py does. The world is registered at
runtime here rather than in runtime/worlds/__init__.py and harness/decision.py
(those registrations are for the owner to add).

Episodes run on a thread pool: each sandbox call is its own node process, so
the threads overlap the spawns and keep the file under a minute.
"""
import copy
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.pipeline import build
from data.gen.brief_budget import check_brief
from harness import decision
from harness.authoring import resolve
from harness.context import build_context
from harness.oracles import rooms as oracle
from harness.run import run_sandbox
from runtime import worlds as W
from runtime.worlds import rooms

for _w in rooms.WORLDS:
    W.WORLDS[_w["name"]] = _w
for _name in (rooms.TRAIN_WORLD, rooms.EXAM_WORLD):
    decision.DECISION_WORLDS[_name] = ("runtime.worlds.rooms",
                                       "harness.oracles.rooms")

WORLDS = [rooms.TRAIN_WORLD, rooms.EXAM_WORLD]
EPISODES = 15


def run(world_name, state, src, seed=5):
    """Compile `src` against this state's observation and execute it, the
    world's end-of-turn hook included."""
    world = W.WORLDS[world_name]
    consts = rooms.observe(state).constants
    ctx, sctx = build_context(world, consts, random.Random(seed))
    res = build(resolve(src, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    return run_sandbox({
        "js": res.js, "state": state, "tools": sctx["tools"],
        "fields": sctx["fields"], "constants": sctx["constants"],
        "now": world["now"], "approval": True, "error_injection": [],
        "initial_registers": {}, "post_hook": world["post_hook"]})


def index_of(state, value):
    consts = rooms.observe(state).constants
    return next(i for i, c in enumerate(consts) if c["value"] == value)


def play(world_name, seed=None, scenario=None):
    """One episode played by the oracle, sampled or a named scenario.
    Returns (won, the states it passed through)."""
    state = (rooms.new_state(scenario) if scenario else rooms.sample_state(
        random.Random(f"{world_name}:{seed}"), world=world_name))
    seen = []
    while state["status"] == "playing" and state["turn"] < state["max_turns"]:
        check_brief(rooms.observe(state).brief)
        seen.append(copy.deepcopy(state))
        out = run(world_name, state, oracle.plan_turn(state, 3))
        assert out["status"] == "ok", (seed, scenario, out.get("error"))
        state = out["state"]
    return rooms.outcome(state)["won"], seen


@pytest.fixture(scope="module")
def played():
    jobs = [(w, s, None) for w in WORLDS for s in range(EPISODES)]
    jobs += [(w, None, name) for w in WORLDS for name in rooms.scenarios_for(w)]
    with ThreadPoolExecutor(12) as ex:
        return dict(zip(jobs, ex.map(lambda j: play(*j), jobs)))


# ------------------------------------------------------------ the oracle

@pytest.mark.parametrize("world_name", WORLDS)
def test_the_oracle_wins_every_sampled_episode(played, world_name):
    lost = [s for (w, s, name), (won, _) in played.items()
            if w == world_name and name is None and not won]
    assert not lost, f"{world_name}: oracle lost seeds {lost}"


def test_the_oracle_finishes_its_scenarios(played):
    lost = [name for (_, _, name), (won, _) in played.items()
            if name and not won]
    assert not lost


def test_every_legal_action_executes(played):
    """Every move `legal_actions` offers (the episode generator's off-path
    detours draw from it) runs without a failure, from states the oracle
    really passes through: every other turn of two episodes a world."""
    jobs = [(w, state, action)
            for (w, s, name), (_, states) in sorted(
                played.items(), key=lambda kv: str(kv[0]))
            if name is None and s < 2
            for state in states[::2]
            for action in rooms.legal_actions(state)]
    assert len(jobs) > 20

    def attempt(job):
        w, state, action = job
        out = run(w, copy.deepcopy(state), action)
        log = out["state"].get("log") or []
        ok = out["status"] == "ok" and not any(
            str(e).startswith("failed:") for e in log)
        return None if ok else (action, out.get("error"))

    with ThreadPoolExecutor(12) as ex:
        bad = [b for b in ex.map(attempt, jobs) if b]
    assert not bad, bad[:5]


# ------------------------------------------------------------ perception

def test_every_brief_fits_the_tiny_planner(played):
    # play() already raised on any brief over budget; this also covers the
    # observations after a failure, which carry the longest "Last turn"
    for _, (_, states) in played.items():
        for state in states:
            check_brief(rooms.observe(state).brief)
    state = rooms.new_state("key_then_door")
    out = run("rooms", state, "CALL @move $0\nCALL @move $2\nCALL @move $2\n"
              "STOP\n")
    after = out["state"]
    out = run("rooms", after, f"CALL @open ${index_of(after, 'door_1')}\n"
              "STOP\n")
    brief = rooms.observe(out["state"]).brief
    check_brief(brief)
    assert "failed: the blue door is locked - you need the blue key" in brief


def test_directions_say_what_is_one_step_away():
    state = rooms.new_state("key_then_door")        # agent (2,3), door (5,2)
    state["entities"]["agent"][0].update(x=4, y=2)
    desc = {c["value"]: c["desc"] for c in rooms.observe(state).constants}
    assert desc["east"] == "east: blue door door_1, locked"
    assert desc["north"] == "north: floor, you can walk here"
    assert desc["key_1"] == "blue key, 1 west 1 north"


def test_sampling_keeps_after_you_for_the_exam():
    # the exam holds out a phrasing of the order, not a tool: put-next-to is
    # trained, " after you" is not
    kinds, putnext = set(), 0
    for seed in range(60):
        train = rooms.sample_state(random.Random(seed), world="rooms")
        exam = rooms.sample_state(random.Random(seed), world="rooms_after")
        kinds.add(train["mission"]["connective"])
        putnext += any(s["verb"] == "putnext" for s in train["mission"]["steps"])
        assert " after you " not in train["mission"]["text"]
        assert exam["mission"]["connective"] == "after"
        assert " after you " in exam["mission"]["text"]
    assert "after" not in kinds and putnext > 0


# ----------------------------------------------------------------- rules

def test_a_locked_door_needs_its_key_in_hand():
    state = rooms.new_state("key_then_door")
    state["entities"]["agent"][0].update(x=4, y=2)   # beside the blue door
    out = run("rooms", state, f"CALL @open ${index_of(state, 'door_1')}\n"
              "STOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert out["error"]["message"] == \
        "the blue door is locked - you need the blue key"
    door = next(o for o in out["state"]["entities"]["object"]
                if o["id"] == "door_1")
    assert door["locked"] and not door["open"]

    state = rooms.new_state("key_then_door")
    state["entities"]["agent"][0].update(x=4, y=2)
    for o in state["entities"]["object"]:
        if o["id"] == "key_1":
            o["held"] = True
    out = run("rooms", state, f"CALL @open ${index_of(state, 'door_1')}\n"
              "CALL @move $2\nSTOP\n")
    assert out["status"] == "ok"
    assert out["state"]["mission"]["done"] == [True, False]
    agent = out["state"]["entities"]["agent"][0]
    assert (agent["x"], agent["y"]) == (5, 2)
    assert "room_2" in out["state"]["visited"]


def test_you_carry_one_object_at_a_time():
    state = rooms.new_state("key_then_door")      # agent (2,3), key (3,1)
    state["entities"]["agent"][0].update(x=3, y=2)
    key = index_of(state, "key_1")
    out = run("rooms", state, f"CALL @pick_up ${key}\nSTOP\n")
    assert out["status"] == "ok"
    state = out["state"]
    state["entities"]["agent"][0].update(x=2, y=4)   # beside the grey box
    out = run("rooms", state,
              f"CALL @pick_up ${index_of(state, 'box_1')}\nSTOP\n")
    assert out["error"]["message"] == \
        "your hands are full - drop the blue key first"
    held = [o["id"] for o in out["state"]["entities"]["object"] if o["held"]]
    assert held == ["key_1"]


def test_open_refuses_a_ball_and_says_what_to_do():
    state = rooms.new_state("key_then_door")
    state["entities"]["agent"][0].update(x=1, y=3)   # beside the grey box
    out = run("rooms", state, f"CALL @open ${index_of(state, 'box_1')}\n"
              "STOP\n")
    assert out["error"]["message"] == \
        "the grey box is not a door - pick it up instead"


def test_put_next_to_sets_the_load_beside_the_target():
    state = rooms.new_state("ball_by_the_box")   # purple box at (7,7)
    for o in state["entities"]["object"]:
        if o["id"] == "ball_1":
            o["held"] = True
    state["entities"]["agent"][0].update(x=6, y=7)
    state["visited"] = ["room_1", "room_2", "room_3", "room_4"]
    out = run("rooms", state,
              f"CALL @put_next_to ${index_of(state, 'box_1')}\nSTOP\n")
    assert out["status"] == "ok"
    ball = next(o for o in out["state"]["entities"]["object"]
                if o["id"] == "ball_1")
    assert (ball["x"], ball["y"], ball["held"]) == (7, 6, False)
    assert out["state"]["status"] == "won"
