"""The workshop crafting worlds (`workshop`, and `boatyard`, its held-out
recipe tree): engine rules, the oracle, the brief budget.

Every case runs through the real pipeline (resolve -> build -> node
sandbox), as tests/test_decision_worlds.py does. The sandbox is one node
process per program, so the episodes run on a thread pool to keep the file
inside a minute.

Plan: .claude/plans/borrowed-worlds.md, "2. Crafting".
"""
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.pipeline import build
from data.gen.brief_budget import check_brief
from harness import decision
from harness.authoring import resolve
from harness.context import build_context
from harness.run import run_sandbox
from runtime import worlds as W
from runtime.worlds import workshop

# Registered here rather than in runtime/worlds/__init__.py and
# harness/decision.py until those shared files take the new names.
for _w in workshop.WORLDS:
    W.WORLDS[_w["name"]] = _w
decision.DECISION_WORLDS.setdefault(
    "workshop", ("runtime.worlds.workshop", "harness.oracles.workshop"))
decision.DECISION_WORLDS.setdefault(
    "boatyard", ("runtime.worlds.workshop", "harness.oracles.workshop"))

from harness.oracles import workshop as oracle  # noqa: E402

POOL = 8


def run(world_name, state, src, post_hook=True, seed=5):
    """Compile `src` against this state's own observation and execute it."""
    world = W.get_world(world_name)
    consts = workshop.observe(state).constants
    ctx, sctx = build_context(world, consts, random.Random(seed),
                              symbols="typed", enums=True)
    res = build(resolve(src, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    payload = {"js": res.js, "state": state, "tools": sctx["tools"],
               "fields": sctx["fields"], "constants": sctx["constants"],
               "now": world["now"], "approval": True, "error_injection": [],
               "initial_registers": {}}
    if post_hook:
        payload["post_hook"] = world["post_hook"]
    return run_sandbox(payload)


def ref(state, name):
    """`$i` for the shown item called `name`."""
    idx = workshop.by_id(state)
    i = next(n for n, iid in enumerate(state["shown"])
             if idx[iid]["name"] == name)
    return f"${i}"


def item(state, name):
    return next(i for i in state["entities"]["item"] if i["name"] == name)


def play(world_name, seed):
    """The oracle plays one sampled job to the end. Returns (final state,
    every brief it was shown)."""
    state = workshop.sample_state(random.Random(f"{world_name}:{seed}"),
                                  world=world_name)
    briefs = []
    while state["status"] == "playing" and state["turn"] < state["max_turns"]:
        briefs.append(workshop.observe(state).brief)
        out = run(world_name, state, oracle.plan_turn(state, budget=3),
                  seed=seed)
        assert out["status"] == "ok", (world_name, seed, out.get("error"))
        state = out["state"]
    briefs.append(workshop.observe(state).brief)
    return state, briefs


EPISODES = [("workshop", s) for s in range(11)] + \
    [("boatyard", s) for s in range(4)]


def test_the_oracle_finishes_every_sampled_job_and_every_brief_fits():
    """An unwinnable draw is a corpus row that teaches a dead end, and a
    brief over the tiny planner's 128 tokens is one it cannot read whole."""
    with ThreadPoolExecutor(POOL) as ex:
        results = list(ex.map(lambda e: play(*e), EPISODES))
    for (world_name, seed), (final, briefs) in zip(EPISODES, results):
        assert final["status"] == "done", (world_name, seed, final["turn"])
        assert workshop.outcome(final)["won"]
        assert workshop.outcome(final)["funnel"]["goal_made"]
        for text in briefs:
            check_brief(text)


def _mid_game_states():
    """States with every kind of legal move on offer: stations up,
    ingredients in hand for several recipes, a distractor on the shelf."""
    stocked = workshop.new_state("lantern_from_scratch")
    for name, n in (("workbench", 1), ("kiln", 1), ("plank", 3),
                    ("sand", 2), ("timber", 1), ("flax", 2),
                    ("beeswax", 1), ("twine", 1), ("wool", 2)):
        item(stocked, name)["have"] = n
    boat = workshop.new_state("boat_oars")
    item(boat, "plank")["have"] = 2
    item(boat, "flax")["have"] = 2
    return [("workshop", stocked), ("boatyard", boat)]


def test_every_legal_action_executes_without_a_failure():
    """The episode generator's off-path detours draw from `legal_actions`
    and subtract it to find illegal moves, so an entry the engine refuses
    would mislabel a detour."""
    jobs = []
    for world_name, state in _mid_game_states():
        actions = workshop.legal_actions(state)
        assert any("@craft" in a for a in actions)
        assert any("@gather" in a for a in actions)
        jobs += [(world_name, state, a) for a in actions]
    with ThreadPoolExecutor(POOL) as ex:
        outs = list(ex.map(lambda j: run(j[0], j[1], j[2], post_hook=False),
                           jobs))
    for (world_name, _, action), out in zip(jobs, outs):
        assert out["status"] == "ok", (world_name, action, out.get("error"))
        assert not any(str(e).startswith("failed:")
                       for e in out["state"]["log"]), (action,
                                                       out["state"]["log"])


def test_crafting_short_of_ingredients_fails_and_names_what_to_get():
    st = workshop.new_state("lantern_from_scratch")      # 1 timber, no plank
    out = run("workshop", st, f"CALL @craft {ref(st, 'workbench')}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert "short 4 plank" in out["error"]["message"]
    assert "craft plank first" in out["error"]["message"]
    # validate then mutate: nothing was used up, nothing was made
    after = out["state"]
    assert item(after, "workbench")["have"] == 0
    assert item(after, "timber")["have"] == 1
    # and the refusal reaches the next turn's brief verbatim
    assert f"failed: {out['error']['message']}" in \
        workshop.observe(after).brief


def test_a_recipe_needs_its_station_standing_and_does_not_use_it_up():
    st = workshop.new_state("lantern_from_scratch")
    item(st, "sand")["have"] = 2
    out = run("workshop", st, f"CALL @craft {ref(st, 'glass pane')}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert "craft the kiln first" in out["error"]["message"]
    assert item(out["state"], "sand")["have"] == 2

    item(st, "kiln")["have"] = 1
    out = run("workshop", st, f"CALL @craft {ref(st, 'glass pane')}\nSTOP\n")
    assert out["status"] == "ok"
    assert item(out["state"], "glass pane")["have"] == 1
    assert item(out["state"], "sand")["have"] == 0
    assert item(out["state"], "kiln")["have"] == 1


def test_gather_and_craft_refuse_each_others_items_and_say_which_applies():
    st = workshop.new_state("lantern_from_scratch")
    out = run("workshop", st, f"CALL @gather {ref(st, 'plank')}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert "craft it from 1 timber" in out["error"]["message"]

    out = run("workshop", st, f"CALL @craft {ref(st, 'timber')}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert "gather it instead" in out["error"]["message"]

    out = run("workshop", st, f"CALL @gather {ref(st, 'timber')}\nSTOP\n")
    assert item(out["state"], "timber")["have"] == 3        # 1 + a trip of 2


def test_the_boatyard_tree_never_appears_in_a_workshop_job():
    """`boatyard` is the held-out exam: a workshop state that showed a hull
    plank or an oar would leak the exam's recipes into training."""
    for seed in range(200):
        st = workshop.sample_state(random.Random(seed), world="workshop")
        names = {i["name"] for i in st["entities"]["item"]}
        assert not names & workshop.BOATYARD_ONLY, (seed,
                                                    names & workshop.BOATYARD_ONLY)
    for seed in range(50):
        st = workshop.sample_state(random.Random(seed), world="boatyard")
        goal = workshop.by_id(st)[st["goal"]["item"]]["name"]
        assert goal in workshop.BOATYARD_ONLY, (seed, goal)


@pytest.mark.parametrize("world_name", ["workshop", "boatyard"])
def test_scenarios_are_filtered_by_world(world_name):
    names = workshop.scenarios_for(world_name)
    assert names
    for n in names:
        assert workshop.new_state(n)["world_name"] == world_name
