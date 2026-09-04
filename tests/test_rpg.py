"""RPG world: engine rules, the turn budget, the enemy phase, and perception.

Every case runs through the real pipeline (resolve -> build -> node sandbox),
never by calling the engine directly, so what is asserted is what a model's
program would actually do. Plan: .claude/plans/rpg-demo-app.md.
"""
import random

from core.pipeline import build
from harness.authoring import resolve
from harness.context import build_context
from harness.run import run_sandbox
from runtime.worlds import get_world, rpg

WORLD = get_world("rpg")


def run(state, src, seed=5, constants=None, post_hook=True):
    """Compile `src` (authoring form) against this state's own observation and
    execute it. Returns (sandbox result, constants used)."""
    consts = rpg.observe(state).constants if constants is None else constants
    ctx, sctx = build_context(WORLD, consts, random.Random(seed))
    res = build(resolve(src, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    payload = {"js": res.js, "state": state, "tools": sctx["tools"],
               "fields": sctx["fields"], "constants": sctx["constants"],
               "now": WORLD["now"], "approval": True, "error_injection": [],
               "initial_registers": {}}
    if post_hook:
        payload["post_hook"] = WORLD["post_hook"]
    return run_sandbox(payload), consts


def player(state):
    return state["entities"]["player"][0]


def const_index(state, value):
    consts = rpg.observe(state).constants
    return next(i for i, c in enumerate(consts) if c["value"] == value)


# ------------------------------------------------------------------ moving

def test_move_walks_and_blocks_on_walls():
    st = rpg.new_state()             # player at (1,7), wall west and south
    out, _ = run(st, "CALL @move $2\nSTOP\n")          # $2 = east
    assert out["status"] == "ok"
    assert (player(out["state"])["x"], player(out["state"])["y"]) == (2, 7)

    out, _ = run(rpg.new_state(), "CALL @move $3\nSTOP\n")   # $3 = west, wall
    assert out["status"] == "error"
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert out["calls"][0]["ok"] is False
    assert (player(out["state"])["x"], player(out["state"])["y"]) == (1, 7)


def test_move_blocked_by_closed_door_and_enemy():
    st = rpg.new_state()
    player(st).update(x=6, y=3)                 # door_1 is the tile east
    out, _ = run(st, "CALL @move $2\nSTOP\n")
    assert out["status"] == "error"
    assert "locked door" in out["error"]["message"]

    st = rpg.new_state()
    player(st).update(x=1, y=6)
    st["entities"]["enemy"][0].update(x=2, y=6)   # standing in the way
    out, _ = run(st, "CALL @move $2\nSTOP\n")
    assert out["status"] == "error"
    assert "an enemy" in out["error"]["message"]


def test_stepping_onto_the_stairs_wins():
    st = rpg.new_state()
    player(st).update(x=8, y=2)                  # exit E is at (9,2)
    out, _ = run(st, "CALL @move $2\nSTOP\n")
    assert out["status"] == "ok"
    assert out["state"]["status"] == "won"
    assert rpg.outcome(out["state"])["won"] is True


# ------------------------------------------------------------------ combat

def test_attack_requires_adjacency_and_kills():
    st = rpg.new_state()
    player(st).update(x=1, y=6)
    st["entities"]["enemy"][0].update(x=3, y=6, hp=3)     # two tiles away
    out, _ = run(st, "CALL @attack $%d\nSTOP\n" % const_index(st, "enemy_1"))
    assert out["status"] == "error"
    assert "not adjacent" in out["error"]["message"]
    assert out["state"]["entities"]["enemy"][0]["hp"] == 3   # unharmed

    st = rpg.new_state()
    player(st).update(x=1, y=6, attack=3)
    st["entities"]["enemy"][0].update(x=2, y=6, hp=3)
    out, _ = run(st, "CALL @attack $%d\nSTOP\n" % const_index(st, "enemy_1"))
    assert out["status"] == "ok"
    assert [e["id"] for e in out["state"]["entities"]["enemy"]] == ["enemy_2"]


def test_wounded_enemy_survives_and_fights_back():
    st = rpg.new_state()
    player(st).update(x=1, y=6, attack=1, hp=10)
    st["entities"]["enemy"][0].update(x=2, y=6, hp=3, attack=2)
    out, _ = run(st, "CALL @attack $%d\nSTOP\n" % const_index(st, "enemy_1"))
    assert out["status"] == "ok"
    enemy = next(e for e in out["state"]["entities"]["enemy"]
                 if e["id"] == "enemy_1")
    assert enemy["hp"] == 2
    assert player(out["state"])["hp"] == 8      # enemy phase struck back


# ------------------------------------------------------------------- items

def test_pick_up_only_on_your_own_tile():
    st = rpg.new_state()
    player(st).update(x=2, y=6)                 # potion is at (2,5)
    out, _ = run(st, "CALL @pick_up $%d\nSTOP\n" % const_index(st, "item_1"))
    assert out["status"] == "error"
    assert "not on your tile" in out["error"]["message"]

    st = rpg.new_state()
    player(st).update(x=2, y=5)
    out, _ = run(st, "CALL @pick_up $%d\nSTOP\n" % const_index(st, "item_1"))
    assert out["status"] == "ok"
    potion = next(i for i in out["state"]["entities"]["item"]
                  if i["id"] == "item_1")
    assert potion["held"] is True


def test_potion_heals_and_is_consumed():
    st = rpg.new_state()
    player(st).update(x=2, y=5, hp=4, max_hp=10)
    st["entities"]["item"][0].update(held=True, x=-1, y=-1)
    out, _ = run(st, "CALL @use_item $%d\nSTOP\n" % const_index(st, "item_1"))
    assert out["status"] == "ok"
    assert player(out["state"])["hp"] == 9
    assert [i["id"] for i in out["state"]["entities"]["item"]] == ["item_2"]


def test_using_a_key_directs_you_to_the_door():
    st = rpg.new_state()
    st["entities"]["item"][1].update(held=True, x=-1, y=-1)
    out, _ = run(st, "CALL @use_item $%d\nSTOP\n" % const_index(st, "item_2"))
    assert out["status"] == "error"
    assert "interact" in out["error"]["message"]


# ------------------------------------------------------------------- doors

def test_locked_door_needs_the_key_and_consumes_it():
    st = rpg.new_state()
    player(st).update(x=6, y=3)
    out, _ = run(st, "CALL @interact $%d\nSTOP\n" % const_index(st, "door_1"))
    assert out["status"] == "error"
    assert "no key" in out["error"]["message"]
    assert out["state"]["entities"]["door"][0]["locked"] is True

    st = rpg.new_state()
    player(st).update(x=6, y=3)
    st["entities"]["item"][1].update(held=True, x=-1, y=-1)  # carrying the key
    out, _ = run(st, "CALL @interact $%d\nSTOP\n" % const_index(st, "door_1"))
    assert out["status"] == "ok"
    door = out["state"]["entities"]["door"][0]
    assert door["locked"] is False and door["open"] is True
    assert "item_2" not in [i["id"] for i in out["state"]["entities"]["item"]]
    assert rpg.outcome(out["state"])["door_opened"] is True


# -------------------------------------------------------------- turn phase

def test_enemy_closes_in_once_per_turn_not_once_per_call():
    st = rpg.new_state()
    player(st).update(x=1, y=6)
    st["entities"]["enemy"][0].update(x=4, y=6)      # 3 away: inside aggro
    out, _ = run(st, "CALL @move $1\nCALL @move $0\nSTOP\n")   # south, north
    enemy = next(e for e in out["state"]["entities"]["enemy"]
                 if e["id"] == "enemy_1")
    assert out["status"] == "ok"
    assert enemy["x"] == 3                # moved exactly one tile, not two
    assert out["state"]["turn"] == 1


def test_distant_enemies_stay_put():
    st = rpg.new_state()                  # both goblins are far from (1,7)
    out, _ = run(st, "CALL @move $2\nSTOP\n")
    assert [(e["x"], e["y"]) for e in out["state"]["entities"]["enemy"]] \
        == [(3, 2), (9, 7)]


def test_enemy_phase_runs_even_when_the_program_errors():
    st = rpg.new_state()
    player(st).update(x=1, y=6)
    st["entities"]["enemy"][0].update(x=3, y=6)
    out, _ = run(st, "CALL @move $3\nSTOP\n")        # west into a wall
    assert out["status"] == "error"
    enemy = next(e for e in out["state"]["entities"]["enemy"]
                 if e["id"] == "enemy_1")
    assert enemy["x"] == 2                            # the turn still passed
    assert out["state"]["turn"] == 1


def test_no_post_hook_means_no_enemy_phase():
    st = rpg.new_state()
    player(st).update(x=1, y=6)
    st["entities"]["enemy"][0].update(x=3, y=6)
    out, _ = run(st, "CALL @move $2\nSTOP\n", post_hook=False)
    assert out["status"] == "ok"
    enemy = next(e for e in out["state"]["entities"]["enemy"]
                 if e["id"] == "enemy_1")
    assert (enemy["x"], out["state"]["turn"]) == (3, 0)


def test_death_ends_the_game():
    st = rpg.new_state()
    player(st).update(x=1, y=6, hp=2, attack=1)
    st["entities"]["enemy"][0].update(x=2, y=6, hp=3, attack=2)
    # trading blows on 2 HP: the enemy phase finishes the player off
    out, _ = run(st, "CALL @attack $%d\nSTOP\n" % const_index(st, "enemy_1"))
    assert out["state"]["status"] == "dead"
    assert player(out["state"])["hp"] == 0
    assert rpg.outcome(out["state"])["dead"] is True

    again, _ = run(out["state"], "CALL @move $0\nSTOP\n")
    assert again["status"] == "error"
    assert "dead" in again["error"]["message"]


# ------------------------------------------------------------ turn budget

def test_turn_budget_caps_actions_and_keeps_the_earlier_ones():
    st = rpg.new_state()
    player(st).update(x=1, y=6)
    src = "CALL @move $2\nCALL @move $2\nCALL @move $2\nCALL @move $2\nSTOP\n"
    out, _ = run(st, src)
    assert out["status"] == "error"
    assert out["error"]["code"] == "RATE_LIMITED"
    assert player(out["state"])["x"] == 4          # three moves stood
    assert [c["ok"] for c in out["calls"]] == [True, True, True, False]


def test_invalid_actions_still_cost_budget():
    # a TRY-wrapped program must not get to probe the map for free
    st = rpg.new_state()
    player(st).update(x=1, y=6)
    src = ("TRY -> r0\n  CALL @move $3\nTRY -> r1\n  CALL @move $3\n"
           "TRY -> r2\n  CALL @move $3\nCALL @move $2\nSTOP\n")
    out, _ = run(st, src)
    assert out["status"] == "error"
    assert out["error"]["code"] == "RATE_LIMITED"
    assert player(out["state"])["x"] == 1           # never actually moved


# -------------------------------------------------------------- perception

def test_observation_window_offsets_and_constants():
    st = rpg.new_state()
    obs = rpg.observe(st)

    assert obs.window == ["##.p.", "##...", "##@..", "#####", "#####"]
    assert [c["value"] for c in obs.constants[:4]] == \
        ["north", "south", "east", "west"]

    # only what is visible becomes a constant: the potion, not the far key,
    # the door or either goblin
    assert [c["value"] for c in obs.constants[4:]] == ["item_1"]
    assert "1 east 2 north" in obs.constants[4]["desc"]
    assert obs.memory == ["item_1"]

    st["memory"] = obs.memory
    player(st).update(x=5, y=7)                     # potion out of the window
    later = rpg.observe(st)
    assert [c["value"] for c in later.constants[4:]] == []
    assert "potion" in later.request.split("Seen earlier")[1]


def test_carried_items_stay_referenceable_out_of_sight():
    st = rpg.new_state()
    st["entities"]["item"][1].update(held=True, x=-1, y=-1)
    obs = rpg.observe(st)
    key = next(c for c in obs.constants if c["value"] == "item_2")
    assert "carrying" in key["desc"]
    assert "carrying key" in obs.request


def test_adjacency_is_stated_for_enemies_and_doors():
    st = rpg.new_state()
    player(st).update(x=6, y=3)
    st["entities"]["enemy"][0].update(x=6, y=2)
    obs = rpg.observe(st)
    enemy = next(c for c in obs.constants if c["value"] == "enemy_1")
    door = next(c for c in obs.constants if c["value"] == "door_1")
    assert "1 north, adjacent" in enemy["desc"]
    assert "locked door 1 east, adjacent" in door["desc"]


# ----------------------------------------------------------------- oracle

def test_oracle_wins_the_keep_through_the_real_pipeline():
    # the scenario has to be winnable by a competent player, or a model
    # failing it proves nothing (harness/rpg_oracle.py)
    from harness.rpg_suite import oracle_planner, run_episode

    row = run_episode(oracle_planner(3), seed=0, max_turns=40, max_actions=3)
    assert row["won"] is True, row["status"]
    assert row["turns"] <= 25
    assert row["key_taken"] and row["door_opened"]
    assert row["compile_failures"] == 0 and row["invalid_calls"] == 0


def test_the_event_log_holds_one_turn_not_the_whole_episode():
    # "Last turn:" in the observation is built from state["log"]; if it were
    # never cleared the prompt would fill with the run's whole history
    # (caught 2026-09-04 on an oracle run: 18 events by turn 7).
    from harness.rpg_suite import oracle_planner, run_episode

    row = run_episode(oracle_planner(3), seed=0, max_turns=10)
    for turn in row["turn_log"]:
        last = turn["request"].split("Last turn: ")[1].split("\n")[0]
        assert last.count(";") <= 6, (turn["turn"], last)
