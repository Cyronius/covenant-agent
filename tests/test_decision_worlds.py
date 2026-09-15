"""The decision worlds beside the dungeon: engine rules, oracles, and the
per-turn score the suite reads them with.

Same discipline as tests/test_rpg.py - every case goes through the real
pipeline (resolve -> build -> node sandbox), so what is asserted is what a
model's program would actually do, not what the engine would do if called
directly. Plan: .claude/plans/archive/task-families.md.
"""
import random

import pytest

from core.pipeline import build
from harness import decision
from harness.authoring import resolve
from harness.context import build_context
from harness.run import run_sandbox
from harness.rpg_suite import oracle_move, oracle_planner, run_episode
from runtime.worlds import cards, elevator, get_world, house, pages, warehouse

TRAINING = ["warehouse_robot", "elevator", "cards"]
EVERY = TRAINING + ["house", "app_settings", "app_checkout", "app_ticket",
                    "app_coursebuilder"]


def run(world_name, state, src, seed=5, post_hook=True):
    """Compile `src` against this state's own observation and execute it."""
    module = decision.world_module(world_name)
    world = get_world(world_name)
    consts = module.observe(state).constants
    ctx, sctx = build_context(world, consts, random.Random(seed))
    res = build(resolve(src, ctx), ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    payload = {"js": res.js, "state": state, "tools": sctx["tools"],
               "fields": sctx["fields"], "constants": sctx["constants"],
               "now": world["now"], "approval": True, "error_injection": [],
               "initial_registers": {}}
    if post_hook:
        payload["post_hook"] = world["post_hook"]
    return run_sandbox(payload), consts


def index_of(world_name, state, value):
    consts = decision.world_module(world_name).observe(state).constants
    return next(i for i, c in enumerate(consts) if c["value"] == value)


# ------------------------------------------------------------- the family

@pytest.mark.parametrize("name", EVERY)
def test_every_world_renders_a_turn_and_offers_moves(name):
    module = decision.world_module(name)
    state = decision.new_state(name)
    obs = module.observe(state)
    assert obs.request.strip()
    assert obs.constants, "a turn with no constants gives the model nothing"
    assert module.legal_actions(state), "no legal move on turn 0"


@pytest.mark.parametrize("name", EVERY)
def test_the_oracle_finishes_its_own_scenarios(name):
    """If the oracle cannot finish, the scenario is unwinnable and a model
    failing it proves nothing."""
    for scenario in decision.scenarios(name):
        budget = decision.default_actions(name)
        row = run_episode(oracle_planner(budget, name), world=name,
                          scenario=scenario, max_actions=budget,
                          max_turns=60, seed=0)
        assert row["won"], f"{name}/{scenario} ended {row['status']}"


@pytest.mark.parametrize("name", TRAINING)
def test_the_oracle_agrees_with_itself_every_turn(name):
    """The per-turn score (plan §4) has to read 100% when the oracle plays,
    or it is measuring something other than agreement."""
    budget = decision.default_actions(name)
    row = run_episode(oracle_planner(budget, name), world=name,
                      max_actions=budget, max_turns=60, seed=3)
    assert row["scored_turns"] > 0
    assert row["tool_match"] == row["scored_turns"]
    assert row["arg_match"] == row["scored_turns"]
    assert row["turn_match"] == row["scored_turns"]


def test_per_turn_score_catches_a_wrong_tool():
    """A planner that always stands is scored wrong on the turns the oracle
    would have hit."""
    def always_stand(input_text, turn_idx, state, ctx):
        return {"text": "CALL @stand\nSTOP\n", "authoring": True,
                "gen_ms": 0.0, "finish_reason": "test"}

    row = run_episode(always_stand, world="cards", max_actions=1,
                      max_turns=60, seed=0)
    assert row["scored_turns"] > row["tool_match"]


def test_scoring_only_the_first_call_flatters_a_planner_that_sprays():
    """Why the report leads with the whole turn. A planner that opens with
    the oracle's move and then tries every direction passes the first-call
    score and fails the one that counts — which is exactly what S3 does on
    the dungeon (results/FAMILIES.md §3)."""
    oracle = decision.oracle_module("warehouse_robot")

    def spray(input_text, turn_idx, state, ctx):
        # the oracle's own turn with one more move tacked on the end
        text = oracle.plan_turn(state, budget=3)
        lines = [ln for ln in text.splitlines() if ln.strip() != "STOP"]
        return {"text": "\n".join(lines + ["CALL @drive $1", "STOP"]) + "\n",
                "authoring": True, "gen_ms": 0.0, "finish_reason": "test"}

    row = run_episode(spray, world="warehouse_robot", max_actions=3,
                      max_turns=8, seed=0)
    scored = [t for t in row["turn_log"] if t["oracle_tool"]]
    assert scored
    assert sum(t["tool_match"] for t in scored) > sum(
        t["turn_match"] for t in scored)


def test_oracle_move_carries_the_whole_turn_not_just_its_opening():
    state = elevator.new_state("morning_rush")
    consts = elevator.observe(state).constants
    move = oracle_move(state, consts, "elevator", 2)
    assert move["tool"] in {"go_to", "open_doors", "hold"}
    assert move["calls"] and move["calls"][0][0] == move["tool"]
    if move["tool"] == "go_to":
        assert move["args"] and isinstance(move["args"][0], int)


def test_a_theme_cannot_shadow_an_engine_world():
    """gen_warehouse.json is a theme domain called `warehouse`, so the
    decision world had to be renamed for it. This is the guard that makes the
    next collision an error instead of a corpus that quietly means something
    else."""
    from data.gen.domains import register_theme

    with pytest.raises(ValueError, match="shadow"):
        register_theme({"domain": "elevator"})


# ---------------------------------------------------------------- warehouse

def test_driving_costs_battery_and_racks_block():
    st = warehouse.new_state("cross_dock")       # robot at (1,1)
    out, _ = run("warehouse_robot", st, "CALL @drive $1\nSTOP\n")   # $1 = south
    robot = out["state"]["entities"]["robot"][0]
    assert (robot["x"], robot["y"]) == (1, 2)
    assert robot["battery"] == 11

    out, _ = run("warehouse_robot", warehouse.new_state("cross_dock"),
                 "CALL @drive $0\nSTOP\n")                    # $0 = north
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert "rack" in out["error"]["message"]


def test_a_tote_can_only_be_lifted_from_the_bay_it_is_on():
    st = warehouse.new_state("cross_dock")
    i = index_of("warehouse_robot", st, "tote_2")      # T-31, three bays away
    out, _ = run("warehouse_robot", st, f"CALL @lift ${i}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert not out["state"]["entities"]["tote"][1]["held"]


def test_charging_only_works_on_the_pad():
    st = warehouse.new_state("cross_dock")
    out, _ = run("warehouse_robot", st, "CALL @charge\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"

    st = warehouse.new_state("cross_dock")
    pad = st["entities"]["charger"][0]
    st["entities"]["robot"][0].update(x=pad["x"], y=pad["y"], battery=3)
    out, _ = run("warehouse_robot", st, "CALL @charge\nSTOP\n")
    assert out["status"] == "ok"
    assert out["state"]["entities"]["robot"][0]["battery"] == 20


def test_a_flat_battery_away_from_the_pad_strands_the_robot():
    st = warehouse.new_state("cross_dock")
    st["entities"]["robot"][0]["battery"] = 1
    out, _ = run("warehouse_robot", st, "CALL @drive $1\nSTOP\n")
    assert out["state"]["status"] == "stranded"
    assert warehouse.outcome(out["state"])["dead"]


@pytest.mark.parametrize("name", TRAINING + ["app_settings", "app_checkout",
                                             "app_ticket"])
def test_sampled_jobs_are_solvable(name):
    """The corpus draws from `sample_state`, so an unwinnable draw is a
    training row that teaches a dead end. Played straight (no detours), the
    oracle has to finish every one."""
    from data.gen import episodes

    for seed in range(6):
        # NOT `(seed, name).__hash__()`: str hashing is salted per process
        # (PYTHONHASHSEED), so that seeded a *different* six draws every run
        # and the test passed or failed at random. Measured 2026-09-15:
        # 2.1% of warehouse_robot draws strand the robot, i.e. ~12% of runs
        # failed. Six fixed draws is a smoke check, not a guarantee — the
        # sampler's dead-end rate is a generator bug, tracked separately.
        rng = random.Random(f"{name}:{seed}")
        _, outcome = episodes.run_episode(
            name, seed, rng, symbols="classic", enums=False, kinds=False,
            offpath=0.0, illegal_share=0.0)
        assert outcome["won"], f"{name} seed {seed} ended {outcome['status']}"


# ----------------------------------------------------------------- elevator

def test_holding_is_a_legal_move_that_only_spends_a_tick():
    st = elevator.new_state("late_call")
    out, _ = run("elevator", st, "CALL @hold\nSTOP\n")
    assert out["status"] == "ok"
    assert out["state"]["tick"] == 1
    assert out["state"]["entities"]["car"][0]["floor"] == 4


def test_the_oracle_holds_for_a_rider_who_is_about_to_arrive():
    """late_call puts Kai on the car's own floor in two ticks and the nearest
    other job two floors away: the tie goes to holding."""
    st = elevator.new_state("late_call")
    assert "hold" in decision.oracle_module("elevator").plan_turn(st, 2)


def test_opening_the_doors_exchanges_riders_and_costs_a_tick():
    st = elevator.new_state("morning_rush")      # Otto waiting on floor 1
    out, _ = run("elevator", st, "CALL @open_doors\nSTOP\n")
    riders = {r["name"]: r for r in out["state"]["entities"]["rider"]}
    assert riders["Otto"]["aboard"]
    assert out["state"]["tick"] == 1


def test_travel_costs_one_tick_a_floor():
    st = elevator.new_state("morning_rush")      # car on floor 1
    i = index_of("elevator", st, 5)
    out, _ = run("elevator", st, f"CALL @go_to ${i}\nSTOP\n")
    assert out["state"]["entities"]["car"][0]["floor"] == 5
    assert out["state"]["tick"] == 4


def test_going_to_the_floor_you_are_on_is_refused():
    st = elevator.new_state("morning_rush")
    i = index_of("elevator", st, 1)
    out, _ = run("elevator", st, f"CALL @go_to ${i}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"


# -------------------------------------------------------------------- cards

def test_doubling_is_refused_after_the_first_two_cards():
    st = cards.new_state("eight_hands")
    out, _ = run("cards", st, "CALL @hit\nSTOP\n")
    out2, _ = run("cards", out["state"], "CALL @double_down\nSTOP\n")
    assert out2["error"]["code"] == "INVALID_ARGUMENT"


def test_a_settled_hand_deals_the_next_one():
    st = cards.new_state("eight_hands")
    before = st["entities"]["table"][0]["hand_no"]
    out, _ = run("cards", st, "CALL @stand\nSTOP\n")
    table = out["state"]["entities"]["table"][0]
    assert table["hand_no"] == before + 1
    assert table["cards_taken"] == 0


def test_basic_strategy_table_matches_the_rules_it_claims():
    from harness.oracles.cards import DOUBLE, HIT, STAND, decide
    assert decide(16, False, 10, True) == HIT       # hard 16 vs ten
    assert decide(16, False, 6, True) == STAND      # hard 16 vs six
    assert decide(11, False, 5, True) == DOUBLE     # eleven vs five
    assert decide(11, False, 1, True) == HIT        # eleven vs an ace
    assert decide(18, True, 9, True) == HIT         # soft 18 vs nine
    assert decide(18, True, 7, True) == STAND       # soft 18 vs seven
    assert decide(11, False, 5, False) == HIT       # cannot double any more


def test_the_python_deal_and_the_engine_agree_on_totals():
    """The opening deal is the one piece of card logic on both sides."""
    st = cards.new_state("short_shoe")
    table = st["entities"]["table"][0]
    out, _ = run("cards", st, "CALL @hit\nSTOP\n")
    # the engine re-syncs the same fields the python deal wrote
    after = out["state"]["entities"]["table"][0]
    assert after["cards_taken"] in (1, 0)     # 0 once the hand settled
    assert after["dealer_up"] in range(1, 11)
    assert table["player_total"] <= 21


# -------------------------------------------------------------------- house

def test_a_locked_way_needs_its_key_in_hand():
    st = house.new_state("cellar_run")
    st["entities"]["explorer"][0]["room"] = "room_4"     # pantry
    i = index_of("house", st, "way_11")                  # down to the cellar
    out, _ = run("house", st, f"CALL @open ${i}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert "no key" in out["error"]["message"]


def test_a_dark_room_turns_you_back_without_a_lit_lamp():
    st = house.new_state("cellar_run")
    st["entities"]["explorer"][0]["room"] = "room_4"     # pantry
    for way in st["entities"]["way"]:
        if way["id"] == "way_11":
            way.update(shut=False, locked=False)
    i = index_of("house", st, "way_11")
    out, _ = run("house", st, f"CALL @go ${i}\nSTOP\n")
    assert out["status"] == "ok"
    assert out["state"]["entities"]["explorer"][0]["room"] == "room_4"
    assert "pitch dark" in " ".join(out["state"]["log"])


# -------------------------------------------------------------------- pages

def test_clicking_a_save_button_says_to_submit_it_instead():
    st = pages.new_state("ticket_urgent")
    i = index_of("app_ticket", st, "element_7")          # Raise ticket
    out, _ = run("app_ticket", st, f"CALL @click ${i}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert "submit" in out["error"]["message"]


def test_a_dropdown_refuses_a_value_that_is_not_one_of_its_choices():
    st = pages.new_state("ticket_urgent")
    field = index_of("app_ticket", st, "element_3")      # Priority
    value = index_of("app_ticket", st, "Door badge reader down")
    out, _ = run("app_ticket", st, f"CALL @fill ${field} ${value}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"


def test_a_form_will_not_save_while_a_required_field_is_empty():
    st = pages.new_state("ticket_urgent")
    i = index_of("app_ticket", st, "element_7")
    out, _ = run("app_ticket", st, f"CALL @submit ${i}\nSTOP\n")
    assert out["error"]["code"] == "INVALID_ARGUMENT"
    assert "empty" in out["error"]["message"]


def test_a_sampled_page_job_is_always_reachable_and_saveable():
    for seed in range(25):
        st = pages.sample_state(random.Random(seed))
        app = pages._app(st)["screen"]
        want = {pages.screen_by_id(st, e["screen"])["id"]
                for e in st["entities"]["element"]
                if e["id"] in st["goal"]["values"]}
        from harness.oracles.pages import _hop
        for target in want:
            assert target == app or _hop(st, app, target), seed
        # every required field on a screen the job saves has to be named
        for eid in st["goal"]["submitted"]:
            screen = next(e["screen"] for e in st["entities"]["element"]
                          if e["id"] == eid)
            for el in pages.elements_on(st, screen):
                if el["required"] and not el["value"]:
                    assert el["id"] in st["goal"]["values"], seed


# ------------------------------------------------------------------ holdout

def test_the_held_out_worlds_are_reserved_and_refused_by_the_corpus():
    import json
    from pathlib import Path

    from data.gen import episodes

    reserved = set(json.loads(
        (Path(__file__).resolve().parent.parent / "data" / "holdout"
         / "reserved.json").read_text())["worlds"])
    assert decision.HELD_OUT <= reserved
    assert not (set(decision.TRAINABLE) & reserved)
    assert episodes.level_for("warehouse_robot") == episodes.LEVEL_DECISION
    assert episodes.level_for("app_settings") == episodes.LEVEL_PAGE
