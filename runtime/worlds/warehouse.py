"""Warehouse robot: family A's grid instance, rendered as prose.

Same skill as the dungeon - partial view, small action set, choose by
description, adapt next turn - with two things changed on purpose. The floor
is described in words rather than drawn in glyphs, so nothing here teaches
grid reading; and the battery makes the shortest route to the tote often the
wrong first move.

Rules live in `runtime/engines/warehouse.js` and run in the sandbox through
the `engine` impl op, as the RPG's do. This module owns the schema, the
scenarios and perception; it never mutates the game.
"""
from __future__ import annotations

import random
from typing import List, Optional

from runtime.worlds.decision import Observation, last_turn, offset_words

NOW = 1_760_000_000

RACK = "#"
DIRECTIONS = ["north", "south", "east", "west"]
DIR_DESC = {
    "north": "direction: north (up the aisle, bay-1)",
    "south": "direction: south (down the aisle, bay+1)",
    "east": "direction: east (one aisle right, aisle+1)",
    "west": "direction: west (one aisle left, aisle-1)",
}

WORLD = {
    # not "warehouse": data/gen/themes/gen_warehouse.json is a theme
    # domain of that name, and register_domains would shadow this
    "name": "warehouse_robot",
    "now": NOW,
    "entities": {
        "robot": {"id": "ID:robot", "x": "INT", "y": "INT",
                  "battery": "INT", "max_battery": "INT"},
        "tote": {"id": "ID:tote", "label": "STR", "x": "INT", "y": "INT",
                 "held": "BOOL"},
        "dock": {"id": "ID:dock", "x": "INT", "y": "INT"},
        "charger": {"id": "ID:charger", "x": "INT", "y": "INT"},
    },
    "tools": [
        {
            "name": "drive",
            "desc": "Drive one bay in a direction. Fails if a rack is in the "
                    "way. Costs one unit of battery.",
            "params": [{"name": "direction", "type": "STR",
                        "desc": "which way to drive: north, south, east "
                                "or west"}],
            "returns": "OBJ:robot",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "warehouse", "fn": "drive"},
        },
        {
            "name": "lift",
            "desc": "Lift a tote off the bay you are standing on. Fails if "
                    "the tote is anywhere else, or if you already carry one.",
            "params": [{"name": "tote", "type": "ID:tote",
                        "desc": "the tote on this bay",
                        "field": ["tote", "id"]}],
            "returns": "OBJ:tote",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "warehouse", "fn": "lift"},
        },
        {
            "name": "set_down",
            "desc": "Put the tote you are carrying down on the bay you are "
                    "standing on. On the outbound dock, that completes it.",
            "params": [{"name": "tote", "type": "ID:tote",
                        "desc": "the tote you are carrying",
                        "field": ["tote", "id"]}],
            "returns": "OBJ:tote",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "warehouse", "fn": "set_down"},
        },
        {
            "name": "charge",
            "desc": "Fill the battery back to full. Works only while you are "
                    "standing on the charging pad.",
            "params": [],
            "returns": "OBJ:robot",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "warehouse", "fn": "charge"},
        },
    ],
    "post_hook": {"module": "warehouse", "fn": "end_turn"},
}

FLOOR_ROWS = [
    "############",
    "#..........#",
    "#.##.##.##.#",
    "#.##.##.##.#",
    "#.##.##.##.#",
    "#..........#",
    "#.##.##.##.#",
    "#..........#",
    "############",
]

# bays a robot can stand on, from the layout above
OPEN_BAYS = [(x, y) for y, row in enumerate(FLOOR_ROWS)
             for x, ch in enumerate(row) if ch != RACK]

LABELS = ["T-12", "T-31", "T-07", "T-44", "T-58", "T-23", "T-66", "T-19",
          "T-85", "T-40"]

SCENARIOS = {
    "cross_dock": {
        "quest": "Bring tote T-12 out to the outbound dock. The pad is the "
                 "only place to charge.",
        "vision": 3,
        "max_turns": 25,
        "robot": {"x": 1, "y": 1, "battery": 12, "max_battery": 20},
        "totes": [{"label": "T-12", "x": 7, "y": 4},
                  {"label": "T-31", "x": 4, "y": 3}],
        "job": ["tote_1"],
        "dock": {"x": 10, "y": 1},
        "charger": {"x": 4, "y": 5},
    },
    "long_haul": {
        "quest": "Bring tote T-44 out to the outbound dock. The pad is the "
                 "only place to charge.",
        "vision": 3,
        "max_turns": 30,
        # the pack has to cover pad -> tote -> dock (19 bays), or the job is
        # unwinnable however well it is played
        "robot": {"x": 10, "y": 7, "battery": 9, "max_battery": 24},
        "totes": [{"label": "T-44", "x": 1, "y": 3},
                  {"label": "T-07", "x": 10, "y": 4}],
        "job": ["tote_1"],
        "dock": {"x": 10, "y": 1},
        "charger": {"x": 7, "y": 5},
    },
}


def _state_from(spec: dict) -> dict:
    return {
        "entities": {
            "robot": [dict(id="robot_1", **spec["robot"])],
            "tote": [dict(id=f"tote_{i + 1}", held=False, **t)
                     for i, t in enumerate(spec["totes"])],
            "dock": [dict(id="dock_1", **spec["dock"])],
            "charger": [dict(id="charger_1", **spec["charger"])],
        },
        "outbox": [],
        "payments": [],
        "map": {"width": len(FLOOR_ROWS[0]), "height": len(FLOOR_ROWS),
                "rows": list(FLOOR_ROWS)},
        "quest": spec["quest"],
        "job": list(spec["job"]),
        "vision": spec["vision"],
        "max_turns": spec["max_turns"],
        "turn": 0,
        "status": "playing",
        "log": [],
        "turn_budget": 3,
        "actions_this_turn": 0,
    }


def new_state(scenario: str = "cross_dock") -> dict:
    return _state_from(SCENARIOS[scenario])


def sample_state(rng: random.Random, world=None) -> dict:
    """A random solvable job. The battery is set from the route the job
    actually needs, so roughly half the draws force a detour to the pad and
    the rest do not - a corpus of nothing but detours would teach "always
    charge first"."""
    spots = rng.sample(OPEN_BAYS, 5)
    start, tote_at, decoy_at, dock_at, pad_at = spots
    labels = rng.sample(LABELS, 2)
    spec = {
        "quest": f"Bring tote {labels[0]} out to the outbound dock. The pad "
                 f"is the only place to charge.",
        "vision": rng.choice([2, 3, 3, 4]),
        "max_turns": 30,
        "robot": {"x": start[0], "y": start[1], "battery": 0,
                  "max_battery": 0},
        "totes": [{"label": labels[0], "x": tote_at[0], "y": tote_at[1]},
                  {"label": labels[1], "x": decoy_at[0], "y": decoy_at[1]}],
        "job": ["tote_1"],
        "dock": {"x": dock_at[0], "y": dock_at[1]},
        "charger": {"x": pad_at[0], "y": pad_at[1]},
    }
    state = _state_from(spec)
    direct = _route_len(state, start, tote_at) + _route_len(state, tote_at,
                                                            dock_at)
    via_pad = (_route_len(state, pad_at, tote_at)
               + _route_len(state, tote_at, dock_at))
    to_pad = _route_len(state, start, pad_at)
    r = state["entities"]["robot"][0]
    # a full pack has to cover the longest leg the job can need, or the draw
    # is unwinnable however well it is played
    r["max_battery"] = max(direct, via_pad) + rng.randint(3, 8)
    if rng.random() < 0.5:
        r["battery"] = direct + rng.randint(1, 4)          # no detour needed
    else:
        # short of the direct route, but never short of the pad: the detour
        # is the lesson, being stranded on turn one is not
        r["battery"] = rng.randint(to_pad, max(to_pad, direct - 1))
    return state


def _route_len(state: dict, a, b) -> Optional[int]:
    from harness.oracles import warehouse as oracle
    route = oracle.route(state, a, [b])
    return None if route is None else len(route)


# -------------------------------------------------------------- perception

def _robot(state: dict) -> dict:
    return state["entities"]["robot"][0]


def tile(state: dict, x: int, y: int) -> str:
    m = state["map"]
    if x < 0 or y < 0 or x >= m["width"] or y >= m["height"]:
        return RACK
    row = m["rows"][y]
    return row[x] if x < len(row) else RACK


def _open_dirs(state: dict) -> List[str]:
    r = _robot(state)
    steps = {"north": (0, -1), "south": (0, 1),
             "east": (1, 0), "west": (-1, 0)}
    return [d for d, (dx, dy) in steps.items()
            if tile(state, r["x"] + dx, r["y"] + dy) != RACK]


def observe(state: dict, vision: Optional[int] = None) -> Observation:
    """The turn, in words. The dock and the pad are fixed plant the robot has
    on its floor plan, so they are always named; totes are only what the
    camera can see, which is the `relevant_cards` rule the kanban demo taught
    us - a constant table with everything in it makes the model pick wrong."""
    r = _robot(state)
    vision = state.get("vision", 3) if vision is None else vision
    here = (r["x"], r["y"])
    dock = state["entities"]["dock"][0]
    pad = state["entities"]["charger"][0]
    carrying = next((t for t in state["entities"]["tote"] if t["held"]), None)
    seen = [t for t in state["entities"]["tote"]
            if not t["held"] and abs(t["x"] - r["x"]) <= vision
            and abs(t["y"] - r["y"]) <= vision]

    constants: List[dict] = [
        {"type": "STR", "value": d, "desc": DIR_DESC[d]} for d in DIRECTIONS
    ]
    for t in sorted(seen, key=lambda t: t["id"]):
        dx, dy = t["x"] - r["x"], t["y"] - r["y"]
        where = ("on your bay" if (dx, dy) == (0, 0)
                 else offset_words(dx, dy, unit="bays"))
        constants.append({"type": "ID:tote", "value": t["id"],
                          "desc": f"tote {t['label']} {where}"})
    if carrying:
        constants.append({"type": "ID:tote", "value": carrying["id"],
                          "desc": f"tote {carrying['label']} on your deck"})

    job_labels = ", ".join(
        t["label"] for t in state["entities"]["tote"] if t["id"] in state["job"])
    sight = "; ".join(
        f"tote {t['label']} "
        f"{'on your bay' if (t['x'], t['y']) == here else offset_words(t['x'] - r['x'], t['y'] - r['y'], unit='bays')}"
        for t in sorted(seen, key=lambda t: t["id"])) or "no totes"
    request = (
        f"Warehouse floor, turn {state.get('turn', 0)}. {state.get('quest', '')}\n"
        f"You are at aisle {r['x']}, bay {r['y']} with {r['battery']} of "
        f"{r['max_battery']} battery, carrying "
        f"{'tote ' + carrying['label'] if carrying else 'nothing'}. "
        f"The job is tote {job_labels}.\n"
        f"You can drive {_and_list(_open_dirs(state))}; racks block the rest.\n"
        f"In camera range ({vision} bays): {sight}.\n"
        f"On the floor plan: the outbound dock is "
        f"{_place(dock, here)}, the charging pad is {_place(pad, here)}.\n"
        f"Last turn: {last_turn(state)}.\n"
        f"Choose up to {state.get('turn_budget', 3)} actions for this turn."
    )
    return Observation(request=request, constants=constants,
                       extra={"vision": vision})


def _place(rec: dict, here) -> str:
    dx, dy = rec["x"] - here[0], rec["y"] - here[1]
    if (dx, dy) == (0, 0):
        return "the bay you are on"
    return offset_words(dx, dy, unit="bays")


def _and_list(words: List[str]) -> str:
    if not words:
        return "nowhere"
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def legal_actions(state: dict) -> List[str]:
    """One-action authoring programs the engine would accept from here. The
    off-path restarts draw from this: a wrong-but-legal move is what puts the
    oracle in a state its golden path never visits."""
    obs = observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    r = _robot(state)
    out = []
    for d in _open_dirs(state):
        if r["battery"] > 0:
            out.append(f"CALL @drive ${index[d]}\nSTOP\n")
    carrying = next((t for t in state["entities"]["tote"] if t["held"]), None)
    if carrying:
        out.append(f"CALL @set_down ${index[carrying['id']]}\nSTOP\n")
    else:
        for t in state["entities"]["tote"]:
            if not t["held"] and (t["x"], t["y"]) == (r["x"], r["y"]) \
                    and t["id"] in index:
                out.append(f"CALL @lift ${index[t['id']]}\nSTOP\n")
    pad = state["entities"]["charger"][0]
    if (pad["x"], pad["y"]) == (r["x"], r["y"]) \
            and r["battery"] < r["max_battery"]:
        out.append("CALL @charge\nSTOP\n")
    return out


def outcome(state: dict) -> dict:
    r = _robot(state)
    dock = state["entities"]["dock"][0]
    job = [t for t in state["entities"]["tote"] if t["id"] in state["job"]]
    return {
        "status": state.get("status", "playing"),
        "won": state.get("status") == "done",
        "dead": state.get("status") == "stranded",
        "turn": state.get("turn", 0),
        "battery": r["battery"],
        "funnel": {
            "tote_lifted": any(t["held"] or (t["x"], t["y"]) == (dock["x"], dock["y"])
                               for t in job),
            "on_dock": all((t["x"], t["y"]) == (dock["x"], dock["y"])
                           and not t["held"] for t in job),
        },
    }


WORLD["default_state"] = new_state()
