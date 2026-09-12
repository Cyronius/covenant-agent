"""Text-adventure house: family A's second held-out instance.

Reserved in `data/holdout/reserved.json` alongside the dungeon, and for the
same reason from the other side. The dungeon is a glyph grid; this is a room
graph in prose with no coordinates anywhere. A family score taken on both is
the one that separates "learned to decide" from "learned to read a grid".

Rules live in `runtime/engines/house.js`.
"""
from __future__ import annotations

import random
from typing import List, Optional

from runtime.worlds.decision import Observation, last_turn

NOW = 1_760_000_000

WORLD = {
    "name": "house",
    "now": NOW,
    "entities": {
        "explorer": {"id": "ID:explorer", "room": "ID:room"},
        "room": {"id": "ID:room", "name": "STR", "dark": "BOOL"},
        "way": {"id": "ID:way", "room": "ID:room", "to": "ID:room",
                "heading": "STR", "shut": "BOOL", "locked": "BOOL"},
        "thing": {"id": "ID:thing", "name": "STR", "kind": "STR",
                  "room": "ID:room", "held": "BOOL", "lit": "BOOL",
                  "opens": "STR"},
    },
    "enums": {("thing", "kind"): ["key", "lamp", "junk"]},
    "tools": [
        {
            "name": "go",
            "desc": "Walk out of this room through one of its ways out. "
                    "Fails if that way is shut or locked.",
            "params": [{"name": "way", "type": "ID:way",
                        "desc": "the way out to take",
                        "field": ["way", "id"]}],
            "returns": "OBJ:room",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "house", "fn": "go"},
        },
        {
            "name": "take",
            "desc": "Pick up something lying in the room you are in.",
            "params": [{"name": "thing", "type": "ID:thing",
                        "desc": "the thing to pick up",
                        "field": ["thing", "id"]}],
            "returns": "OBJ:thing",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "house", "fn": "take"},
        },
        {
            "name": "use",
            "desc": "Use something you are carrying. A lamp lights or goes "
                    "out. A key is not used this way - open its way out with "
                    "it instead.",
            "params": [{"name": "thing", "type": "ID:thing",
                        "desc": "the carried thing to use",
                        "field": ["thing", "id"]}],
            "returns": "OBJ:thing",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "house", "fn": "use"},
        },
        {
            "name": "open",
            "desc": "Open a shut way out of this room. A locked one opens "
                    "only if you are carrying the key that fits it.",
            "params": [{"name": "way", "type": "ID:way",
                        "desc": "the way out to open",
                        "field": ["way", "id"]}],
            "returns": "OBJ:way",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "house", "fn": "open"},
        },
    ],
    "post_hook": {"module": "house", "fn": "end_turn"},
}

# room_key -> (display name, dark?)
ROOMS = {
    "hall": ("front hall", False),
    "kitchen": ("kitchen", False),
    "study": ("study", False),
    "pantry": ("pantry", False),
    "landing": ("upstairs landing", False),
    "attic": ("attic", True),
    "cellar": ("cellar", True),
}

# (from, to, heading, shut, locked)
WAYS = [
    ("hall", "kitchen", "north", False, False),
    ("kitchen", "hall", "south", False, False),
    ("hall", "study", "east", False, False),
    ("study", "hall", "west", False, False),
    ("kitchen", "pantry", "east", True, False),
    ("pantry", "kitchen", "west", False, False),
    ("hall", "landing", "up the stairs", False, False),
    ("landing", "hall", "down the stairs", False, False),
    ("landing", "attic", "up the ladder", True, False),
    ("attic", "landing", "down the ladder", False, False),
    ("pantry", "cellar", "down the steps", True, True),
    ("cellar", "pantry", "up the steps", False, False),
]

SCENARIOS = {
    "cellar_run": {
        "goal": "cellar",
        "quest": "Get down into the cellar.",
        "start": "hall",
        "max_turns": 25,
        "things": [
            {"name": "brass key", "kind": "key", "room": "study",
             "opens": "way_11"},
            {"name": "oil lamp", "kind": "lamp", "room": "kitchen"},
            {"name": "stack of post", "kind": "junk", "room": "hall"},
        ],
    },
    "attic_run": {
        "goal": "attic",
        "quest": "Get up into the attic.",
        "start": "pantry",
        "max_turns": 25,
        "things": [
            {"name": "oil lamp", "kind": "lamp", "room": "study"},
            {"name": "brass key", "kind": "key", "room": "hall",
             "opens": "way_11"},
            {"name": "cracked mug", "kind": "junk", "room": "kitchen"},
        ],
    },
}


def _state_from(spec: dict) -> dict:
    room_ids = {key: f"room_{i + 1}" for i, key in enumerate(ROOMS)}
    rooms = [{"id": room_ids[key], "name": name, "dark": dark}
             for key, (name, dark) in ROOMS.items()]
    ways = [{"id": f"way_{i + 1}", "room": room_ids[a], "to": room_ids[b],
             "heading": heading, "shut": shut, "locked": locked}
            for i, (a, b, heading, shut, locked) in enumerate(WAYS)]
    things = [{"id": f"thing_{i + 1}", "name": t["name"], "kind": t["kind"],
               "room": room_ids[t["room"]], "held": False, "lit": False,
               "opens": t.get("opens", "")}
              for i, t in enumerate(spec["things"])]
    return {
        "entities": {
            "explorer": [{"id": "explorer_1",
                          "room": room_ids[spec["start"]]}],
            "room": rooms,
            "way": ways,
            "thing": things,
        },
        "outbox": [],
        "payments": [],
        "quest": spec["quest"],
        "goal_room": room_ids[spec["goal"]],
        "max_turns": spec["max_turns"],
        "turn": 0,
        "status": "playing",
        "log": [],
        "turn_budget": 3,
        "actions_this_turn": 0,
    }


def new_state(scenario: str = "cellar_run") -> dict:
    return _state_from(SCENARIOS[scenario])


def sample_state(rng: random.Random, world=None) -> dict:
    """Held out - this exists so the suite can draw extra exam games, not so
    the corpus can train on them."""
    goal = rng.choice(["cellar", "attic"])
    spec = dict(SCENARIOS["cellar_run" if goal == "cellar" else "attic_run"])
    spec["start"] = rng.choice([k for k in ROOMS if k != goal])
    return _state_from(spec)


# -------------------------------------------------------------- perception

def _explorer(state: dict) -> dict:
    return state["entities"]["explorer"][0]


def room_by_id(state: dict, rid: str) -> dict:
    return next(r for r in state["entities"]["room"] if r["id"] == rid)


def ways_from(state: dict, rid: str) -> List[dict]:
    return [w for w in state["entities"]["way"] if w["room"] == rid]


def observe(state: dict, vision: Optional[int] = None) -> Observation:
    p = _explorer(state)
    room = room_by_id(state, p["room"])
    ways = sorted(ways_from(state, p["room"]), key=lambda w: w["id"])
    here = [t for t in state["entities"]["thing"]
            if not t["held"] and t["room"] == p["room"]]
    held = [t for t in state["entities"]["thing"] if t["held"]]

    constants: List[dict] = []
    for w in ways:
        shut = ("locked" if w["locked"] else "shut" if w["shut"] else "open")
        constants.append({"type": "ID:way", "value": w["id"],
                          "desc": f"the {w['heading']} way out of the "
                                  f"{room['name']}, {shut}"})
    for t in sorted(here, key=lambda t: t["id"]):
        constants.append({"type": "ID:thing", "value": t["id"],
                          "desc": f"{t['name']}, lying in the {room['name']}"})
    for t in sorted(held, key=lambda t: t["id"]):
        lit = " (lit)" if t["lit"] else ""
        constants.append({"type": "ID:thing", "value": t["id"],
                          "desc": f"{t['name']} you are carrying{lit}"})

    exits = "; ".join(
        f"{w['heading']} to the {room_by_id(state, w['to'])['name']}"
        f"{' - locked' if w['locked'] else ' - shut' if w['shut'] else ''}"
        for w in ways) or "no way out at all"
    lying = ", ".join(t["name"] for t in here) or "nothing worth taking"
    carrying = ", ".join(t["name"] + (" (lit)" if t["lit"] else "")
                         for t in held) or "nothing"
    goal_name = room_by_id(state, state["goal_room"])["name"]

    request = (
        f"You are in the {room['name']}. Turn {state.get('turn', 0)}. "
        f"{state.get('quest', '')}\n"
        f"Ways out: {exits}.\n"
        f"Lying here: {lying}. You are carrying: {carrying}.\n"
        f"Some rooms are pitch dark and you cannot enter one without a lit "
        f"lamp in hand; the {goal_name} is one of them.\n"
        f"Last turn: {last_turn(state)}.\n"
        f"Choose up to {state.get('turn_budget', 3)} actions for this turn."
    )
    return Observation(request=request, constants=constants)


def legal_actions(state: dict) -> List[str]:
    obs = observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    p = _explorer(state)
    out = []
    for w in ways_from(state, p["room"]):
        if not w["shut"]:
            out.append(f"CALL @go ${index[w['id']]}\nSTOP\n")
        else:
            out.append(f"CALL @open ${index[w['id']]}\nSTOP\n")
    for t in state["entities"]["thing"]:
        if not t["held"] and t["room"] == p["room"]:
            out.append(f"CALL @take ${index[t['id']]}\nSTOP\n")
        elif t["held"] and t["kind"] == "lamp":
            out.append(f"CALL @use ${index[t['id']]}\nSTOP\n")
    return out


def outcome(state: dict) -> dict:
    p = _explorer(state)
    things = state["entities"]["thing"]
    goal_ways = [w for w in state["entities"]["way"]
                 if w["to"] == state["goal_room"]]
    return {
        "status": state.get("status", "playing"),
        "won": state.get("status") == "done",
        "dead": False,
        "turn": state.get("turn", 0),
        "funnel": {
            "lamp_lit": any(t["kind"] == "lamp" and t["lit"] for t in things),
            "way_open": any(not w["shut"] for w in goal_ways),
            "arrived": p["room"] == state["goal_room"],
        },
    }


WORLD["default_state"] = new_state()
