"""Elevator dispatcher: family A's instance where waiting is a legal move.

Two things vary from the dungeon on purpose. The state is a table of floors
and calls, not a picture; and `hold` is a real action that is sometimes the
right one, because a rider announced two ticks out on the floor you are
already standing on is cheaper to wait for than to drive away from and come
back to. No tuned checkpoint has ever emitted a do-nothing turn.

Rules live in `runtime/engines/elevator.js`.
"""
from __future__ import annotations

import random
from typing import List, Optional

from runtime.worlds.decision import Observation, last_turn

NOW = 1_760_000_000

WORLD = {
    "name": "elevator",
    "now": NOW,
    "entities": {
        "car": {"id": "ID:car", "floor": "INT", "doors_open": "BOOL"},
        "rider": {"id": "ID:rider", "name": "STR", "origin": "INT",
                  "dest": "INT", "appears": "INT", "aboard": "BOOL",
                  "delivered": "BOOL"},
    },
    "tools": [
        {
            "name": "go_to",
            "desc": "Send the car to a floor. It takes one tick per floor "
                    "travelled. Fails if the car is already there.",
            "params": [{"name": "floor", "type": "INT",
                        "desc": "the floor to travel to",
                        "field": ["car", "floor"]}],
            "returns": "OBJ:car",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "elevator", "fn": "go_to"},
        },
        {
            "name": "open_doors",
            "desc": "Open the doors where the car is: everyone waiting on "
                    "this floor gets on, everyone whose stop this is gets "
                    "off, then the doors shut. Costs a tick even if nobody "
                    "is there.",
            "params": [],
            "returns": "OBJ:car",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "elevator", "fn": "open_doors"},
        },
        {
            "name": "hold",
            "desc": "Keep the car where it is for one tick and do nothing "
                    "else. The right move when someone is about to press the "
                    "button on this floor.",
            "params": [],
            "returns": "OBJ:car",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "elevator", "fn": "hold"},
        },
    ],
    "post_hook": {"module": "elevator", "fn": "end_turn"},
}

NAMES = ["Otto", "Priya", "Wren", "Kai", "Lena", "Femi", "Rosa", "Dev",
         "Hana", "Ivan", "June", "Uma"]

SCENARIOS = {
    "morning_rush": {
        "floors": 6,
        "car": 1,
        "max_turns": 25,
        "riders": [
            {"name": "Otto", "origin": 1, "dest": 5, "appears": 0},
            {"name": "Priya", "origin": 4, "dest": 1, "appears": 0},
            {"name": "Wren", "origin": 1, "dest": 3, "appears": 3},
        ],
    },
    "late_call": {
        "floors": 7,
        "car": 4,
        "max_turns": 25,
        "riders": [
            {"name": "Kai", "origin": 4, "dest": 7, "appears": 2},
            {"name": "Lena", "origin": 6, "dest": 2, "appears": 0},
            {"name": "Rosa", "origin": 2, "dest": 5, "appears": 6},
        ],
    },
}


def _state_from(spec: dict) -> dict:
    return {
        "entities": {
            "car": [{"id": "car_1", "floor": spec["car"], "doors_open": False}],
            "rider": [dict(id=f"rider_{i + 1}", aboard=False, delivered=False,
                           **r)
                      for i, r in enumerate(spec["riders"])],
        },
        "outbox": [],
        "payments": [],
        "floors": spec["floors"],
        "tick": 0,
        "capacity": spec.get("capacity", 4),
        "patience": spec.get("patience", 25),
        "max_turns": spec["max_turns"],
        "turn": 0,
        "status": "playing",
        "log": [],
        "turn_budget": 2,
        "actions_this_turn": 0,
    }


def new_state(scenario: str = "morning_rush") -> dict:
    return _state_from(SCENARIOS[scenario])


def sample_state(rng: random.Random, world=None) -> dict:
    """A random building. At least one rider is always announced ahead of
    time on a floor the car will be standing on, so `hold` turns up in the
    corpus at a rate the model can learn rather than as a curiosity."""
    floors = rng.randint(5, 8)
    car = rng.randint(1, floors)
    names = rng.sample(NAMES, rng.randint(2, 4))
    riders = []
    for i, name in enumerate(names):
        origin = rng.randint(1, floors)
        dest = rng.choice([f for f in range(1, floors + 1) if f != origin])
        appears = 0 if i == 0 else rng.choice([0, 0, 2, 3, 5, 8])
        riders.append({"name": name, "origin": origin, "dest": dest,
                       "appears": appears})
    if rng.random() < 0.5:
        # the hold case: someone turns up shortly on the car's own floor
        riders[-1]["origin"] = car
        riders[-1]["appears"] = rng.randint(2, 4)
        riders[-1]["dest"] = rng.choice(
            [f for f in range(1, floors + 1) if f != car])
    return _state_from({"floors": floors, "car": car, "max_turns": 30,
                        "riders": riders})


# -------------------------------------------------------------- perception

def _car(state: dict) -> dict:
    return state["entities"]["car"][0]


def waiting(state: dict) -> List[dict]:
    tick = state.get("tick", 0)
    return [r for r in state["entities"]["rider"]
            if r["appears"] <= tick and not r["aboard"] and not r["delivered"]]


def upcoming(state: dict) -> List[dict]:
    tick = state.get("tick", 0)
    return [r for r in state["entities"]["rider"]
            if r["appears"] > tick and not r["delivered"]]


def aboard(state: dict) -> List[dict]:
    return [r for r in state["entities"]["rider"] if r["aboard"]]


def observe(state: dict, vision: Optional[int] = None) -> Observation:
    c = _car(state)
    tick = state.get("tick", 0)
    constants: List[dict] = [
        {"type": "INT", "value": f, "desc": _floor_desc(f, c["floor"])}
        for f in range(1, state["floors"] + 1)
    ]

    riding = ", ".join(f"{r['name']} for floor {r['dest']}"
                       for r in aboard(state)) or "nobody"
    calls = "; ".join(
        f"{r['name']} on floor {r['origin']} going "
        f"{'up' if r['dest'] > r['origin'] else 'down'} to floor {r['dest']}, "
        f"waiting {tick - r['appears']} ticks"
        for r in sorted(waiting(state), key=lambda r: r["id"])) or "nobody"
    soon = "; ".join(
        f"{r['name']} on floor {r['origin']} in {r['appears'] - tick} ticks, "
        f"for floor {r['dest']}"
        for r in sorted(upcoming(state), key=lambda r: r["id"])) or "nobody"

    request = (
        f"Elevator bank, turn {state.get('turn', 0)} (tick {tick}). "
        f"Get everyone in the building where they are going.\n"
        f"The car is on floor {c['floor']} of {state['floors']}, doors shut, "
        f"carrying {riding}.\n"
        f"Waiting now: {calls}.\n"
        f"Due to press the button: {soon}.\n"
        f"A rider who has waited more than {state['patience']} ticks gives up "
        f"and the job is blown.\n"
        f"Last turn: {last_turn(state)}.\n"
        f"Choose up to {state.get('turn_budget', 2)} actions for this turn."
    )
    return Observation(request=request, constants=constants)


def _floor_desc(floor: int, car_floor: int) -> str:
    if floor == car_floor:
        return f"floor {floor} (the car is on it)"
    gap = abs(floor - car_floor)
    way = "up" if floor > car_floor else "down"
    return f"floor {floor} ({gap} {way})"


def legal_actions(state: dict) -> List[str]:
    obs = observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    c = _car(state)
    out = ["CALL @open_doors\nSTOP\n", "CALL @hold\nSTOP\n"]
    for f in range(1, state["floors"] + 1):
        if f != c["floor"]:
            out.append(f"CALL @go_to ${index[f]}\nSTOP\n")
    return out


def outcome(state: dict) -> dict:
    riders = state["entities"]["rider"]
    return {
        "status": state.get("status", "playing"),
        "won": state.get("status") == "done",
        "dead": state.get("status") == "walked_out",
        "turn": state.get("turn", 0),
        "tick": state.get("tick", 0),
        "funnel": {
            "anyone_boarded": any(r["aboard"] or r["delivered"]
                                  for r in riders),
            "half_delivered": sum(r["delivered"] for r in riders) * 2
            >= len(riders),
            "all_delivered": all(r["delivered"] for r in riders),
        },
    }


WORLD["default_state"] = new_state()
