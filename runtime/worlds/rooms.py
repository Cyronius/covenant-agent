"""Grid rooms: family A's BabyAI-style instance, rendered in words.

A few rooms joined by coloured doors, some locked, with coloured balls, boxes
and keys lying about, and a mission in BabyAI's own grammar ("open the blue
door, then pick up the grey key"). The mission sentences, the colours and the
rules (a locked door wants the key of its colour; one object in hand at a
time) come from BabyAI and Minigrid: `data/borrowed/babyai/README.md`.

Two things are deliberate. Nothing here is drawn: the held-out dungeon is a
glyph grid, so this world says in words what is one step away in each
direction (the direction constants, exactly as `rpg._exit_desc` does) and
where each thing in view is. And every tool shares its argument type with a
sibling (move and drop take a direction and return the agent;
pick_up, open and put_next_to take an object and return it), so no tool can
be picked by its signature alone.

One module serves two world names, as `pages` does: `rooms` (go to, pick up,
open, put next to, alone or joined by ", then" / " and") is for training, and
`rooms_after` ("X after you Y", where the sentence names the steps in the
reverse of the order they have to happen) is the held-out mission type. The
exam holds out a way of phrasing the order, not a tool: holding out
put-next-to would have left `put_next_to` a tool no training row ever
calls, and the exam a test of a tool the model has never seen used.

Rules live in `runtime/engines/rooms.js`. This module owns the schema, the
sampler and perception, and the Python mirror of the rules that the oracle
and `legal_actions` plan with; it never mutates a game.
"""
from __future__ import annotations

import copy
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from runtime.worlds.decision import Observation, last_turn, offset_words

NOW = 1_760_000_000
ROOT = Path(__file__).resolve().parent.parent.parent
GRAMMAR = json.loads((ROOT / "data" / "borrowed" / "babyai" / "grammar.json")
                     .read_text(encoding="utf-8"))
COLORS: List[str] = GRAMMAR["colors"]
PICKABLE: List[str] = GRAMMAR["pickable_types"]

WALL = "#"
DIRECTIONS = ["north", "south", "east", "west"]
STEP = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

TRAIN_WORLD = "rooms"
EXAM_WORLD = "rooms_after"

TOOLS = [
    {
        "name": "move",
        "desc": "Walk one tile in a direction. Fails if a wall, a door that "
                "is not open, or an object is in the way.",
        "params": [{"name": "direction", "type": "STR",
                    "desc": "which way to walk: north, south, east or west",
                    "field": ["agent", "facing"]}],
        "returns": "OBJ:agent",
        "effects": ["mutates"],
        "impl": {"op": "engine", "module": "rooms", "fn": "move"},
    },
    {
        "name": "drop",
        "desc": "Set the object you are carrying down on the empty floor tile "
                "one step away in a direction, leaving your hands free.",
        "params": [{"name": "direction", "type": "STR",
                    "desc": "which side to set it down: north, south, east "
                            "or west",
                    "field": ["agent", "facing"]}],
        "returns": "OBJ:agent",
        "effects": ["mutates"],
        "impl": {"op": "engine", "module": "rooms", "fn": "drop"},
    },
    {
        "name": "pick_up",
        "desc": "Pick up a ball, box or key one step away from you. You can "
                "carry one object at a time.",
        "params": [{"name": "target", "type": "ID:object",
                    "desc": "the object to pick up",
                    "field": ["object", "id"]}],
        "returns": "OBJ:object",
        "effects": ["mutates"],
        "impl": {"op": "engine", "module": "rooms", "fn": "pick_up"},
    },
    {
        "name": "open",
        "desc": "Open a door one step away from you. A locked door opens only "
                "while you carry the key of its colour.",
        "params": [{"name": "target", "type": "ID:object",
                    "desc": "the door to open",
                    "field": ["object", "id"]}],
        "returns": "OBJ:object",
        "effects": ["mutates"],
        "impl": {"op": "engine", "module": "rooms", "fn": "open"},
    },
    {
        "name": "put_next_to",
        "desc": "Set the object you are carrying down beside another object "
                "one step away from you, on a free tile touching both of you.",
        "params": [{"name": "target", "type": "ID:object",
                    "desc": "the object to put it beside",
                    "field": ["object", "id"]}],
        "returns": "OBJ:object",
        "effects": ["mutates"],
        "impl": {"op": "engine", "module": "rooms", "fn": "put_next_to"},
    },
]

ENTITIES = {
    # facing: the way the agent last moved or set something down. It is
    # here so move and drop share one argument slot and one signature.
    "agent": {"id": "ID:agent", "x": "INT", "y": "INT", "facing": "STR"},
    "object": {"id": "ID:object", "kind": "STR", "color": "STR",
               "x": "INT", "y": "INT", "held": "BOOL", "locked": "BOOL",
               "open": "BOOL"},
}


def _world(name: str) -> dict:
    return {
        "name": name,
        "now": NOW,
        "entities": copy.deepcopy(ENTITIES),
        "tools": copy.deepcopy(TOOLS),
        "post_hook": {"module": "rooms", "fn": "end_turn"},
    }


# ------------------------------------------------------------------ layouts

ROOM_NAMES = {
    (1, 2): [["west room", "east room"]],
    (1, 3): [["west room", "middle room", "east room"]],
    (2, 2): [["north-west room", "north-east room"],
             ["south-west room", "south-east room"]],
}


def _rooms(rows: int, cols: int, size: int) -> List[dict]:
    out = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * (size + 1) + 1, r * (size + 1) + 1
            out.append({"id": f"room_{len(out) + 1}",
                        "name": ROOM_NAMES[(rows, cols)][r][c],
                        "r": r, "c": c, "x0": x0, "y0": y0,
                        "x1": x0 + size - 1, "y1": y0 + size - 1})
    return out


def _map_rows(rooms: List[dict], width: int, height: int,
              door_cells: List[Tuple[int, int]]) -> List[str]:
    grid = [[WALL] * width for _ in range(height)]
    for rm in rooms:
        for y in range(rm["y0"], rm["y1"] + 1):
            for x in range(rm["x0"], rm["x1"] + 1):
                grid[y][x] = "."
    for x, y in door_cells:
        grid[y][x] = "."
    return ["".join(row) for row in grid]


def _mission_state(spec: dict) -> dict:
    """A state from a fully specified layout (the fixed scenarios and the
    sampler both come through here)."""
    rows, cols, size = spec["shape"]
    rooms = _rooms(rows, cols, size)
    width, height = cols * (size + 1) + 1, rows * (size + 1) + 1
    counters: Dict[str, int] = {}
    objs = []
    for o in spec["objects"]:
        counters[o["kind"]] = counters.get(o["kind"], 0) + 1
        objs.append({"id": f"{o['kind']}_{counters[o['kind']]}",
                        "kind": o["kind"], "color": o["color"],
                        "x": o["x"], "y": o["y"], "held": False,
                        "locked": bool(o.get("locked", False)),
                        "open": bool(o.get("open", False))})
    doors = [(o["x"], o["y"]) for o in objs if o["kind"] == "door"]
    state = {
        "entities": {
            "agent": [{"id": "agent_1", "x": spec["agent"][0],
                       "y": spec["agent"][1], "facing": "north"}],
            "object": objs,
        },
        "outbox": [],
        "payments": [],
        "map": {"width": width, "height": height,
                "rows": _map_rows(rooms, width, height, doors)},
        "rooms": [{k: rm[k] for k in ("id", "name", "x0", "y0", "x1", "y1")}
                  for rm in rooms],
        "visited": [],
        "mission": copy.deepcopy(spec["mission"]),
        "max_turns": spec.get("max_turns", 40),
        "turn": 0,
        "status": "playing",
        "log": [],
        "turn_budget": 3,
        "actions_this_turn": 0,
    }
    m = state["mission"]
    m["text"] = mission_text(objects(state), m["steps"], m["connective"])
    m["done"] = [False] * len(m["steps"])
    a = agent(state)
    state["visited"] = rooms_of(state, a["x"], a["y"])
    return state


# ----------------------------------------------------------------- missions

def _desc_text(state_objs: List[dict], desc: dict) -> str:
    """BabyAI's ObjDesc.surface: '[colour] type', 'the' when one object in
    the level matches, else 'a' (data/borrowed/babyai/grammar.json)."""
    words = (f"{desc['color']} " if desc.get("color") else "") + desc["kind"]
    n = sum(1 for o in state_objs if matches(o, desc))
    return f"{GRAMMAR['article']['one_match' if n == 1 else 'several_match']}" \
           f" {words}"


def step_text(objs: List[dict], step: dict) -> str:
    tmpl = GRAMMAR["actions"][step["verb"]]
    return tmpl.format(obj=_desc_text(objs, step["obj"]),
                       fixed=_desc_text(objs, step["fixed"])
                       if step.get("fixed") else "")


def mission_text(objs: List[dict], steps: List[dict], connective: str) -> str:
    """`steps` are in the order they must be done; "after" says them the
    other way round, as BabyAI's AfterInstr does."""
    if connective == "single":
        return step_text(objs, steps[0])
    if connective == "after":
        a, b = steps[1], steps[0]
    else:
        a, b = steps[0], steps[1]
    return GRAMMAR["connectives"][connective].format(
        a=step_text(objs, a), b=step_text(objs, b))


def matches(obj: dict, desc: dict) -> bool:
    return obj["kind"] == desc["kind"] and (
        not desc.get("color") or obj["color"] == desc["color"])


# --------------------------------------------------------------- scenarios

SCENARIOS = {
    "key_then_door": {
        "world": TRAIN_WORLD,
        "shape": (1, 2, 4),
        "agent": (2, 3),
        "objects": [
            {"kind": "door", "color": "blue", "x": 5, "y": 2, "locked": True},
            {"kind": "key", "color": "blue", "x": 3, "y": 1},
            {"kind": "ball", "color": "red", "x": 8, "y": 3},
            {"kind": "box", "color": "grey", "x": 1, "y": 4},
        ],
        "mission": {"connective": "before",
                    "steps": [{"verb": "open",
                               "obj": {"kind": "door", "color": "blue"}},
                              {"verb": "goto",
                               "obj": {"kind": "ball", "color": "red"}}]},
        "max_turns": 30,
    },
    "fetch_the_key": {
        "world": TRAIN_WORLD,
        "shape": (1, 3, 3),
        "agent": (6, 2),
        "objects": [
            {"kind": "door", "color": "green", "x": 4, "y": 1},
            {"kind": "door", "color": "yellow", "x": 8, "y": 3},
            {"kind": "key", "color": "grey", "x": 10, "y": 2},
            {"kind": "ball", "color": "purple", "x": 1, "y": 1},
            {"kind": "box", "color": "red", "x": 5, "y": 3},
        ],
        "mission": {"connective": "single",
                    "steps": [{"verb": "pickup",
                               "obj": {"kind": "key", "color": "grey"}}]},
        "max_turns": 30,
    },
    "ball_by_the_box": {
        "world": TRAIN_WORLD,
        "shape": (2, 2, 3),
        "agent": (1, 1),
        "objects": [
            {"kind": "door", "color": "red", "x": 4, "y": 2},
            {"kind": "door", "color": "yellow", "x": 6, "y": 4},
            {"kind": "door", "color": "blue", "x": 2, "y": 4},
            {"kind": "ball", "color": "green", "x": 3, "y": 3},
            {"kind": "box", "color": "purple", "x": 7, "y": 7},
            {"kind": "key", "color": "grey", "x": 1, "y": 7},
        ],
        "mission": {"connective": "single",
                    "steps": [{"verb": "putnext",
                               "obj": {"kind": "ball", "color": "green"},
                               "fixed": {"kind": "box", "color": "purple"}}]},
        "max_turns": 30,
    },
    "ball_after_the_key": {
        "world": EXAM_WORLD,
        "shape": (1, 3, 3),
        "agent": (6, 2),
        "objects": [
            {"kind": "door", "color": "green", "x": 4, "y": 1},
            {"kind": "door", "color": "yellow", "x": 8, "y": 3},
            {"kind": "key", "color": "grey", "x": 10, "y": 2},
            {"kind": "ball", "color": "purple", "x": 1, "y": 1},
            {"kind": "box", "color": "red", "x": 5, "y": 3},
        ],
        "mission": {"connective": "after",
                    "steps": [{"verb": "goto",
                               "obj": {"kind": "ball", "color": "purple"}},
                              {"verb": "pickup",
                               "obj": {"kind": "key", "color": "grey"}}]},
        "max_turns": 30,
    },
}


def scenarios_for(world_name: str) -> List[str]:
    return sorted(k for k, s in SCENARIOS.items() if s["world"] == world_name)


def new_state(scenario: str = "key_then_door") -> dict:
    return _mission_state(SCENARIOS[scenario])


def sample_state(rng: random.Random, world: Optional[str] = None) -> dict:
    """A random layout and mission that the oracle can finish. `world`
    picks the mission type: `rooms` never draws an "after you" mission and
    `rooms_after` draws nothing else."""
    world = world or TRAIN_WORLD
    for _ in range(500):
        spec = _sample_layout(rng)
        if spec is None:
            continue
        state = _mission_state(dict(spec, mission={
            "connective": "single",
            "steps": [{"verb": "goto", "obj": {"kind": "door"}}]}))
        mission = _sample_mission(rng, state, exam=world == EXAM_WORLD)
        if mission is None:
            continue
        state["mission"] = mission
        state["mission"]["done"] = [False] * len(mission["steps"])
        return state
    raise RuntimeError("could not sample a rooms layout")


def _sample_layout(rng: random.Random) -> Optional[dict]:
    rows, cols = rng.choice([(1, 2), (1, 2), (1, 3), (2, 2)])
    size = 4
    rooms = _rooms(rows, cols, size)
    by_rc = {(rm["r"], rm["c"]): i for i, rm in enumerate(rooms)}
    edges = []
    for (r, c), i in by_rc.items():
        if (r, c + 1) in by_rc:
            edges.append((i, by_rc[(r, c + 1)], "v"))
        if (r + 1, c) in by_rc:
            edges.append((i, by_rc[(r + 1, c)], "h"))
    # a random spanning tree, and in the 2x2 layout sometimes the fourth
    # door as well, so a locked door is not always the only way through
    rng.shuffle(edges)
    parent = list(range(len(rooms)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    chosen, spare = [], []
    for e in edges:
        a, b = find(e[0]), find(e[1])
        if a != b:
            parent[a] = b
            chosen.append(e)
        else:
            spare.append(e)
    if spare and rng.random() < 0.3:
        chosen.append(spare[0])

    colors = rng.sample(COLORS, len(chosen))
    objects = []
    for (a, b, orient), color in zip(chosen, colors):
        ra = rooms[a]
        if orient == "v":
            x, y = ra["x1"] + 1, rng.randint(ra["y0"], ra["y1"])
        else:
            x, y = rng.randint(ra["x0"], ra["x1"]), ra["y1"] + 1
        objects.append({"kind": "door", "color": color, "x": x, "y": y,
                        "joins": (a, b), "locked": False,
                        "open": rng.random() < 0.25})
    locked = None
    if rng.random() < 0.5:
        locked = rng.choice(objects)
        locked.update(locked=True, open=False)

    start_room = rng.randrange(len(rooms))
    # the rooms reachable from the start without the locked door: its key
    # has to lie in one of them, as BabyAI's add_locked_room places it
    reach = {start_room}
    grew = True
    while grew:
        grew = False
        for d in objects:
            if d is locked:
                continue
            a, b = d["joins"]
            if (a in reach) != (b in reach):
                reach |= {a, b}
                grew = True

    door_cells = {(d["x"], d["y"]) for d in objects}
    near_door = {(x + dx, y + dy) for x, y in door_cells
                 for dx, dy in STEP.values()}
    free = {i: [(x, y) for y in range(rm["y0"], rm["y1"] + 1)
                for x in range(rm["x0"], rm["x1"] + 1)
                if (x, y) not in near_door]
            for i, rm in enumerate(rooms)}
    for cells in free.values():
        rng.shuffle(cells)

    def take(room: int) -> Optional[Tuple[int, int]]:
        return free[room].pop() if free[room] else None

    things = []
    if locked is not None:
        cell = take(rng.choice(sorted(reach)))
        if cell is None:
            return None
        things.append({"kind": "key", "color": locked["color"],
                       "x": cell[0], "y": cell[1]})
    for i in range(len(rooms)):
        for _ in range(rng.randint(1, 2)):
            cell = take(i)
            if cell is None:
                break
            things.append({"kind": rng.choice(PICKABLE),
                           "color": rng.choice(COLORS),
                           "x": cell[0], "y": cell[1]})
    start = take(start_room)
    if start is None:
        return None
    for d in objects:
        d.pop("joins")
    spec = {"shape": (rows, cols, size), "agent": start,
            "objects": objects + things, "max_turns": 40}
    probe = _mission_state(dict(spec, mission={
        "connective": "single",
        "steps": [{"verb": "goto", "obj": {"kind": "door"}}]}))
    return spec if _all_reachable(probe) and _keys_first(probe) else None


def _reached(state: dict, through_locked: bool = True) -> set:
    """Floor tiles the agent can walk to, doors counted as passable (locked
    ones only when `through_locked`)."""
    a = agent(state)
    seen = {(a["x"], a["y"])}
    todo = deque(seen)
    while todo:
        x, y = todo.popleft()
        for dx, dy in STEP.values():
            c = (x + dx, y + dy)
            if c in seen or tile(state, *c) == WALL:
                continue
            o = object_at(state, *c)
            if o is not None and (o["kind"] != "door"
                                  or (o["locked"] and not through_locked)):
                continue
            seen.add(c)
            todo.append(c)
    return seen


def _all_reachable(state: dict) -> bool:
    """Every free floor tile connected to the agent with doors counted as
    passable, and every object with a free tile beside it: an object wall
    across a room would make a draw unwinnable."""
    seen = _reached(state)
    floor = {(x, y) for y, row in enumerate(state["map"]["rows"])
             for x, ch in enumerate(row) if ch != WALL}
    blocked = {(o["x"], o["y"]) for o in objects(state)
               if o["kind"] != "door" and not o["held"]}
    if (floor - blocked) - seen:
        return False
    return all(any((o["x"] + dx, o["y"] + dy) in seen
                   for dx, dy in STEP.values())
               for o in objects(state) if o["kind"] != "door"
               and not o["held"])


def _keys_first(state: dict) -> bool:
    """Each locked door's key, and the door itself, within reach without
    going through a locked door - BabyAI's add_locked_room puts the key
    outside the locked room; this also catches objects that wall a key
    off from the only open way to it."""
    seen = _reached(state, through_locked=False)

    def near(o):
        return any((o["x"] + dx, o["y"] + dy) in seen
                   for dx, dy in STEP.values())

    for door in objects(state):
        if door["kind"] != "door" or not door["locked"]:
            continue
        keys = [k for k in objects(state) if k["kind"] == "key"
                and k["color"] == door["color"]]
        if not near(door) or not any(near(k) for k in keys):
            return False
    return True


def _pick_desc(rng: random.Random, objs: List[dict], target: dict) -> dict:
    """A description of `target`: colour and type, or (one time in seven,
    BabyAI's [None, *colors] draw) the type alone."""
    if rng.random() < 1 / 7:
        return {"kind": target["kind"], "color": None}
    return {"kind": target["kind"], "color": target["color"]}


def _sample_step(rng: random.Random, state: dict, verb: str) -> Optional[dict]:
    objs = objects(state)
    a = agent(state)
    if verb == "goto":
        pool = objs
    elif verb == "pickup":
        pool = [o for o in objs if o["kind"] in PICKABLE]
    else:  # open
        pool = [o for o in objs if o["kind"] == "door" and not o["open"]]
    if not pool:
        return None
    target = rng.choice(pool)
    desc = _pick_desc(rng, objs, target)
    hits = [o for o in objs if matches(o, desc)]
    if verb == "goto" and any(abs(o["x"] - a["x"]) + abs(o["y"] - a["y"]) <= 1
                              for o in hits):
        return None
    if verb == "open" and any(o["open"] for o in hits):
        return None
    return {"verb": verb, "obj": desc}


def _sample_mission(rng: random.Random, state: dict,
                    exam: bool = False) -> Optional[dict]:
    """GoTo / Pickup / Open, alone or two of them joined by BabyAI's
    connectives (levelgen.py rand_instr), and in training a put-next-to on
    its own a fifth of the time. The exam draws only " after you", the
    connective training never sees."""
    verbs = ["goto", "pickup", "open"]
    if exam:
        kind = "after"
    elif rng.random() < 0.2:
        return _sample_putnext(rng, state)
    else:
        kind = rng.choices(["single", "before", "and"], weights=[4, 2, 1])[0]
    n = 1 if kind == "single" else 2
    steps = []
    for _ in range(n):
        step = _sample_step(rng, state, rng.choice(verbs))
        if step is None or step in steps:
            return None
        steps.append(step)
    return {"text": mission_text(objects(state), steps, kind),
            "connective": kind, "steps": steps}


def _sample_putnext(rng: random.Random, state: dict) -> Optional[dict]:
    """Put a ball, box or key next to another ball, box or key that it is
    not already beside, with room around the fixed one to do it."""
    objs = objects(state)
    movable = [o for o in objs if o["kind"] in PICKABLE]
    if len(movable) < 2:
        return None
    mover, fixed = rng.sample(movable, 2)
    dm, df = _pick_desc(rng, objs, mover), _pick_desc(rng, objs, fixed)
    if any(matches(o, dm) and matches(o, df) for o in objs):
        return None                     # "put a key next to the green key"
    for m in (o for o in objs if matches(o, dm)):
        for f in (o for o in objs if matches(o, df)):
            if m is not f and abs(m["x"] - f["x"]) + abs(m["y"] - f["y"]) == 1:
                return None
    # somewhere to stand beside the fixed object with a free tile beside
    # both, once the mover has been lifted out of the way
    trial = copy.deepcopy(state)
    for o in objects(trial):
        if o["id"] == mover["id"]:
            o["held"] = True
    if not any(put_spots(trial, f) for f in objects(trial)
               if matches(f, df) and not f["held"]):
        return None
    steps = [{"verb": "putnext", "obj": dm, "fixed": df}]
    return {"text": mission_text(objs, steps, "single"),
            "connective": "single", "steps": steps}


# -------------------------------------------------------- the rules, mirrored
# runtime/engines/rooms.js is the authority; these are what the oracle and
# legal_actions plan with, and tests/test_rooms.py holds them to the engine.

def agent(state: dict) -> dict:
    return state["entities"]["agent"][0]


def objects(state: dict) -> List[dict]:
    return state["entities"]["object"]


def carried(state: dict) -> Optional[dict]:
    return next((o for o in objects(state) if o["held"]), None)


def tile(state: dict, x: int, y: int) -> str:
    m = state["map"]
    if x < 0 or y < 0 or x >= m["width"] or y >= m["height"]:
        return WALL
    row = m["rows"][y]
    return row[x] if x < len(row) else WALL


def object_at(state: dict, x: int, y: int) -> Optional[dict]:
    return next((o for o in objects(state)
                 if not o["held"] and o["x"] == x and o["y"] == y), None)


def room_by_id(state: dict, rid: str) -> dict:
    return next(r for r in state["rooms"] if r["id"] == rid)


def _inside(rm: dict, x: int, y: int) -> bool:
    return rm["x0"] <= x <= rm["x1"] and rm["y0"] <= y <= rm["y1"]


def rooms_of(state: dict, x: int, y: int) -> List[str]:
    """The room a tile is in; a doorway belongs to both rooms it joins."""
    inside = [r["id"] for r in state["rooms"] if _inside(r, x, y)]
    if inside:
        return inside
    return [r["id"] for r in state["rooms"]
            if any(_inside(r, x + dx, y + dy) for dx, dy in STEP.values())]


def door_rooms(state: dict, door: dict) -> List[str]:
    return rooms_of(state, door["x"], door["y"])


def dist(a: dict, o: dict) -> int:
    return abs(a["x"] - o["x"]) + abs(a["y"] - o["y"])


def walkable(state: dict, x: int, y: int) -> bool:
    if tile(state, x, y) == WALL:
        return False
    o = object_at(state, x, y)
    return o is None or (o["kind"] == "door" and o["open"])


def droppable(state: dict, x: int, y: int) -> bool:
    """Empty floor inside a room: not a wall, not a doorway, nothing on it."""
    return (tile(state, x, y) != WALL and object_at(state, x, y) is None
            and any(_inside(r, x, y) for r in state["rooms"]))


def put_spots(state: dict, target: dict) -> List[Tuple[int, int]]:
    """Tiles the agent can stand on (beside `target`) from which put_next_to
    has a free tile touching both: the two tiles beside the target that are
    at right angles to the agent's side of it."""
    out = []
    for dx, dy in STEP.values():
        ax, ay = target["x"] + dx, target["y"] + dy
        if not walkable(state, ax, ay) and not (
                (ax, ay) == (agent(state)["x"], agent(state)["y"])):
            continue
        if put_cell(state, target, ax, ay) is not None:
            out.append((ax, ay))
    return out


def put_cell(state: dict, target: dict, ax: int, ay: int
             ) -> Optional[Tuple[int, int]]:
    """Where put_next_to sets the object down when the agent stands at
    (ax, ay) beside `target`, in the engine's order; None if nowhere."""
    dx, dy = ax - target["x"], ay - target["y"]
    if abs(dx) + abs(dy) != 1:
        return None
    for d in DIRECTIONS:
        px, py = STEP[d]
        if px * dx + py * dy != 0:
            continue                    # only at right angles to the agent
        c = (target["x"] + px, target["y"] + py)
        if droppable(state, *c):
            return c
    return None


def can_move(state: dict, d: str) -> bool:
    a = agent(state)
    return walkable(state, a["x"] + STEP[d][0], a["y"] + STEP[d][1])


def can_drop(state: dict, d: str) -> bool:
    a = agent(state)
    return carried(state) is not None and droppable(
        state, a["x"] + STEP[d][0], a["y"] + STEP[d][1])


def can_pick(state: dict, o: dict) -> bool:
    return (o["kind"] in PICKABLE and not o["held"] and carried(state) is None
            and dist(agent(state), o) == 1)


def can_open(state: dict, o: dict) -> bool:
    if o["kind"] != "door" or o["open"] or dist(agent(state), o) != 1:
        return False
    if not o["locked"]:
        return True
    held = carried(state)
    return bool(held and held["kind"] == "key" and held["color"] == o["color"])


def can_put(state: dict, o: dict) -> bool:
    a = agent(state)
    return (carried(state) is not None and not o["held"]
            and dist(a, o) == 1
            and put_cell(state, o, a["x"], a["y"]) is not None)


# -------------------------------------------------------------- perception

def _name(o: dict) -> str:
    return f"{o['color']} {o['kind']}"


def _door_word(o: dict) -> str:
    return "open" if o["open"] else "locked" if o["locked"] else "shut"


def exit_desc(state: dict, d: str) -> str:
    """What is one step away in `d`, as the direction constant's
    description - the dungeon's `_exit_desc` shape."""
    a = agent(state)
    x, y = a["x"] + STEP[d][0], a["y"] + STEP[d][1]
    if tile(state, x, y) == WALL:
        return f"{d}: wall, blocked"
    o = object_at(state, x, y)
    if o is None:
        return f"{d}: floor, you can walk here"
    if o["kind"] == "door":
        if o["open"]:
            return f"{d}: open {_name(o)} {o['id']}, you can walk here"
        return f"{d}: {_name(o)} {o['id']}, {_door_word(o)}"
    return f"{d}: {_name(o)} {o['id']}, blocked"


def _where(state: dict) -> str:
    a = agent(state)
    here = rooms_of(state, a["x"], a["y"])
    names = [room_by_id(state, r)["name"] for r in here]
    if len(names) == 1:
        return f"in the {names[0]}"
    return f"in the doorway between the {names[0]} and the {names[1]}"


def in_view(state: dict) -> List[dict]:
    a = agent(state)
    here = set(rooms_of(state, a["x"], a["y"]))
    return [o for o in objects(state) if not o["held"]
            and set(rooms_of(state, o["x"], o["y"])) & here]


def seen_earlier(state: dict) -> List[dict]:
    view = {o["id"] for o in in_view(state)}
    visited = set(state.get("visited") or [])
    return [o for o in objects(state) if not o["held"] and o["id"] not in view
            and set(rooms_of(state, o["x"], o["y"])) & visited]


def _obj_desc(state: dict, o: dict, earlier: bool = False) -> str:
    a = agent(state)
    where = offset_words(o["x"] - a["x"], o["y"] - a["y"])
    text = f"{_name(o)}, {where}"
    if o["kind"] == "door":
        text += f", {_door_word(o)}"
        visited = set(state.get("visited") or [])
        beyond = [r for r in door_rooms(state, o) if r not in visited]
        if beyond:
            text += f", leads to the {room_by_id(state, beyond[0])['name']}" \
                    f" (not explored)"
    if earlier:
        rid = rooms_of(state, o["x"], o["y"])[0]
        text += f", in the {room_by_id(state, rid)['name']}, out of view"
    return text


def unexplored(state: dict) -> List[str]:
    visited = set(state.get("visited") or [])
    return [r["name"] for r in state["rooms"] if r["id"] not in visited]


def done_text(state: dict) -> str:
    m = state["mission"]
    objs = objects(state)
    return "; ".join(step_text(objs, s) for s, d in zip(m["steps"], m["done"])
                     if d)


def observe(state: dict, vision: Optional[int] = None, *, exits: bool = True,
            paths: bool = False) -> Observation:
    """The turn in words. Directions first (so $0..$3 are stable), then what
    is in view, what you carry, and what you saw in rooms you have been
    in - the constants the model may name. `exits` and `paths` are the
    dungeon's observe options, accepted so callers can pass them; this
    world always states the exits and never the paths."""
    held = carried(state)
    view = sorted(in_view(state), key=lambda o: o["id"])
    earlier = sorted(seen_earlier(state), key=lambda o: o["id"])
    constants: List[dict] = [
        {"type": "STR", "value": d, "desc": exit_desc(state, d)}
        for d in DIRECTIONS
    ]
    for o in view:
        constants.append({"type": "ID:object", "value": o["id"],
                          "desc": _obj_desc(state, o)})
    if held:
        constants.append({"type": "ID:object", "value": held["id"],
                          "desc": f"{_name(held)}, you are carrying it"})
    for o in earlier:
        constants.append({"type": "ID:object", "value": o["id"],
                          "desc": _obj_desc(state, o, earlier=True)})

    carrying = f"the {_name(held)}" if held else "nothing"
    done = done_text(state)
    budget = state.get("turn_budget", 3)
    view_line = "; ".join(f"{o['id']} {_obj_desc(state, o)}"
                          for o in view) or "nothing"
    earlier_line = "; ".join(f"{o['id']} {_obj_desc(state, o, True)}"
                             for o in earlier) or "nothing"
    todo = unexplored(state)
    request = (
        f"Rooms, turn {state.get('turn', 0)}. "
        f"Mission: {state['mission']['text']}.\n"
        f"You are {_where(state)}, carrying {carrying}."
        + (f" Done so far: {done}." if done else "") + "\n"
        f"In view: {view_line}.\n"
        f"Seen earlier: {earlier_line}.\n"
        f"Not explored yet: "
        f"{', '.join('the ' + n for n in todo) if todo else 'nothing'}.\n"
        f"Last turn: {last_turn(state)}.\n"
        f"Choose up to {budget} actions for this turn."
    )
    head = (f"Turn {state.get('turn', 0)}. "
            f"Mission: {state['mission']['text']}. "
            + (f"Done: {done}. " if done else "")
            + f"You are {_where(state)}, carrying {carrying}. ")
    brief = _brief(head, state.get("log") or [], f"Up to {budget} actions.")
    return Observation(request=request, constants=constants, brief=brief)


BRIEF_CHARS = 280


def _brief(head: str, log: List[str], tail: str) -> str:
    """The first version that fits BRIEF_CHARS, dropping last turn's events
    past the first (the one that failed is kept: it is the one to answer)."""
    fails = [e for e in log if e.startswith("failed:")]
    ordered = fails + [e for e in log if not e.startswith("failed:")]

    def compose(keep: int) -> str:
        if not log:
            events = "nothing yet"
        else:
            shown = log if keep >= len(log) else ordered[:keep]
            more = len(log) - len(shown)
            events = "; ".join(shown) + (f"; and {more} more" if more > 0
                                         else "")
        return f"{head}Last turn: {events}. {tail}"

    text = compose(len(log))
    for keep in (len(log), 3, 2, 1):
        text = compose(keep)
        if len(text) <= BRIEF_CHARS:
            return text
    return text


def legal_actions(state: dict) -> List[str]:
    """One-action authoring programs the engine accepts from here."""
    if state.get("status") != "playing":
        return []
    obs = observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    out = []
    for d in DIRECTIONS:
        if can_move(state, d):
            out.append(f"CALL @move ${index[d]}\nSTOP\n")
    for d in DIRECTIONS:
        if can_drop(state, d):
            out.append(f"CALL @drop ${index[d]}\nSTOP\n")
    for o in objects(state):
        if o["id"] not in index:
            continue
        i = index[o["id"]]
        if can_pick(state, o):
            out.append(f"CALL @pick_up ${i}\nSTOP\n")
        if can_open(state, o):
            out.append(f"CALL @open ${i}\nSTOP\n")
        if can_put(state, o):
            out.append(f"CALL @put_next_to ${i}\nSTOP\n")
    return out


def outcome(state: dict) -> dict:
    m = state["mission"]
    return {
        "status": state.get("status", "playing"),
        "won": state.get("status") == "won",
        "dead": state.get("status") == "out_of_time",
        "turn": state.get("turn", 0),
        "funnel": {
            "left_start_room": len(state.get("visited") or []) > 1
            or any(m["done"]),
            "first_step_done": bool(m["done"]) and m["done"][0],
            "mission_done": all(m["done"]),
        },
    }


WORLDS = []
for _name_ in (TRAIN_WORLD, EXAM_WORLD):
    _w = _world(_name_)
    _w["default_state"] = new_state(scenarios_for(_name_)[0])
    WORLDS.append(_w)
WORLD = WORLDS[0]
