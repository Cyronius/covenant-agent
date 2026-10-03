"""Grid RPG world: a turn-based dungeon, as a held-out decision probe.

Plan: `.claude/plans/rpg-demo-app.md`. Unlike every other world here, the
tools are not CRUD — the rules live in `runtime/engines/rpg.js` and run
inside the sandbox through the `engine` impl op, so the browser demo and the
headless suite execute identical dynamics. This module owns the schema, the
scenarios, and *perception*: turning a state into the request text and
constants the model sees. It never mutates the game.

Reserved in `data/holdout/reserved.json` — never in training data.

State shape, beyond the standard `entities`/`outbox`/`payments`:
  map              {width, height, rows[]}   '#' wall, '.' floor, 'E' stairs,
                                             'D' the door's tile
  turn             int, incremented by end_turn
  status           playing | won | dead
  log              events from the last turn (rendered as "Last turn:")
  memory           entity ids seen at least once (written by the caller from
                   Observation.memory)
  turn_budget      actions allowed per turn (default 3)
  actions_this_turn spent so far this turn; end_turn resets it
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

NOW = 1_760_000_000

WALL = "#"
EXIT = "E"
DIRECTIONS = ["north", "south", "east", "west"]
DIR_DESC = {
    "north": "direction: north (up on the map, y-1)",
    "south": "direction: south (down on the map, y+1)",
    "east": "direction: east (right on the map, x+1)",
    "west": "direction: west (left on the map, x-1)",
}

WORLD = {
    "name": "rpg",
    "now": NOW,
    "entities": {
        "player": {
            "id": "ID:player",
            "x": "INT",
            "y": "INT",
            "hp": "INT",
            "max_hp": "INT",
            "attack": "INT",
        },
        "enemy": {
            "id": "ID:enemy",
            "kind": "STR",
            "x": "INT",
            "y": "INT",
            "hp": "INT",
            "attack": "INT",
        },
        "item": {
            "id": "ID:item",
            "kind": "STR",     # key | potion
            "x": "INT",
            "y": "INT",
            "held": "BOOL",
        },
        "door": {
            "id": "ID:door",
            "x": "INT",
            "y": "INT",
            "locked": "BOOL",
            "open": "BOOL",
        },
    },
    "enums": {("item", "kind"): ["key", "potion"]},
    "tools": [
        {
            "name": "move",
            "desc": "Walk one tile in a direction. Fails if a wall, a closed "
                    "door, or an enemy is in the way.",
            "params": [{"name": "direction", "type": "STR",
                        "desc": "which way to walk: north, south, east or west"}],
            "returns": "OBJ:player",
            "effects": ["mutates"],
            "impl": {"op": "engine", "module": "rpg", "fn": "move"},
        },
        {
            "name": "attack",
            "desc": "Strike an enemy standing on a tile next to you. Fails if "
                    "it is further away than one tile.",
            "params": [{"name": "target", "type": "ID:enemy",
                        "desc": "the enemy to hit", "field": ["enemy", "id"]}],
            "returns": "OBJ:enemy",
            "effects": ["mutates"],
            "impl": {"op": "engine", "module": "rpg", "fn": "attack"},
        },
        {
            "name": "pick_up",
            "desc": "Take an item lying on the tile you are standing on. Fails "
                    "if the item is anywhere else.",
            "params": [{"name": "item", "type": "ID:item",
                        "desc": "the item on the floor here",
                        "field": ["item", "id"]}],
            "returns": "OBJ:item",
            "effects": ["mutates"],
            "impl": {"op": "engine", "module": "rpg", "fn": "pick_up"},
        },
        {
            "name": "use_item",
            "desc": "Use an item you are carrying. Drinking a potion restores "
                    "health. A key is not used this way — open its door with "
                    "interact instead.",
            "params": [
                {"name": "item", "type": "ID:item",
                 "desc": "the carried item to use", "field": ["item", "id"]},
                {"name": "target", "type": "ID:enemy", "required": False,
                 "desc": "enemy to use it on, when the item needs one",
                 "field": ["enemy", "id"]},
            ],
            "returns": "OBJ:player",
            "effects": ["mutates"],
            "impl": {"op": "engine", "module": "rpg", "fn": "use_item"},
        },
        {
            "name": "interact",
            "desc": "Open a door on a tile next to you. A locked door opens "
                    "only if you are carrying its key, which is used up.",
            "params": [{"name": "target", "type": "ID:door",
                        "desc": "the door to open", "field": ["door", "id"]}],
            "returns": "OBJ:door",
            "effects": ["mutates"],
            "impl": {"op": "engine", "module": "rpg", "fn": "interact"},
        },
    ],
    # runtime/sandbox.js runs this after the program ends: the enemy phase.
    "post_hook": {"module": "rpg", "fn": "end_turn"},
}


# --------------------------------------------------------------- scenarios

KEEP_ROWS = [
    "############",
    "#......#...#",
    "#..g...#.E.#",
    "#......D...#",
    "#......#####",
    "#.p....#..k#",
    "#..........#",
    "#@.....#.g.#",
    "############",
]

SCENARIOS = {
    "keep": {
        "quest": "Find the stairs down. They are in the north-east, behind a "
                 "locked door; the key is somewhere off to the east.",
        "vision": 2,
        "max_turns": 40,
        "rows": KEEP_ROWS,
        "player": {"x": 1, "y": 7, "hp": 10, "max_hp": 10, "attack": 3},
        "enemies": [
            {"kind": "goblin", "x": 3, "y": 2, "hp": 3, "attack": 2},
            {"kind": "goblin", "x": 9, "y": 7, "hp": 3, "attack": 2},
        ],
        "items": [
            {"kind": "potion", "x": 2, "y": 5},
            {"kind": "key", "x": 10, "y": 5},
        ],
        "doors": [{"x": 7, "y": 3, "locked": True, "open": False}],
    },
}


def new_state(scenario: str = "keep") -> dict:
    """A fresh game. The map glyphs are display only — walls are the one
    thing they decide; doors and the exit are entities/tiles the engine
    reads."""
    sc = SCENARIOS[scenario]
    rows = list(sc["rows"])
    return {
        "entities": {
            "player": [dict(id="player_1", **sc["player"])],
            "enemy": [dict(id=f"enemy_{i + 1}", **e)
                      for i, e in enumerate(sc["enemies"])],
            "item": [dict(id=f"item_{i + 1}", held=False, **it)
                     for i, it in enumerate(sc["items"])],
            "door": [dict(id=f"door_{i + 1}", **d)
                     for i, d in enumerate(sc["doors"])],
        },
        "outbox": [],
        "payments": [],
        "map": {"width": len(rows[0]), "height": len(rows), "rows": rows},
        "scenario": scenario,
        "quest": sc["quest"],
        "vision": sc["vision"],
        "max_turns": sc["max_turns"],
        "turn": 0,
        "status": "playing",
        "log": [],
        "memory": [],
        "turn_budget": 3,
        "actions_this_turn": 0,
    }


WORLD["default_state"] = new_state()


# -------------------------------------------------------------- perception

@dataclass
class Observation:
    """What one turn looks like to the model, plus the structured form the UI
    draws its fog from. `memory` is the updated seen-set: the caller writes it
    back to state["memory"] (perception is otherwise pure).

    `brief` is the request for the tiny planner: one line, no grid, short
    enough for its 128-token request budget (the grid alone would not fit,
    and it cannot read one). The situation it leaves out reaches that
    planner through the constants' descriptions instead."""
    request: str
    constants: List[dict]
    window: List[str]
    nearby: List[dict]
    memory: List[str]
    brief: str = ""


def _player(state: dict) -> dict:
    return state["entities"]["player"][0]


def _tile(state: dict, x: int, y: int) -> str:
    m = state["map"]
    if x < 0 or y < 0 or x >= m["width"] or y >= m["height"]:
        return WALL
    row = m["rows"][y]
    return row[x] if x < len(row) else WALL


def _offset_words(dx: int, dy: int) -> str:
    """(dx, dy) in tiles -> "2 east 1 south"; (0,0) -> "on your tile"."""
    parts = []
    if dx:
        parts.append(f"{abs(dx)} {'east' if dx > 0 else 'west'}")
    if dy:
        parts.append(f"{abs(dy)} {'south' if dy > 0 else 'north'}")
    return " ".join(parts) if parts else "on your tile"


def _visible(state: dict, vision: int) -> dict:
    """Everything inside the square window, by kind. Vision is a plain radius:
    no line-of-sight, so a wall does not hide what is behind it (deliberately
    simple — the probe is decision-making, not raycasting)."""
    p = _player(state)
    ents = state["entities"]

    def near(e):
        return abs(e["x"] - p["x"]) <= vision and abs(e["y"] - p["y"]) <= vision

    return {
        "enemy": [e for e in ents.get("enemy", []) if e["hp"] > 0 and near(e)],
        "item": [i for i in ents.get("item", [])
                 if not i.get("held") and near(i)],
        "door": [d for d in ents.get("door", []) if near(d)],
        "held": [i for i in ents.get("item", []) if i.get("held")],
    }


STEP = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def _exit_desc(state: dict, direction: str, vis: dict) -> str:
    """What is one step away in `direction`, as a direction constant's
    description. The ASCII window says the same thing, but only to a reader
    that lines up columns across three lines of flattened text, and the
    tuned planners do not (.claude/plans/rpg-exits-perception.md). Kept to
    one short sentence so a planner that trims descriptions keeps all of it."""
    p = _player(state)
    dx, dy = STEP[direction]
    x, y = p["x"] + dx, p["y"] + dy
    for e in vis["enemy"]:
        if (e["x"], e["y"]) == (x, y):
            return f"{direction}: {e['kind']} {e['id']}, blocked"
    for d in vis["door"]:
        if (d["x"], d["y"]) == (x, y):
            if d["open"]:
                return f"{direction}: open door, you can walk here"
            word = "locked" if d["locked"] else "shut"
            return f"{direction}: {word} door {d['id']}, blocked until opened"
    tile = _tile(state, x, y)
    if tile == WALL:
        return f"{direction}: wall, blocked"
    if tile == EXIT:
        return f"{direction}: stairs down, you can walk here"
    for i in vis["item"]:
        if (i["x"], i["y"]) == (x, y):
            return f"{direction}: floor with a {i['kind']}, you can walk here"
    return f"{direction}: floor, you can walk here"


def _paths(state: dict, vis: dict, vision: int) -> dict:
    """Moves to reach each visible thing, and the first move, found by a
    breadth-first search over the window only, so nothing outside the view
    leaks. "Reach" is standing next to an enemy or door and standing on an
    item. {id: (moves, first direction)}; an id is absent when no path
    exists inside the window."""
    p = _player(state)
    blocked = {(e["x"], e["y"]) for e in vis["enemy"]}
    blocked |= {(d["x"], d["y"]) for d in vis["door"] if not d["open"]}

    def inside(x, y):
        return abs(x - p["x"]) <= vision and abs(y - p["y"]) <= vision

    start = (p["x"], p["y"])
    first = {start: None}
    dist = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for x, y in frontier:
            for name in DIRECTIONS:
                sx, sy = STEP[name]
                cell = (x + sx, y + sy)
                if cell in dist or not inside(*cell) or cell in blocked \
                        or _tile(state, *cell) == WALL:
                    continue
                dist[cell] = dist[(x, y)] + 1
                first[cell] = first[(x, y)] or name
                nxt.append(cell)
        frontier = nxt

    out = {}
    for e in vis["enemy"] + vis["door"]:
        goals = [(e["x"] + sx, e["y"] + sy) for sx, sy in STEP.values()]
        best = min((c for c in goals if c in dist), key=lambda c: dist[c],
                   default=None)
        if best is not None:
            out[e["id"]] = (dist[best], first[best])
    for i in vis["item"]:
        cell = (i["x"], i["y"])
        if cell in dist:
            out[i["id"]] = (dist[cell], first[cell])
    return out


def _path_words(pid: str, paths: dict) -> str:
    """", 3 moves to reach, first move north" — or nothing when already
    there, or ", no path in view"."""
    if pid not in paths:
        return ", no path in view"
    moves, direction = paths[pid]
    if moves == 0:
        return ""
    return f", {moves} move{'s' if moves != 1 else ''} to reach, first move {direction}"


def _glyph(state: dict, x: int, y: int, vis: dict) -> str:
    p = _player(state)
    if x == p["x"] and y == p["y"]:
        return "@"
    for e in vis["enemy"]:
        if e["x"] == x and e["y"] == y:
            return e["kind"][0].lower()
    for i in vis["item"]:
        if i["x"] == x and i["y"] == y:
            return "k" if i["kind"] == "key" else "p"
    for d in vis["door"]:
        if d["x"] == x and d["y"] == y:
            return "/" if d["open"] else "D"
    t = _tile(state, x, y)
    return t if t in (WALL, EXIT) else "."


def observe(state: dict, vision: Optional[int] = None, *, exits: bool = False,
            paths: bool = False) -> Observation:
    """Render one turn: the request text the model reads and the constants it
    may name. Only things it can see (or carries) become constants — the
    kanban lesson (`relevant_cards`, server/dev_server.py): dumping every
    entity made the tuned model pick the wrong one. Directions come first so
    the oracle's $0..$3 are stable.

    `exits`: each direction constant says what is one step that way (wall,
    floor, door, enemy) instead of only which way it points. `paths`: each
    visible thing also says how many moves reach it and which move comes
    first, searched inside the window. Both off reproduces the observation
    every stored E-rpg run was scored on, byte for byte."""
    p = _player(state)
    vision = state.get("vision", 2) if vision is None else vision
    vis = _visible(state, vision)
    routes = _paths(state, vis, vision) if paths else None

    def route(pid: str) -> str:
        return _path_words(pid, routes) if routes is not None else ""

    window = []
    for y in range(p["y"] - vision, p["y"] + vision + 1):
        window.append("".join(_glyph(state, x, y, vis)
                              for x in range(p["x"] - vision, p["x"] + vision + 1)))

    constants: List[dict] = [
        {"type": "STR", "value": d,
         "desc": _exit_desc(state, d, vis) if exits else DIR_DESC[d]}
        for d in DIRECTIONS
    ]
    nearby: List[dict] = []

    for e in sorted(vis["enemy"], key=lambda e: e["id"]):
        dx, dy = e["x"] - p["x"], e["y"] - p["y"]
        adj = abs(dx) + abs(dy) == 1
        where = _offset_words(dx, dy)
        desc = (f"{e['kind']} {where}, {'adjacent' if adj else 'not adjacent'}"
                f", {e['hp']} HP{route(e['id']) if not adj else ''}")
        constants.append({"type": "ID:enemy", "value": e["id"], "desc": desc})
        nearby.append({"id": e["id"], "kind": e["kind"], "type": "enemy",
                       "dx": dx, "dy": dy, "adjacent": adj, "hp": e["hp"]})

    for i in sorted(vis["item"], key=lambda i: i["id"]):
        dx, dy = i["x"] - p["x"], i["y"] - p["y"]
        here = dx == 0 and dy == 0
        desc = (f"{i['kind']} on the floor "
                f"{'here, on your tile' if here else _offset_words(dx, dy)}"
                f"{route(i['id']) if not here else ''}")
        constants.append({"type": "ID:item", "value": i["id"], "desc": desc})
        nearby.append({"id": i["id"], "kind": i["kind"], "type": "item",
                       "dx": dx, "dy": dy, "here": here})

    for d in sorted(vis["door"], key=lambda d: d["id"]):
        dx, dy = d["x"] - p["x"], d["y"] - p["y"]
        adj = abs(dx) + abs(dy) == 1
        state_word = ("open" if d["open"]
                      else "locked" if d["locked"] else "shut")
        desc = (f"{state_word} door {_offset_words(dx, dy)}, "
                f"{'adjacent' if adj else 'not adjacent'}"
                f"{route(d['id']) if not adj else ''}")
        constants.append({"type": "ID:door", "value": d["id"], "desc": desc})
        nearby.append({"id": d["id"], "type": "door", "dx": dx, "dy": dy,
                       "adjacent": adj, "locked": d["locked"], "open": d["open"]})

    for i in sorted(vis["held"], key=lambda i: i["id"]):
        constants.append({"type": "ID:item", "value": i["id"],
                          "desc": f"{i['kind']} you are carrying"})

    memory = list(state.get("memory", []))
    for group in ("enemy", "item", "door"):
        for e in vis[group]:
            if e["id"] not in memory:
                memory.append(e["id"])

    visible_ids = {e["id"] for g in ("enemy", "item", "door") for e in vis[g]}
    remembered = _remembered_line(state, memory, visible_ids)
    remembered_brief = _remembered_line(state, memory, visible_ids,
                                        groups=("item", "door"))
    carrying = ", ".join(sorted(i["kind"] for i in vis["held"])) or "nothing"
    nearby_line = "; ".join(
        f"{n.get('kind', 'door')} {n['id']} "
        f"{'on your tile' if n.get('here') else _offset_words(n['dx'], n['dy'])}"
        for n in nearby) or "nothing"
    last = "; ".join(state.get("log") or []) or "nothing yet"

    request = (
        f"Dungeon crawl, turn {state.get('turn', 0)}. {state.get('quest', '')} "
        f"You are at ({p['x']},{p['y']}) with {p['hp']}/{p['max_hp']} HP, "
        f"carrying {carrying}.\n"
        f"What you can see ({2 * vision + 1}x{2 * vision + 1} tiles around you, "
        f"north is up, you are @ in the middle): "
        f"# wall, . floor, E stairs down, D locked door, / open door, "
        f"g goblin, k key, p potion.\n"
        + "\n".join(window) + "\n"
        f"Nearby: {nearby_line}.\n"
        f"Seen earlier, not visible now: {remembered}.\n"
        f"Last turn: {last}.\n"
        f"Choose up to {state.get('turn_budget', 3)} actions for this turn."
    )
    brief = _brief(f"Turn {state.get('turn', 0)}, "
                   f"{p['hp']}/{p['max_hp']} HP, carrying {carrying}. "
                   f"{state.get('quest', '')} ",
                   remembered_brief.split("; ") if remembered_brief != "nothing" else [],
                   state.get("log") or [],
                   f"Up to {state.get('turn_budget', 3)} actions.")
    return Observation(request=request, constants=constants, window=window,
                       nearby=nearby, memory=memory, brief=brief)


# The tiny planner keeps 128 request tokens and its tokenizer averages about
# 2.4 characters a token on this text, so 280 characters leaves headroom.
# tests/test_tiny_planner_demo.py checks the worst case with the real
# tokenizer.
BRIEF_CHARS = 280


def _brief(head: str, seen: List[str], log: List[str], tail: str) -> str:
    """The first version that fits BRIEF_CHARS, giving up detail in this
    order: last turn's events past the first (a busy turn logs up to six),
    then the remembered things past the first two."""
    def events(keep: int) -> str:
        if not log:
            return "nothing yet"
        more = len(log) - keep
        return "; ".join(log[:keep]) + (f"; and {more} more" if more > 0 else "")

    def compose(n_seen: int, keep: int) -> str:
        shown = seen[:n_seen]
        return (head
                + (f"Seen earlier: {'; '.join(shown)}. " if shown else "")
                + f"Last turn: {events(keep)}. {tail}")

    tries = [(len(seen), 3), (len(seen), 2), (len(seen), 1), (2, 1), (1, 1), (0, 1)]
    for n_seen, keep in tries:
        text = compose(n_seen, keep)
        if len(text) <= BRIEF_CHARS:
            return text
    return text


def _remembered_line(state: dict, memory: List[str],
                     visible_ids: set, groups=("enemy", "item", "door")) -> str:
    """Things seen before and out of view now. The brief passes
    groups=("item", "door"): a goblin's remembered spot is stale by the next
    turn, and the brief has a fixed token budget."""
    ents = state["entities"]
    by_id = {}
    for group in ("enemy", "item", "door"):
        for e in ents.get(group, []):
            by_id[e["id"]] = (group, e)
    out = []
    p = _player(state)
    for eid in memory:
        if eid in visible_ids or eid not in by_id:
            continue
        group, e = by_id[eid]
        if group not in groups:
            continue
        if group == "item" and e.get("held"):
            continue
        if group == "enemy" and e["hp"] <= 0:
            continue
        name = e.get("kind", "door")
        out.append(f"{name} {_offset_words(e['x'] - p['x'], e['y'] - p['y'])}")
    return "; ".join(out) if out else "nothing"


def outcome(state: dict) -> dict:
    """Terminal predicate + the progress funnel the suite scores episodes on
    (goal_success is state equality against one reference — wrong shape for a
    game with many winning play-throughs)."""
    p = _player(state)
    items = state["entities"].get("item", [])
    doors = state["entities"].get("door", [])
    return {
        "status": state.get("status", "playing"),
        "won": state.get("status") == "won",
        "dead": state.get("status") == "dead",
        "turn": state.get("turn", 0),
        "hp": p["hp"],
        # the key is consumed by unlocking, so "taken" means picked up or used
        "key_taken": not any(i["kind"] == "key" and not i.get("held")
                             for i in items),
        "door_opened": all(d["open"] for d in doors) if doors else False,
        "enemies_left": len([e for e in state["entities"].get("enemy", [])
                             if e["hp"] > 0]),
    }
