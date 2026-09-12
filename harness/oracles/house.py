"""Scripted explorer for the `house` world.

Policy: work out what the route to the goal needs (a key for each locked way,
a lit lamp for each dark room), fetch and light those first, then walk the
route - opening a shut way when it is the next step.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from runtime.worlds import house


def _held(state: dict) -> List[dict]:
    return [t for t in state["entities"]["thing"] if t["held"]]


def _lamp_lit(state: dict) -> bool:
    return any(t["kind"] == "lamp" and t["lit"] for t in _held(state))


def _keys(state: dict) -> set:
    return {t["opens"] for t in _held(state) if t["kind"] == "key"}


def route(state: dict, start: str, goal: str, keys: set,
          lit: bool) -> Optional[List[dict]]:
    """Ways to walk from room `start` to room `goal`, given what is in hand.
    A shut-but-unlocked way counts as passable: opening it is a step the
    caller emits when it comes up."""
    if start == goal:
        return []
    seen = {start}
    queue = deque([(start, [])])
    rooms = {r["id"]: r for r in state["entities"]["room"]}
    while queue:
        rid, path = queue.popleft()
        for w in house.ways_from(state, rid):
            if w["to"] in seen:
                continue
            if w["locked"] and w["id"] not in keys:
                continue
            if rooms[w["to"]]["dark"] and not lit:
                continue
            if w["to"] == goal:
                return path + [w]
            seen.add(w["to"])
            queue.append((w["to"], path + [w]))
    return None


def _needs(state: dict) -> Dict[str, str]:
    """What the route to the goal is missing: {"key": way_id} / {"lamp": ""}.
    Computed against a route that ignores locks and darkness, so it names the
    obstacle rather than reporting the goal unreachable."""
    p = state["entities"]["explorer"][0]
    every_key = {w["id"] for w in state["entities"]["way"] if w["locked"]}
    ideal = route(state, p["room"], state["goal_room"], every_key, True)
    if ideal is None:
        return {}
    rooms = {r["id"]: r for r in state["entities"]["room"]}
    keys = _keys(state)
    for w in ideal:
        if w["locked"] and w["id"] not in keys:
            return {"key": w["id"]}
    if not _lamp_lit(state) and any(rooms[w["to"]]["dark"] for w in ideal):
        return {"lamp": ""}
    return {}


def plan_turn(state: dict, budget: int = 3) -> str:
    obs = house.observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    p = state["entities"]["explorer"][0]
    things = state["entities"]["thing"]
    keys, lit = _keys(state), _lamp_lit(state)

    need = _needs(state)
    target_room = state["goal_room"]
    if "lamp" in need:
        lamp = next((t for t in things if t["kind"] == "lamp" and t["held"]),
                    None)
        if lamp:
            return f"CALL @use ${index[lamp['id']]}\nSTOP\n"
        lamp = next((t for t in things if t["kind"] == "lamp" and not t["held"]),
                    None)
        if lamp is None:
            return "ABORT NOT_FOUND\n"
        if lamp["room"] == p["room"]:
            return f"CALL @take ${index[lamp['id']]}\nSTOP\n"
        target_room = lamp["room"]
    elif "key" in need:
        key = next((t for t in things
                    if t["kind"] == "key" and t["opens"] == need["key"]
                    and not t["held"]), None)
        if key is None:
            return "ABORT NOT_FOUND\n"
        if key["room"] == p["room"]:
            return f"CALL @take ${index[key['id']]}\nSTOP\n"
        target_room = key["room"]

    path = route(state, p["room"], target_room, keys, lit)
    if path is None:
        return "ABORT NOT_FOUND\n"
    if not path:
        return "ABORT UNSUPPORTED\n"
    first = path[0]
    if first["shut"]:
        return f"CALL @open ${index[first['id']]}\nSTOP\n"

    # one hop a turn, whatever the budget: only the ways out of the room you
    # are standing in are in this turn's constant table
    return f"CALL @go ${index[first['id']]}\nSTOP\n"
