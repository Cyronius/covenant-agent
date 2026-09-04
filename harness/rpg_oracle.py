"""A scripted player for the `rpg` world: the engine self-check and the
ceiling every model planner is read against.

It cheats on *perception* — it plans over the whole state, not the 5x5 window
— but not on *mechanics*: it emits ordinary Agent Core programs in authoring
form and they go through resolve -> typecheck -> compile -> sandbox like any
model's. If the oracle cannot win a scenario, the scenario is unwinnable and
a model failing it proves nothing.

Policy, in order: kill what is next to you, drink when hurt, take what you
are standing on, unlock the door you are beside, otherwise walk toward the
current objective (key -> door -> stairs).
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from runtime.worlds import rpg

STEPS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def _passable(state: dict, x: int, y: int, ignore_enemies: bool = False) -> bool:
    if rpg._tile(state, x, y) == rpg.WALL:
        return False
    for d in state["entities"].get("door", []):
        if d["x"] == x and d["y"] == y and not d["open"]:
            return False
    if not ignore_enemies:
        for e in state["entities"].get("enemy", []):
            if e["hp"] > 0 and e["x"] == x and e["y"] == y:
                return False
    return True


def _path(state: dict, start: Tuple[int, int],
          goals: List[Tuple[int, int]]) -> Optional[List[str]]:
    """Shortest route from `start` to any goal tile, as direction names."""
    if not goals:
        return None
    goal_set = set(goals)
    if start in goal_set:
        return []
    seen = {start}
    queue = deque([(start, [])])
    while queue:
        (x, y), route = queue.popleft()
        for name, (dx, dy) in STEPS.items():
            nxt = (x + dx, y + dy)
            if nxt in seen or not _passable(state, *nxt):
                continue
            if nxt in goal_set:
                return route + [name]
            seen.add(nxt)
            queue.append((nxt, route + [name]))
    return None


def _neighbours(state: dict, x: int, y: int) -> List[Tuple[int, int]]:
    return [(x + dx, y + dy) for dx, dy in STEPS.values()
            if _passable(state, x + dx, y + dy)]


def plan_turn(state: dict, budget: int = 3) -> str:
    """The next turn's program, in authoring form ($k indexes the constants
    `rpg.observe` produced for this same state, so directions are $0-$3)."""
    obs = rpg.observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    p = state["entities"]["player"][0]
    here = (p["x"], p["y"])
    enemies = [e for e in state["entities"].get("enemy", []) if e["hp"] > 0]
    items = state["entities"].get("item", [])
    doors = state["entities"].get("door", [])

    def line(tool: str, *args: str) -> str:
        return f"CALL @{tool} " + " ".join(args)

    # 1. anything adjacent gets hit until it drops (never more than the budget)
    for e in enemies:
        if abs(e["x"] - p["x"]) + abs(e["y"] - p["y"]) == 1:
            hits = min(budget, -(-e["hp"] // max(1, p["attack"])))
            arg = f"${index[e['id']]}"
            return "\n".join([line("attack", arg)] * hits + ["STOP"]) + "\n"

    # 2. drink before the next exchange can kill
    potion = next((i for i in items
                   if i["kind"] == "potion" and i.get("held")), None)
    if potion and p["hp"] <= max(4, p["max_hp"] // 2):
        return line("use_item", f"${index[potion['id']]}") + "\nSTOP\n"

    # 3. take what is underfoot
    underfoot = next((i for i in items if not i.get("held")
                      and (i["x"], i["y"]) == here), None)
    if underfoot:
        return line("pick_up", f"${index[underfoot['id']]}") + "\nSTOP\n"

    # 4. unlock the door you are standing beside
    key_held = any(i["kind"] == "key" and i.get("held") for i in items)
    for d in doors:
        if d["open"]:
            continue
        if abs(d["x"] - p["x"]) + abs(d["y"] - p["y"]) != 1:
            continue
        if d["locked"] and not key_held:
            continue
        return line("interact", f"${index[d['id']]}") + "\nSTOP\n"

    # 5. otherwise walk toward the objective
    route = _path(state, here, _goals(state, key_held))
    if not route:
        # boxed in: hit whatever is nearest rather than stalling the episode
        reachable = [e for e in enemies
                     if (e["x"], e["y"]) in _neighbours_all(state, here)]
        if reachable and reachable[0]["id"] in index:
            return line("attack", f"${index[reachable[0]['id']]}") + "\nSTOP\n"
        return "ABORT NOT_FOUND\n"
    moves = [line("move", f"${index[name]}") for name in route[:budget]]
    return "\n".join(moves + ["STOP"]) + "\n"


def _neighbours_all(state: dict, here: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = here
    return [(x + dx, y + dy) for dx, dy in STEPS.values()]


def _goals(state: dict, key_held: bool) -> List[Tuple[int, int]]:
    """Where to head next: the key, then a tile beside the locked door, then
    the stairs (which become reachable once the door is open)."""
    items = state["entities"].get("item", [])
    doors = state["entities"].get("door", [])
    p = state["entities"]["player"][0]

    shut = [d for d in doors if not d["open"]]
    if shut and not key_held:
        key = next((i for i in items
                    if i["kind"] == "key" and not i.get("held")), None)
        if key:
            return [(key["x"], key["y"])]
    if shut and key_held:
        return _neighbours(state, shut[0]["x"], shut[0]["y"])

    m = state["map"]
    exits = [(x, y) for y in range(m["height"]) for x in range(m["width"])
             if rpg._tile(state, x, y) == rpg.EXIT]
    if exits:
        return exits
    return [(p["x"], p["y"])]
