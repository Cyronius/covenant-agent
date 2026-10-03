"""Scripted player for the grid rooms: BabyAI's bot idea, cut down.

Each turn it reads the first mission step not yet done (either one, for
"and"), picks the subgoal that step needs from what the agent knows - things
in the rooms it has been in - and walks a breadth-first shortest path to it:

  go to X        stand beside the nearest known X
  pick up X      stand beside it, drop whatever is in hand, pick it up
  open X         stand beside the door and open it; a locked one first sends
                 it for the key of that colour
  put X next to Y  pick X up as above, then stand beside Y where a free tile
                 touches both, and put it there

A target it has not seen yet sends it through the nearest door into a room
it has not been in (BabyAI's ExploreSubgoal, babyai/bot.py:452-500). Doors on
the path are opened on the way; a locked one on the path needs its key first.

The turn is the path's first moves up to the budget, cut after the step
that brings a new room into view (the next turn re-plans with what it shows)
and ending with the subgoal's own action.
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Dict, List, Optional, Tuple

from runtime.worlds import rooms as W

Cell = Tuple[int, int]


def _known(state: dict) -> List[dict]:
    """Objects the agent knows about: in a room it has been in."""
    visited = set(state.get("visited") or [])
    return [o for o in W.objects(state) if not o["held"]
            and set(W.rooms_of(state, o["x"], o["y"])) & visited]


def _known_cell(state: dict, c: Cell) -> bool:
    visited = set(state.get("visited") or [])
    return bool(set(W.rooms_of(state, *c)) & visited)


def _bfs(state: dict, goals: set, allow_locked: bool = False
         ) -> Optional[List[Cell]]:
    """Shortest path (cells after the start) through known tiles to any goal
    cell. Shut doors count as passable (they get opened on the way); a
    locked one only with its key in hand, or when `allow_locked`."""
    a = W.agent(state)
    start = (a["x"], a["y"])
    if start in goals:
        return []
    held = W.carried(state)
    prev: Dict[Cell, Optional[Cell]] = {start: None}
    todo = deque([start])
    while todo:
        cur = todo.popleft()
        for d in W.DIRECTIONS:
            dx, dy = W.STEP[d]
            c = (cur[0] + dx, cur[1] + dy)
            if c in prev or W.tile(state, *c) == W.WALL \
                    or not _known_cell(state, c):
                continue
            o = W.object_at(state, *c)
            if o is not None:
                if o["kind"] != "door":
                    continue
                if o["locked"] and not allow_locked and not (
                        held and held["kind"] == "key"
                        and held["color"] == o["color"]):
                    continue
            prev[c] = cur
            if c in goals:
                path = [c]
                while prev[path[-1]] != start:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            todo.append(c)
    return None


def _beside(state: dict, o: dict) -> set:
    return {(o["x"] + dx, o["y"] + dy) for dx, dy in W.STEP.values()}


def _dir(a: Cell, b: Cell) -> str:
    step = (b[0] - a[0], b[1] - a[1])
    return next(d for d, v in W.STEP.items() if v == step)


def _walk(state: dict, path: List[Cell]) -> List[tuple]:
    """Actions along a path: open any shut door first, then step on."""
    a = W.agent(state)
    cur = (a["x"], a["y"])
    out = []
    for c in path:
        o = W.object_at(state, *c)
        if o is not None and o["kind"] == "door" and not o["open"]:
            out.append(("open", o["id"]))
        out.append(("move", _dir(cur, c), c))
        cur = c
    return out


def _dropped(state: dict, d: str) -> dict:
    """The state after dropping what is in hand to `d` (planning only)."""
    trial = copy.deepcopy(state)
    a = W.agent(trial)
    held = W.carried(trial)
    held.update(held=False, x=a["x"] + W.STEP[d][0], y=a["y"] + W.STEP[d][1])
    return trial


def _drop_score(state: dict, d: str) -> tuple:
    """Lower is safer. A careless drop can make the rest of the mission
    impossible three ways, worst first: take the last reachable tile a
    put-next-to needs, cut the floor in two, or block a doorway."""
    trial = _dropped(state, d)
    held = W.carried(state)
    a = W.agent(state)
    c = (a["x"] + W.STEP[d][0], a["y"] + W.STEP[d][1])
    doors = {(o["x"], o["y"]) for o in W.objects(state) if o["kind"] == "door"}
    by_door = any((c[0] + dx, c[1] + dy) in doors for dx, dy in W.STEP.values())
    room_left = True
    step = _current_step(state)
    if step is not None and step["verb"] == "putnext":
        for o in W.objects(trial):
            if o["id"] == held["id"] and W.matches(o, step["obj"]):
                o["held"] = True        # the mover goes back in hand
        spots = {c for f in W.objects(trial)
                 if not f["held"] and W.matches(f, step["fixed"])
                 for c in W.put_spots(trial, f)}
        room_left = bool(spots) and (
            _bfs(trial, spots, allow_locked=True) is not None
            or not all(_known_cell(trial, c) for c in spots))
    return (not room_left, not W._all_reachable(trial), by_door)


def _free_hands(state: dict) -> Optional[tuple]:
    """The safest drop from where the agent stands, and its score."""
    best = None
    for d in W.DIRECTIONS:
        if not W.can_drop(state, d):
            continue
        score = _drop_score(state, d)
        if best is None or score < best[0]:
            best = (score, d)
    return best


def _nearest(state: dict, cands: List[dict], goals_of) -> Optional[tuple]:
    """(target, path) for the candidate with the shortest path."""
    best = None
    for o in sorted(cands, key=lambda o: o["id"]):
        path = _bfs(state, goals_of(o))
        if path is not None and (best is None or len(path) < len(best[1])):
            best = (o, path)
    return best


def _explore(state: dict) -> Optional[List[tuple]]:
    """Walk into the doorway of the nearest room not yet explored."""
    visited = set(state.get("visited") or [])
    doors = [o for o in _known(state) if o["kind"] == "door"
             and set(W.door_rooms(state, o)) - visited]
    hit = _nearest(state, doors, lambda o: {(o["x"], o["y"])})
    if hit is not None:
        return _walk(state, hit[1])
    # every way on is locked: fetch a key that opens one
    for door in sorted(doors, key=lambda o: o["id"]):
        plan = _fetch_key(state, door)
        if plan:
            return plan
    return None


def _fetch_key(state: dict, door: dict) -> Optional[List[tuple]]:
    held = W.carried(state)
    if held and held["kind"] == "key" and held["color"] == door["color"]:
        return None
    keys = [o for o in _known(state)
            if o["kind"] == "key" and o["color"] == door["color"]]
    return _pickup(state, keys)


def _moved(state: dict, path: List[Cell]) -> dict:
    """The state after walking `path`, doors on it opened (planning only)."""
    trial = copy.deepcopy(state)
    for c in path:
        o = W.object_at(trial, *c)
        if o is not None and o["kind"] == "door":
            o.update(open=True, locked=False)
    if path:
        a = W.agent(trial)
        a["x"], a["y"] = path[-1]
    return trial


def _pickup(state: dict, cands: List[dict]) -> Optional[List[tuple]]:
    """Walk beside the nearest candidate and pick it up. Hands full: set the
    load down at the first point along the way with a safe tile beside it
    (see `_drop_score`), or at the least bad one when none is safe."""
    goals = lambda s: (lambda o: _beside(s, o))  # noqa: E731
    hit = _nearest(state, cands, goals(state))
    if hit is None:
        return _unblock(state, cands, goals(state))
    target, path = hit
    if W.carried(state) is None:
        return _walk(state, path) + [("pick_up", target["id"])]
    best = None
    for i in range(len(path) + 1):
        here = _moved(state, path[:i])
        for d in W.DIRECTIONS:
            if not W.can_drop(here, d):
                continue
            score = _drop_score(here, d)
            if best is not None and score >= best[0]:
                continue
            after = _dropped(here, d)
            rest = _nearest(after, [target], goals(after))
            if rest is not None:
                best = (score, i, d, after, rest[1])
        if best is not None and not any(best[0]):
            break
    if best is None:
        return None
    _, i, d, after, rest = best
    return (_walk(state, path[:i]) + [("drop", d)] + _walk(after, rest)
            + [("pick_up", target["id"])])


def _unblock(state: dict, cands: List[dict], goals_of) -> Optional[List[tuple]]:
    """No path: if a locked door is in the way, go and get its key."""
    for o in sorted(cands, key=lambda o: o["id"]):
        path = _bfs(state, goals_of(o), allow_locked=True)
        if path is None:
            continue
        for c in path:
            door = W.object_at(state, *c)
            if door is not None and door["kind"] == "door" and door["locked"]:
                return _fetch_key(state, door)
    return None


def _reach(state: dict, cands: List[dict], goals_of) -> Optional[List[tuple]]:
    hit = _nearest(state, cands, goals_of)
    if hit is None:
        return _unblock(state, cands, goals_of)
    return _walk(state, hit[1])


def _step_plan(state: dict, step: dict) -> Optional[List[tuple]]:
    known = _known(state)
    held = W.carried(state)
    desc = step["obj"]
    verb = step["verb"]

    if verb == "goto":
        if held and W.matches(held, desc) and not any(
                W.matches(o, desc) for o in known):
            drop = _free_hands(state)
            return [("drop", drop[1])] if drop else None
        cands = [o for o in known if W.matches(o, desc)]
        plan = _reach(state, cands, lambda o: _beside(state, o)) \
            if cands else None
        return plan if plan is not None else _explore(state)

    if verb == "pickup":
        cands = [o for o in known if W.matches(o, desc)]
        plan = _pickup(state, cands) if cands else None
        return plan if plan is not None else _explore(state)

    if verb == "open":
        cands = [o for o in known if W.matches(o, desc) and not o["open"]]
        if not cands:
            return _explore(state)
        hit = _nearest(state, cands, lambda o: _beside(state, o))
        if hit is None:
            return _unblock(state, cands, lambda o: _beside(state, o)) \
                or _explore(state)
        door, path = hit
        if door["locked"] and not (held and held["kind"] == "key"
                                   and held["color"] == door["color"]):
            return _fetch_key(state, door) or _explore(state)
        return _walk(state, path) + [("open", door["id"])]

    if verb == "putnext":
        # a locked door between here and the fixed object is dealt with
        # before the mover is picked up: fetching the key means setting the
        # mover down, and picking it up first would undo that every turn
        first = _door_first(state, step["fixed"])
        if first:
            return first
        if not (held and W.matches(held, desc)):
            cands = [o for o in known if W.matches(o, desc)]
            plan = _pickup(state, cands) if cands else None
            return plan if plan is not None else _explore(state)
        fixed = [o for o in known if W.matches(o, step["fixed"])]
        hit = _nearest(state, fixed, lambda o: set(W.put_spots(state, o)))
        if hit is None:
            plan = _unblock(state, fixed,
                            lambda o: set(W.put_spots(state, o))) \
                if fixed else None
            if plan is not None:
                return plan
            return _explore(state)
        target, path = hit
        return _walk(state, path) + [("put_next_to", target["id"])]
    return None


def _door_first(state: dict, fixed_desc: dict) -> Optional[List[tuple]]:
    """The key-fetching plan when a locked door stands between the agent and
    the fixed object (or, while that is still unseen, between it and every
    room left to explore); None when the way is clear."""
    fixed = [o for o in _known(state) if W.matches(o, fixed_desc)]
    if fixed:
        spots = lambda o: set(W.put_spots(state, o))  # noqa: E731
        if _nearest(state, fixed, spots) is not None:
            return None
        plan = _unblock(state, fixed, spots)
        if plan:
            return plan
    # nothing it knows will do: the answer is in a room still to explore
    visited = set(state.get("visited") or [])
    doors = [o for o in _known(state) if o["kind"] == "door"
             and set(W.door_rooms(state, o)) - visited]
    if _nearest(state, doors, lambda o: {(o["x"], o["y"])}) is not None:
        return None
    for door in sorted(doors, key=lambda o: o["id"]):
        plan = _fetch_key(state, door)
        if plan:
            return plan
    return None


def _current_step(state: dict) -> Optional[dict]:
    m = state["mission"]
    for step, done in zip(m["steps"], m["done"]):
        if not done:
            return step
    return None


def _use_key(state: dict) -> Optional[List[tuple]]:
    """Carrying a key whose locked door is known and reachable: open it
    before anything else. The key was fetched for that door, and without
    this rule the next subgoal can set the key down to pick something else
    up and then need it again (BabyAI's bot keeps the key for the same
    reason, `reason='KeepKey'`). Not when the key is itself what a step
    still to do names - then it stays in hand for that."""
    held = W.carried(state)
    if not held or held["kind"] != "key":
        return None
    m = state["mission"]
    for step, done in zip(m["steps"], m["done"]):
        if not done and W.matches(held, step["obj"]):
            return None
    doors = [o for o in _known(state) if o["kind"] == "door" and o["locked"]
             and o["color"] == held["color"]]
    hit = _nearest(state, doors, lambda o: _beside(state, o))
    if hit is None:
        return None
    door, path = hit
    return _walk(state, path) + [("open", door["id"])]


def plan_actions(state: dict) -> Optional[List[tuple]]:
    step = _current_step(state)
    if step is None:
        return None
    return _use_key(state) or _step_plan(state, step)


def plan_turn(state: dict, budget: int = 3) -> str:
    acts = plan_actions(state)
    if not acts:
        return "ABORT NOT_FOUND\n"
    obs = W.observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    visited = set(state.get("visited") or [])
    lines = []
    for act in acts[:budget]:
        verb, arg = act[0], act[1]
        lines.append(f"CALL @{verb} ${index[arg]}")
        if verb == "move" and set(W.rooms_of(state, *act[2])) - visited:
            break                       # a new room comes into view: re-plan
    return "\n".join(lines + ["STOP"]) + "\n"
