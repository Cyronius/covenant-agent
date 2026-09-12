"""Scripted warehouse robot.

Policy, in order: finish the drop if you are standing on the dock with the
job aboard, lift the job tote if you are standing on it, top up if you are on
the pad and the rest of the route will not fit in the pack, otherwise drive
toward whatever is next - and route via the pad when the direct route does
not fit.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from runtime.worlds import warehouse as wh

STEPS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def route(state: dict, start: Tuple[int, int],
          goals: List[Tuple[int, int]]) -> Optional[List[str]]:
    """Shortest drive from `start` to any goal bay, as direction names."""
    if not goals:
        return None
    goal_set = set(goals)
    if start in goal_set:
        return []
    seen = {start}
    queue = deque([(start, [])])
    while queue:
        (x, y), path = queue.popleft()
        for name, (dx, dy) in STEPS.items():
            nxt = (x + dx, y + dy)
            if nxt in seen or wh.tile(state, *nxt) == wh.RACK:
                continue
            if nxt in goal_set:
                return path + [name]
            seen.add(nxt)
            queue.append((nxt, path + [name]))
    return None


def _len(r: Optional[List[str]]) -> int:
    return 10 ** 6 if r is None else len(r)


def plan_turn(state: dict, budget: int = 3) -> str:
    obs = wh.observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    r = state["entities"]["robot"][0]
    here = (r["x"], r["y"])
    dock = state["entities"]["dock"][0]
    pad = state["entities"]["charger"][0]
    dock_at = (dock["x"], dock["y"])
    pad_at = (pad["x"], pad["y"])

    job = [t for t in state["entities"]["tote"] if t["id"] in state["job"]]
    carrying = next((t for t in job if t["held"]), None)
    pending = next((t for t in job
                    if not t["held"] and (t["x"], t["y"]) != dock_at), None)

    if carrying and here == dock_at:
        return f"CALL @set_down ${index[carrying['id']]}\nSTOP\n"
    if pending and (pending["x"], pending["y"]) == here:
        if pending["id"] in index:
            return f"CALL @lift ${index[pending['id']]}\nSTOP\n"

    target = dock_at if carrying else (
        (pending["x"], pending["y"]) if pending else dock_at)
    direct = _len(route(state, here, [target]))
    if carrying is None and pending is not None:
        rest = _len(route(state, target, [dock_at]))
    else:
        rest = 0
    need = direct + rest

    # the pad is only worth a detour when the pack cannot cover what is left
    if r["battery"] < need:
        if here == pad_at:
            if r["battery"] < r["max_battery"]:
                return "CALL @charge\nSTOP\n"
        else:
            to_pad = route(state, here, [pad_at])
            if to_pad is not None and len(to_pad) <= r["battery"]:
                return _drive(index, to_pad, budget, r["battery"])

    path = route(state, here, [target])
    if not path:
        # nowhere to go and nothing to do: say so rather than stall the clock
        return "ABORT NOT_FOUND\n"
    return _drive(index, path, budget, r["battery"])


def _drive(index: dict, path: List[str], budget: int, battery: int) -> str:
    steps = path[:min(budget, max(0, battery))]
    if not steps:
        return "ABORT UNSUPPORTED\n"
    lines = [f"CALL @drive ${index[d]}" for d in steps]
    return "\n".join(lines + ["STOP"]) + "\n"
