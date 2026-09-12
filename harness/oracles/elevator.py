"""Scripted elevator dispatcher.

Nearest-work first, with one extra rule that is the point of this world: if
somebody is due to press the button on the floor the car is already standing
on, and no other job is closer than that wait, hold.

Cost is measured in ticks, which is what makes the hold comparable to a
drive: waiting n ticks for a rider on this floor against |floor - target|
ticks to reach the nearest other job.
"""
from __future__ import annotations

from typing import List, Optional

from runtime.worlds import elevator as ev


def _targets(state: dict) -> List[tuple]:
    """(floor, ticks away) for every job the car could serve now."""
    car = state["entities"]["car"][0]["floor"]
    out = []
    for r in ev.aboard(state):
        out.append((r["dest"], abs(r["dest"] - car)))
    capacity = state.get("capacity", 4)
    if len(ev.aboard(state)) < capacity:
        for r in ev.waiting(state):
            out.append((r["origin"], abs(r["origin"] - car)))
    return out


def _hold_wait(state: dict) -> Optional[int]:
    """Ticks until the next rider appears on the car's own floor."""
    car = state["entities"]["car"][0]["floor"]
    tick = state.get("tick", 0)
    waits = [r["appears"] - tick for r in ev.upcoming(state)
             if r["origin"] == car]
    return min(waits) if waits else None


def plan_turn(state: dict, budget: int = 2) -> str:
    obs = ev.observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    car = state["entities"]["car"][0]["floor"]

    here = [t for t in _targets(state) if t[0] == car]
    if here:
        return "CALL @open_doors\nSTOP\n"

    targets = _targets(state)
    wait = _hold_wait(state)
    nearest = min((c for _, c in targets), default=None)
    if wait is not None and (nearest is None or wait <= nearest):
        return "CALL @hold\nSTOP\n"

    if not targets:
        # nothing live and nothing announced on this floor: sit on the next
        # call's floor rather than burn the clock in the wrong place
        soon = sorted(ev.upcoming(state), key=lambda r: (r["appears"], r["id"]))
        if not soon:
            return "ABORT NOT_FOUND\n"
        floor = soon[0]["origin"]
        if floor == car:
            return "CALL @hold\nSTOP\n"
        return f"CALL @go_to ${index[floor]}\nSTOP\n"

    floor = min(targets, key=lambda t: (t[1], t[0]))[0]
    lines = [f"CALL @go_to ${index[floor]}"]
    # the doors are the reason for the trip, so open them in the same turn
    # when the budget allows it
    if budget > 1:
        lines.append("CALL @open_doors")
    return "\n".join(lines + ["STOP"]) + "\n"
