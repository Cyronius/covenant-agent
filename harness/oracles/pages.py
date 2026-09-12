"""Scripted operator for the page worlds.

Policy: finish the screen you are on before leaving it - set every value the
task names, then press its save button - and otherwise walk the navigation
toward the screen holding the next thing that is still wrong.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from runtime.worlds import pages


def _by_id(state: dict) -> dict:
    return {e["id"]: e for e in state["entities"]["element"]}


def unmet(state: dict) -> Tuple[List[tuple], List[str]]:
    """(values still wrong, save buttons still unpressed)."""
    goal = state["goal"]
    by_id = _by_id(state)
    vals = [(eid, want) for eid, want in goal["values"].items()
            if by_id.get(eid, {}).get("value") != want]
    subs = [eid for eid in goal["submitted"]
            if not by_id.get(eid, {}).get("done")]
    return vals, subs


def _hop(state: dict, start: str, goal: str) -> Optional[str]:
    """First screen to open on the way from `start` to `goal`."""
    if start == goal:
        return None
    seen = {start}
    queue = deque([(start, None)])
    while queue:
        sid, first = queue.popleft()
        for nxt in state["nav"].get(sid, []):
            if nxt in seen:
                continue
            step = first or nxt
            if nxt == goal:
                return step
            seen.add(nxt)
            queue.append((nxt, step))
    return None


def plan_turn(state: dict, budget: int = 3) -> str:
    obs = pages.observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    by_id = _by_id(state)
    here = state["entities"]["app"][0]["screen"]
    vals, subs = unmet(state)

    lines: List[str] = []
    for eid, want in vals:
        el = by_id[eid]
        if el["screen"] != here or eid not in index:
            continue
        if el["kind"] == "toggle":
            lines.append(f"CALL @click ${index[eid]}")
        elif want in index:
            lines.append(f"CALL @fill ${index[eid]} ${index[want]}")
        if len(lines) >= budget:
            break
    if lines:
        return "\n".join(lines + ["STOP"]) + "\n"

    for eid in subs:
        el = by_id[eid]
        if el["screen"] != here:
            continue
        # a save button only works once the screen's own fields are right
        if any(by_id[v]["screen"] == here for v, _ in vals):
            continue
        blank = next((e for e in pages.elements_on(state, here)
                      if e["required"] and not e["value"]), None)
        if blank is not None:
            # the form demands a value nobody gave us; saying so beats
            # pressing a button that will keep failing
            return "ABORT NEEDS_INFO\n"
        if eid in index:
            return f"CALL @submit ${index[eid]}\nSTOP\n"

    target = None
    if vals:
        target = by_id[vals[0][0]]["screen"]
    elif subs:
        target = by_id[subs[0]]["screen"]
    elif state["goal"].get("screen"):
        target = state["goal"]["screen"]
    if target is None:
        return "ABORT UNSUPPORTED\n"
    step = _hop(state, here, target)
    if step is None or step not in index:
        return "ABORT NOT_FOUND\n"
    return f"CALL @open ${index[step]}\nSTOP\n"
