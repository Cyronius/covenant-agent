"""Scripted crafter for the workshop worlds: a recursive recipe planner.

To make n of an item: a raw material is gathered until there is enough; a
crafted one first needs its station standing (made the same way), then every
ingredient in hand at once, then one batch crafted - repeated until there is
enough. Obtaining one ingredient can use up another (both a workbench and the
desk want planks), so the ingredient check loops until all of them hold at the
same time. The recipe trees are acyclic, so the loop ends.

The plan is simulated on a copy of the shelf, and the turn is its first
`budget` actions. Re-planning every turn from the real state is what makes it
recover from any detour.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from runtime.worlds import workshop as ws


def full_plan(state: dict) -> List[Tuple[str, str]]:
    """[(verb, item id)] that finishes the job from this state."""
    idx = ws.by_id(state)
    shelf: Dict[str, int] = {i["id"]: i["have"] for i in ws.items(state)}
    recipes = state["recipes"]
    steps: List[Tuple[str, str]] = []

    def obtain(iid: str, qty: int, depth: int = 0) -> None:
        if depth > 40:
            raise RuntimeError(f"recipe loop at {iid}")
        while shelf[iid] < qty:
            item = idx[iid]
            if item["raw"]:
                steps.append(("gather", iid))
                shelf[iid] += item["gives"]
                continue
            recipe = recipes[iid]
            if recipe["at"]:
                obtain(recipe["at"], 1, depth + 1)
            while any(shelf[i] < c for i, c in recipe["needs"]):
                for i, c in recipe["needs"]:
                    obtain(i, c, depth + 1)
            steps.append(("craft", iid))
            for i, c in recipe["needs"]:
                shelf[i] -= c
            shelf[iid] += recipe["makes"]

    goal = state["goal"]
    obtain(goal["item"], goal["count"])
    return steps


def plan_turn(state: dict, budget: int = 3) -> str:
    steps = full_plan(state)
    if not steps:
        return "ABORT NOT_FOUND\n"
    index = {iid: n for n, iid in enumerate(state["shown"])}
    lines = [f"CALL @{verb} ${index[iid]}" for verb, iid in steps[:budget]]
    return "\n".join(lines + ["STOP"]) + "\n"
