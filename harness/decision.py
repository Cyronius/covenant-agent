"""The family-A registry: which worlds are decision worlds, which oracle
plays each, and which of them are held out.

Family A is "observe, then act": a partial view of a situation, a small set
of tools, one to three straight-line calls, and feedback next turn. The
dungeon (`rpg`) was the first and is the exam; `warehouse`, `elevator` and
`cards` are the training siblings, varied in rendering and topology on
purpose (results/RPG.md is the evidence that a second grid dungeon would
only test the domain); `house` is the second held-out instance, so the
family score is not grid to grid.

Plan: `.claude/plans/archive/task-families.md`.
"""
from __future__ import annotations

import importlib
from typing import List, Optional

# world name -> (world module path, oracle module path)
DECISION_WORLDS = {
    "rpg": ("runtime.worlds.rpg", "harness.rpg_oracle"),
    "warehouse_robot": ("runtime.worlds.warehouse",
                        "harness.oracles.warehouse"),
    "elevator": ("runtime.worlds.elevator", "harness.oracles.elevator"),
    "cards": ("runtime.worlds.cards", "harness.oracles.cards"),
    "house": ("runtime.worlds.house", "harness.oracles.house"),
    "app_settings": ("runtime.worlds.pages", "harness.oracles.pages"),
    "app_checkout": ("runtime.worlds.pages", "harness.oracles.pages"),
    "app_ticket": ("runtime.worlds.pages", "harness.oracles.pages"),
    "app_coursebuilder": ("runtime.worlds.pages", "harness.oracles.pages"),
}

# never in training data (mirrors data/holdout/reserved.json)
HELD_OUT = {"rpg", "house", "app_coursebuilder"}

TRAINABLE = [w for w in DECISION_WORLDS if w not in HELD_OUT]


def world_module(name: str):
    return importlib.import_module(DECISION_WORLDS[name][0])


def oracle_module(name: str):
    return importlib.import_module(DECISION_WORLDS[name][1])


def default_actions(name: str) -> int:
    """The per-turn action budget each world is built around."""
    from runtime.worlds import get_world
    return get_world(name)["default_state"].get("turn_budget", 3)


def scenarios(name: str) -> List[str]:
    """Named games for `name`. The page apps share one module, so it filters
    the scenario table itself."""
    module = world_module(name)
    if hasattr(module, "scenarios_for"):
        return module.scenarios_for(name)
    return sorted(module.SCENARIOS)


def new_state(name: str, scenario: Optional[str] = None) -> dict:
    return world_module(name).new_state(scenario or scenarios(name)[0])


def sample_state(name: str, rng) -> dict:
    """A randomised, solvable instance. The corpus draws from this so the
    family is not four fixed games."""
    return world_module(name).sample_state(rng, world=name)
