"""Shared perception helpers for the decision worlds (family A).

`runtime/worlds/rpg.py` is the template every one of these follows; it keeps
its own copy of these helpers because it is the held-out exam and nothing
here should be able to move it.

A decision world module exposes, on top of the usual WORLD dict:

  SCENARIOS            {name: spec} - fixed games, for paired eval episodes
  new_state(scenario)  a fresh state for one of them
  sample_state(rng)    a randomised, solvable instance (the corpus draws
                       from this so training is not four fixed games)
  observe(state)       -> Observation: the request text and the constants
                       the model may name this turn
  outcome(state)       -> terminal predicate + the progress funnel
  legal_actions(state) -> authoring-form one-action programs that the engine
                       would accept from this state, for the off-path
                       restarts the episode recipe needs

The oracle for each world lives in harness/oracles/ and exposes
plan_turn(state, budget) -> authoring program text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Observation:
    """One turn as the model sees it. `extra` carries whatever a world's UI
    or suite wants alongside (the RPG's window and nearby list).

    `brief` is the same turn for the tiny planner: one line, within its
    request budget (data/gen/brief_budget.py), with the situation carried by
    the constants' descriptions rather than the request
    (.claude/plans/borrowed-worlds.md, "Rendering rules")."""
    request: str
    constants: List[dict]
    extra: dict = field(default_factory=dict)
    brief: str = ""


def offset_words(dx: int, dy: int, axes=("east", "west", "south", "north"),
                 unit: str = "") -> str:
    """(dx, dy) in tiles -> "2 east 1 south"; (0,0) -> "right here"."""
    east, west, south, north = axes
    parts = []

    def leg(n: int, word: str) -> str:
        if not unit:
            return f"{n} {word}"
        return f"{n} {unit if n != 1 else unit.rstrip('s')} {word}"

    if dx:
        parts.append(leg(abs(dx), east if dx > 0 else west))
    if dy:
        parts.append(leg(abs(dy), south if dy > 0 else north))
    return " ".join(parts) if parts else "right here"


def turn_header(state: dict, title: str) -> str:
    return f"{title}, turn {state.get('turn', 0)}."


def last_turn(state: dict) -> str:
    return "; ".join(state.get("log") or []) or "nothing yet"


# The tiny planner keeps 128 request tokens; its tokenizer averages about 2.4
# characters a token on these observations, so 280 characters leaves headroom
# (runtime/worlds/rpg.py BRIEF_CHARS; data/gen/brief_budget.py checks it).
BRIEF_CHARS = 280


def fit_brief(head: str, log: List[str], tail: str,
              cap: int = BRIEF_CHARS) -> str:
    """`head` + "Last turn: <events>. " + `tail`, within `cap` characters.
    A busy turn logs several events; when they do not all fit, the others
    go first and every `failed:` event stays, in order, because the failure
    is what the next move has to answer."""
    def events(keep: int) -> str:
        if not log:
            return "nothing yet"
        kept, others = [], 0
        for e in log:
            if str(e).startswith("failed:") or others < keep:
                kept.append(str(e))
                others += not str(e).startswith("failed:")
        more = len(log) - len(kept)
        return "; ".join(kept) + (f"; and {more} more" if more > 0 else "")

    for keep in (len(log), 3, 2, 1, 0):
        text = f"{head} Last turn: {events(keep)}. {tail}"
        if len(text) <= cap:
            return text
    return text
