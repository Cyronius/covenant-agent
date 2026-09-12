"""Scripted card player: basic strategy for the rules this table runs
(dealer stands on all 17, double on any first two cards, no splits, no
surrender). One decision per turn, from four numbers.
"""
from __future__ import annotations

HIT, STAND, DOUBLE = "hit", "stand", "double_down"


def decide(total: int, soft: bool, dealer_up: int, can_double: bool) -> str:
    """The strategy table, flattened. `dealer_up` uses the engine's ranks,
    so 1 is an ace and 10 covers every ten-count card."""
    ace_up = dealer_up == 1
    if soft:
        if total >= 19:
            return STAND
        if total == 18:
            if 3 <= dealer_up <= 6 and can_double:
                return DOUBLE
            if dealer_up in (2, 7, 8):
                return STAND
            return HIT
        if total == 17:
            return DOUBLE if (3 <= dealer_up <= 6 and can_double) else HIT
        if total in (15, 16):
            return DOUBLE if (4 <= dealer_up <= 6 and can_double) else HIT
        if total in (13, 14):
            return DOUBLE if (5 <= dealer_up <= 6 and can_double) else HIT
        return HIT
    if total >= 17:
        return STAND
    if 13 <= total <= 16:
        return STAND if 2 <= dealer_up <= 6 else HIT
    if total == 12:
        return STAND if 4 <= dealer_up <= 6 else HIT
    if total == 11:
        return DOUBLE if (can_double and not ace_up) else HIT
    if total == 10:
        return DOUBLE if (can_double and 2 <= dealer_up <= 9) else HIT
    if total == 9:
        return DOUBLE if (can_double and 3 <= dealer_up <= 6) else HIT
    return HIT


def plan_turn(state: dict, budget: int = 1) -> str:
    t = state["entities"]["table"][0]
    can_double = bool(t["can_double"]) and t["cards_taken"] == 0
    tool = decide(t["player_total"], bool(t["player_soft"]), t["dealer_up"],
                  can_double)
    return f"CALL @{tool}\nSTOP\n"
