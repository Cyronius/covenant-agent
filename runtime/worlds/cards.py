"""Card table: family A's smallest instance.

No map, no ids, three tools, and the whole situation is four numbers - your
total, whether it is soft, the dealer's upcard, and whether you may still
double. If a model cannot choose here, grid reading was never what beat it
on the dungeon.

Rules live in `runtime/engines/cards.js`. The only card logic on this side is
the opening deal (state construction); every hand after that is dealt by the
engine, and `observe` reads the numbers the engine syncs onto the table
record.
"""
from __future__ import annotations

import random
from typing import List, Optional

from runtime.worlds.decision import Observation, last_turn

NOW = 1_760_000_000

WORLD = {
    "name": "cards",
    "now": NOW,
    "entities": {
        "table": {"id": "ID:table", "bankroll": "INT", "bet": "INT",
                  "hand_no": "INT", "hands": "INT", "dealer_up": "INT",
                  "player_total": "INT", "player_soft": "BOOL",
                  "cards_taken": "INT", "can_double": "BOOL"},
    },
    "tools": [
        {
            "name": "hit",
            "desc": "Take one more card. Going over 21 loses the hand at "
                    "once.",
            "params": [],
            "returns": "OBJ:table",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "cards", "fn": "hit"},
        },
        {
            "name": "stand",
            "desc": "Take no more cards; the dealer then plays the hand out "
                    "and it settles.",
            "params": [],
            "returns": "OBJ:table",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "cards", "fn": "stand"},
        },
        {
            "name": "double_down",
            "desc": "Double the bet, take exactly one more card, and stop. "
                    "Only allowed on the first two cards of a hand.",
            "params": [],
            "returns": "OBJ:table",
            "effects": ["WRITE"],
            "impl": {"op": "engine", "module": "cards", "fn": "double_down"},
        },
    ],
    "post_hook": {"module": "cards", "fn": "end_turn"},
}

# 1 is an ace; 10 covers ten, jack, queen and king
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

SCENARIOS = {
    # fixed shoes, so every model plays the identical cards
    "eight_hands": {"hands": 8, "bankroll": 100, "base_bet": 10,
                    "seed": 20260910, "max_turns": 60},
    "short_shoe": {"hands": 5, "bankroll": 60, "base_bet": 10,
                   "seed": 20260911, "max_turns": 40},
}


def _shoe(rng: random.Random, decks: int = 4) -> List[int]:
    cards = [r for _ in range(decks * 4) for r in RANKS]
    rng.shuffle(cards)
    return cards


def _score(cards: List[int]) -> tuple:
    """Mirrors `score` in runtime/engines/cards.js - the only place the two
    sides both count, and only for the opening deal."""
    total = sum(11 if c == 1 else c for c in cards)
    aces = sum(1 for c in cards if c == 1)
    soft = aces > 0
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
        soft = aces > 0
    return total, soft


def _deal(state: dict) -> None:
    """Deal hands until one of them needs a decision. Naturals settle
    themselves, exactly as the engine settles them."""
    t = state["entities"]["table"][0]
    while True:
        t["hand_no"] += 1
        if t["hand_no"] > t["hands"]:
            state["status"] = "done"
            return
        t["bet"] = state["base_bet"]
        state["player_cards"] = [_draw(state), _draw(state)]
        state["dealer_cards"] = [_draw(state), _draw(state)]
        p, soft = _score(state["player_cards"])
        d, _ = _score(state["dealer_cards"])
        t["player_total"], t["player_soft"] = p, soft
        t["cards_taken"] = 0
        t["dealer_up"] = state["dealer_cards"][0]
        t["can_double"] = t["bankroll"] >= t["bet"]
        if p != 21 and d != 21:
            return
        if p == 21 and d != 21:
            t["bankroll"] += (t["bet"] * 3) // 2
        elif d == 21 and p != 21:
            t["bankroll"] -= t["bet"]


def _draw(state: dict) -> int:
    card = state["deck"][state["deck_pos"]]
    state["deck_pos"] += 1
    return card


def _state_from(spec: dict) -> dict:
    rng = random.Random(spec["seed"])
    state = {
        "entities": {"table": [{
            "id": "table_1", "bankroll": spec["bankroll"], "bet": 0,
            "hand_no": 0, "hands": spec["hands"], "dealer_up": 0,
            "player_total": 0, "player_soft": False, "cards_taken": 0,
            "can_double": False}]},
        "outbox": [],
        "payments": [],
        "deck": _shoe(rng),
        "deck_pos": 0,
        "player_cards": [],
        "dealer_cards": [],
        "base_bet": spec["base_bet"],
        "start_bankroll": spec["bankroll"],
        "max_turns": spec["max_turns"],
        "turn": 0,
        "status": "playing",
        "log": [],
        "turn_budget": 1,
        "actions_this_turn": 0,
    }
    _deal(state)
    return state


def new_state(scenario: str = "eight_hands") -> dict:
    return _state_from(SCENARIOS[scenario])


def sample_state(rng: random.Random, world=None) -> dict:
    """The bankroll is set from the stake and the number of hands, not drawn
    on its own: a short shoe at a big bet can go broke however well it is
    played, and an unwinnable draw is a training row that teaches a dead
    end."""
    hands = rng.randint(4, 10)
    bet = rng.choice([5, 10, 20])
    return _state_from({"hands": hands,
                        "bankroll": bet * hands * 3,
                        "base_bet": bet,
                        "seed": rng.randrange(1 << 30),
                        "max_turns": 60})


# -------------------------------------------------------------- perception

CARD_WORD = {1: "an ace", 10: "a ten", 2: "a 2", 3: "a 3", 4: "a 4",
             5: "a 5", 6: "a 6", 7: "a 7", 8: "an 8", 9: "a 9"}


def observe(state: dict, vision: Optional[int] = None) -> Observation:
    t = state["entities"]["table"][0]
    constants: List[dict] = [
        {"type": "INT", "value": t["player_total"],
         "desc": "your total this hand"},
        {"type": "INT", "value": t["dealer_up"],
         "desc": "the dealer's upcard (1 is an ace)"},
    ]
    request = (
        f"Card table, turn {state.get('turn', 0)}. Hand {t['hand_no']} of "
        f"{t['hands']}, bankroll {t['bankroll']} chips, {t['bet']} on this "
        f"hand.\n"
        f"You hold {t['player_total']} "
        f"{'soft' if t['player_soft'] else 'hard'} from "
        f"{t['cards_taken'] + 2} cards. The dealer shows "
        f"{CARD_WORD.get(t['dealer_up'], t['dealer_up'])}.\n"
        f"Doubling {'is' if t['can_double'] and t['cards_taken'] == 0 else 'is not'}"
        f" available on this hand.\n"
        f"The dealer draws to 16 and stands on 17. Blackjack pays 3 to 2.\n"
        f"Last turn: {last_turn(state)}.\n"
        f"Choose one action for this turn."
    )
    return Observation(request=request, constants=constants)


def legal_actions(state: dict) -> List[str]:
    t = state["entities"]["table"][0]
    out = ["CALL @hit\nSTOP\n", "CALL @stand\nSTOP\n"]
    if t["cards_taken"] == 0 and t["can_double"]:
        out.append("CALL @double_down\nSTOP\n")
    return out


def outcome(state: dict) -> dict:
    t = state["entities"]["table"][0]
    start = state["start_bankroll"]
    return {
        "status": state.get("status", "playing"),
        # playing the shoe out without going broke, not finishing up. The
        # house edge means even perfect play loses a fixed shoe about half
        # the time, so money is a funnel stage to read, never the bar - the
        # decisions are scored per turn against the oracle.
        "won": state.get("status") == "done",
        "dead": state.get("status") == "broke",
        "turn": state.get("turn", 0),
        "bankroll": t["bankroll"],
        "funnel": {
            "finished_shoe": state.get("status") == "done",
            "kept_half": t["bankroll"] >= start // 2,
            "level_or_better": t["bankroll"] >= start,
        },
    }


WORLD["default_state"] = new_state()
