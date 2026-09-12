"""Scheduling probe world: family G, and an IR question before it is a data
question.

Held out, and deliberately tiny. Its job is to let someone hand-write the
three canonical scheduling asks and see whether Agent Core can express them
at all (`harness/schedule_probe.py`):

  1. the earliest slot free in two people's calendars
  2. the cheapest room that fits a headcount and is free
  3. the shift that still has cover

Plain CRUD impls throughout - no engine. If the IR turns out to need a named
form for list intersection the way `MOST` replaced the count loop (R4), that
is a spec change and it happens before any corpus work, which is what the
plan (.claude/plans/archive/task-families.md §3 G) reserves this for.
"""
from __future__ import annotations

NOW = 1_760_000_000
HOUR = 3600

WORLD = {
    "name": "scheduling",
    "now": NOW,
    "entities": {
        "person": {"id": "ID:person", "name": "STR"},
        "slot": {"id": "ID:slot", "person": "ID:person", "start": "TIME",
                 "end": "TIME", "free": "BOOL"},
        "room": {"id": "ID:room", "name": "STR", "seats": "INT",
                 "cost": "INT", "free": "BOOL"},
        "shift": {"id": "ID:shift", "start": "TIME", "needs": "INT",
                  "covered": "INT"},
    },
    "tools": [
        {"name": "list_people", "desc": "List everyone with a calendar.",
         "params": [], "returns": "LIST OBJ:person", "effects": ["READ"],
         "impl": {"op": "list", "entity": "person"}},
        {"name": "list_slots",
         "desc": "List one person's calendar slots, free and busy.",
         "params": [{"name": "person", "type": "ID:person",
                     "desc": "whose calendar", "field": ["person", "id"]}],
         "returns": "LIST OBJ:slot", "effects": ["READ"],
         "impl": {"op": "list_by", "entity": "slot", "field": "person",
                  "id_param": 0}},
        {"name": "list_rooms", "desc": "List every meeting room.",
         "params": [], "returns": "LIST OBJ:room", "effects": ["READ"],
         "impl": {"op": "list", "entity": "room"}},
        {"name": "list_shifts", "desc": "List every shift on the rota.",
         "params": [], "returns": "LIST OBJ:shift", "effects": ["READ"],
         "impl": {"op": "list", "entity": "shift"}},
        {"name": "book_room",
         "desc": "Book a room for a slot. The room stops being free.",
         "params": [{"name": "room", "type": "ID:room", "desc": "the room",
                     "field": ["room", "id"]},
                    {"name": "slot", "type": "ID:slot", "desc": "the slot",
                     "field": ["slot", "id"]}],
         "returns": "OBJ:room", "effects": ["WRITE"],
         "impl": {"op": "update", "entity": "room", "id_param": 0,
                  "set_const": {"free": False}}},
        {"name": "take_slot",
         "desc": "Take a calendar slot so nobody else can book it.",
         "params": [{"name": "slot", "type": "ID:slot", "desc": "the slot",
                     "field": ["slot", "id"]}],
         "returns": "OBJ:slot", "effects": ["WRITE"],
         "impl": {"op": "update", "entity": "slot", "id_param": 0,
                  "set_const": {"free": False}}},
    ],
}


def _slots(person: str, base: int, pattern: str) -> list:
    """pattern is one character per hour: '.' free, 'x' busy."""
    return [{"id": f"slot_{person}_{i}", "person": person,
             "start": base + i * HOUR, "end": base + (i + 1) * HOUR,
             "free": ch == "."}
            for i, ch in enumerate(pattern)]


def new_state() -> dict:
    base = NOW + 24 * HOUR
    return {
        "entities": {
            "person": [{"id": "person_1", "name": "Ada Brooks"},
                       {"id": "person_2", "name": "Femi Adler"}],
            # the first hour free in both is index 3
            "slot": _slots("person_1", base, ".x..x.")
                    + _slots("person_2", base, "xx.x.."),
            "room": [
                {"id": "room_1", "name": "Harbour", "seats": 4, "cost": 20,
                 "free": True},
                {"id": "room_2", "name": "Foundry", "seats": 10, "cost": 55,
                 "free": True},
                {"id": "room_3", "name": "Mill", "seats": 8, "cost": 35,
                 "free": True},
                {"id": "room_4", "name": "Quay", "seats": 12, "cost": 15,
                 "free": False},
            ],
            "shift": [
                {"id": "shift_1", "start": base, "needs": 3, "covered": 3},
                {"id": "shift_2", "start": base + 8 * HOUR, "needs": 4,
                 "covered": 2},
                {"id": "shift_3", "start": base + 16 * HOUR, "needs": 2,
                 "covered": 2},
            ],
        },
        "outbox": [],
        "payments": [],
    }


WORLD["default_state"] = new_state()
