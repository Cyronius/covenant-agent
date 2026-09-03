"""Course-builder world: the mobi course editor as the planner sees it
(plan .claude/plans/real-sessions-eval-suite.md §2). Held out: never in
training data; it is the E-real-sessions eval world.

Entities and default state are hand-written here. The tool list is
generated from the real agent's schemas (lm-admin frontend tool maps and
lm-python-functions backend factories) by data/gen/world_from_schemas.py
into data/schemas/coursebuilder_tools.json, and loaded below. Regenerate
with:

  python -m data.gen.world_from_schemas
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_JSON = ROOT / "data" / "schemas" / "coursebuilder_tools.json"

DAY = 86400
NOW = 1_760_000_000

_tools = json.loads(TOOLS_JSON.read_text(encoding="utf-8"))["tools"] if TOOLS_JSON.exists() else []

WORLD = {
    "name": "coursebuilder",
    "now": NOW,
    "entities": {
        "course": {
            "id": "ID:course",
            "name": "STR",
            "description": "STR",
            "header": "STR",          # header image path
        },
        "module": {
            "id": "ID:module",
            "name": "STR",
            "type": "STR",            # lesson | quiz
            "position": "INT",
        },
        "element": {
            "id": "ID:element",
            "module": "ID:module",
            "type": "STR",            # paragraph | heading1 | image | accordion | …
            "heading": "STR",
            "text": "STR",
            "backgroundColor": "STR",
            "fontFamily": "STR",
            "image": "STR",           # image URL
            "position": "INT",
        },
        "sub_item": {
            "id": "ID:sub_item",
            "element": "ID:element",
            "title": "STR",
            "text": "STR",
            "image": "STR",
        },
    },
    "enums": {("module", "type"): ["lesson", "quiz"]},
    "tools": _tools,
    "default_state": {
        "entities": {
            "course": [
                {"id": "course_1", "name": "Front Desk Essentials",
                 "description": "Onboarding for new front desk staff.", "header": ""},
            ],
            "module": [
                {"id": "module_1", "name": "Welcome and Overview", "type": "lesson", "position": 0},
                {"id": "module_2", "name": "Greeting Customers", "type": "lesson", "position": 1},
                {"id": "module_3", "name": "Knowledge Check", "type": "quiz", "position": 2},
            ],
            "element": [
                {"id": "element_1", "module": "module_1", "type": "heading1",
                 "heading": "Welcome to the team", "text": "", "backgroundColor": "",
                 "fontFamily": "", "image": "", "position": 0},
                {"id": "element_2", "module": "module_1", "type": "paragraph",
                 "heading": "", "text": "This course covers what a new front desk hire needs on day one.",
                 "backgroundColor": "", "fontFamily": "", "image": "", "position": 1},
                {"id": "element_3", "module": "module_1", "type": "image",
                 "heading": "", "text": "", "backgroundColor": "", "fontFamily": "",
                 "image": "https://example.test/img/lobby.jpg", "position": 2},
                {"id": "element_4", "module": "module_2", "type": "paragraph",
                 "heading": "", "text": "Greet every customer within ten seconds of arrival.",
                 "backgroundColor": "#ffffff", "fontFamily": "", "image": "", "position": 0},
                {"id": "element_5", "module": "module_2", "type": "accordion",
                 "heading": "Common situations", "text": "", "backgroundColor": "#e8f0fe",
                 "fontFamily": "", "image": "", "position": 1},
                {"id": "element_6", "module": "module_2", "type": "flipCards",
                 "heading": "Key terms", "text": "", "backgroundColor": "",
                 "fontFamily": "", "image": "", "position": 2},
                {"id": "element_7", "module": "module_2", "type": "highlight1",
                 "heading": "Tip", "text": "Smile before you speak.", "backgroundColor": "#fff4ce",
                 "fontFamily": "", "image": "", "position": 3},
                {"id": "element_8", "module": "module_3", "type": "multipleChoice",
                 "heading": "How soon should a customer be greeted?", "text": "",
                 "backgroundColor": "", "fontFamily": "", "image": "", "position": 0},
            ],
            "sub_item": [
                {"id": "sub_item_1", "element": "element_5", "title": "Long queue",
                 "text": "Acknowledge everyone waiting, then serve in order.", "image": ""},
                {"id": "sub_item_2", "element": "element_5", "title": "Angry customer",
                 "text": "Stay calm, listen, and offer a next step.", "image": ""},
                {"id": "sub_item_3", "element": "element_6", "title": "Walk-in",
                 "text": "A customer with no appointment.", "image": ""},
                {"id": "sub_item_4", "element": "element_6", "title": "Hold",
                 "text": "A reservation kept without payment.", "image": ""},
            ],
        },
        "outbox": [],
        "payments": [],
    },
}
