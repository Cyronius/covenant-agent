"""Page worlds: family C, UI navigation and form filling.

Family A in the skin the agent actually ships in. A screen is the situation -
what is on it, what it currently says, where you can go from here - and the
five tools are what a person does to a screen. The engine
(`runtime/engines/page.js`) and the episode machinery are family A's,
unchanged; what is new is that the goal is a form's end state.

Four apps, identical tool sets, deliberately different shapes:

  app_settings       three tabs, toggles and text fields, save per tab
  app_checkout       a linear flow with a select and a required address
  app_ticket         one long form with a select and an optional attachment
  app_coursebuilder  HELD OUT - the outline / module / element-panel shape of
                     the real builder (data/schemas/mobi_frontend_tools.json:
                     add_element takes a type, props and a position, so the
                     panel here asks for exactly those)

`app_coursebuilder` is reserved in data/holdout/reserved.json with the
dungeon and the house. It is what makes this family close toward R8.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from runtime.worlds.decision import Observation, last_turn

NOW = 1_760_000_000

TOOLS = [
    {
        "name": "open",
        "desc": "Open another screen from the navigation on the screen you "
                "are on. Fails if there is no way there from here.",
        "params": [{"name": "screen", "type": "ID:screen",
                    "desc": "the screen to open", "field": ["screen", "id"]}],
        "returns": "OBJ:screen",
        "effects": ["WRITE"],
        "impl": {"op": "engine", "module": "page", "fn": "open"},
    },
    {
        "name": "click",
        "desc": "Click a link, a toggle or a plain button on this screen. "
                "Not for anything that takes a typed value, and not for the "
                "button that saves a form.",
        "params": [{"name": "control", "type": "ID:element",
                    "desc": "the control to click",
                    "field": ["element", "id"]}],
        "returns": "OBJ:element",
        "effects": ["WRITE"],
        "impl": {"op": "engine", "module": "page", "fn": "click"},
    },
    {
        "name": "fill",
        "desc": "Put a value into a field or a dropdown on this screen. A "
                "dropdown only takes one of the choices it lists.",
        "params": [{"name": "field", "type": "ID:element",
                    "desc": "the field to fill", "field": ["element", "id"]},
                   {"name": "value", "type": "STR",
                    "desc": "what to put in it"}],
        "returns": "OBJ:element",
        "effects": ["WRITE"],
        "impl": {"op": "engine", "module": "page", "fn": "fill"},
    },
    {
        "name": "submit",
        "desc": "Save the form on this screen with its save button. Fails "
                "while any required field on the screen is still empty.",
        "params": [{"name": "button", "type": "ID:element",
                    "desc": "the form's save button",
                    "field": ["element", "id"]}],
        "returns": "OBJ:element",
        "effects": ["WRITE"],
        "impl": {"op": "engine", "module": "page", "fn": "submit"},
    },
    {
        "name": "read",
        "desc": "Read back what a field or label on this screen currently "
                "says.",
        "params": [{"name": "element", "type": "ID:element",
                    "desc": "the element to read",
                    "field": ["element", "id"]}],
        "returns": "STR",
        "effects": ["READ"],
        "impl": {"op": "engine", "module": "page", "fn": "read"},
    },
]

ENTITIES = {
    "app": {"id": "ID:app", "screen": "ID:screen"},
    "screen": {"id": "ID:screen", "name": "STR"},
    "element": {"id": "ID:element", "screen": "ID:screen", "kind": "STR",
                "label": "STR", "value": "STR", "options": "STR",
                "target": "STR", "required": "BOOL", "done": "BOOL"},
}


def _world(name: str) -> dict:
    return {"name": name, "now": NOW, "entities": dict(ENTITIES),
            "enums": {("element", "kind"):
                      ["field", "select", "toggle", "link", "button",
                       "submit", "text"]},
            "tools": [dict(t) for t in TOOLS],
            "post_hook": {"module": "page", "fn": "end_turn"}}


# ------------------------------------------------------------------- apps
# element: (kind, label, starting value, extras)
#   extras: required, options (list), target (screen key)

APPS: Dict[str, dict] = {
    "app_settings": {
        "start": "profile",
        "screens": {
            "profile": [
                ("field", "Display name", "Ada Brooks", {"required": True}),
                ("field", "Contact email", "ada@acme.test", {"required": True}),
                ("submit", "Save profile", "", {}),
            ],
            "notifications": [
                ("toggle", "Weekly digest", "on", {}),
                ("toggle", "Mentions", "on", {}),
                ("field", "Reply-to address", "team@acme.test",
                 {"required": True}),
                ("submit", "Save notifications", "", {}),
            ],
            "billing": [
                ("select", "Plan", "team",
                 {"options": ["starter", "team", "business"]}),
                ("field", "Billing email", "ap@acme.test", {"required": True}),
                ("submit", "Save billing", "", {}),
            ],
        },
        "names": {"profile": "Profile", "notifications": "Notifications",
                  "billing": "Billing"},
        "nav": {"profile": ["notifications", "billing"],
                "notifications": ["profile", "billing"],
                "billing": ["profile", "notifications"]},
    },
    "app_checkout": {
        "start": "cart",
        "screens": {
            "cart": [
                ("text", "Basket", "2 items, 48.00", {}),
                ("link", "Continue to delivery", "", {"target": "shipping"}),
            ],
            "shipping": [
                ("field", "Street", "", {"required": True}),
                ("field", "Postcode", "", {"required": True}),
                ("select", "Speed", "standard",
                 {"options": ["standard", "express", "collect"]}),
                ("submit", "Save address", "", {"target": "payment"}),
            ],
            "payment": [
                ("select", "Method", "card", {"options": ["card", "invoice"]}),
                ("field", "Reference", "", {}),
                ("submit", "Place order", "", {"target": "done"}),
            ],
            "done": [("text", "Confirmation", "Order placed", {})],
        },
        "names": {"cart": "Basket", "shipping": "Delivery",
                  "payment": "Payment", "done": "Confirmation"},
        "nav": {"cart": ["shipping"], "shipping": ["cart", "payment"],
                "payment": ["shipping"], "done": []},
    },
    "app_ticket": {
        "start": "new_ticket",
        "screens": {
            "new_ticket": [
                ("field", "Subject", "", {"required": True}),
                ("field", "Description", "", {"required": True}),
                ("select", "Priority", "normal",
                 {"options": ["low", "normal", "high", "urgent"]}),
                ("select", "Queue", "support",
                 {"options": ["support", "billing", "engineering"]}),
                ("toggle", "Notify watchers", "off", {}),
                ("link", "Attachments", "", {"target": "attachments"}),
                ("submit", "Raise ticket", "", {}),
            ],
            "attachments": [
                ("field", "File name", "", {}),
                ("button", "Attach", "", {}),
                ("link", "Back to the ticket", "", {"target": "new_ticket"}),
            ],
        },
        "names": {"new_ticket": "New ticket", "attachments": "Attachments"},
        "nav": {"new_ticket": ["attachments"],
                "attachments": ["new_ticket"]},
    },
    # held out
    "app_coursebuilder": {
        "start": "outline",
        "screens": {
            "outline": [
                ("text", "Course", "Fire safety refresher", {}),
                ("field", "Course title", "Fire safety refresher",
                 {"required": True}),
                ("link", "Open module 2", "", {"target": "module"}),
                ("submit", "Save course", "", {}),
            ],
            "module": [
                ("text", "Module", "2 - Evacuation routes", {}),
                ("field", "Module name", "Evacuation routes",
                 {"required": True}),
                ("link", "Add an element", "", {"target": "element"}),
                ("submit", "Save module", "", {}),
            ],
            "element": [
                ("select", "Element type", "text",
                 {"options": ["text", "image", "video", "quiz", "accordion"]}),
                ("field", "Heading", "", {"required": True}),
                ("field", "Body", "", {}),
                ("select", "Position", "end",
                 {"options": ["start", "after the current element", "end"]}),
                ("submit", "Add element", "", {"target": "module"}),
            ],
        },
        "names": {"outline": "Course outline", "module": "Module editor",
                  "element": "New element"},
        "nav": {"outline": ["module"], "module": ["outline", "element"],
                "element": ["module"]},
    },
}

VALUE_BANK = {
    "email": ["ops@acme.test", "alerts@acme.test", "billing@acme.test",
              "duty@acme.test"],
    "name": ["Ada Brooks", "Femi Adler", "Rosa Quist", "Kai Lund"],
    "street": ["14 Harbour Row", "3 Mill Lane", "88 Foundry Street"],
    "postcode": ["EC1 4RT", "M2 7HG", "BS1 5PQ"],
    "text": ["Alarm panel beeping in stairwell B", "Door badge reader down",
             "Projector will not wake", "Kitchen tap leaking"],
}


def _bank_for(label: str) -> List[str]:
    low = label.lower()
    if "email" in low or "address" in low:
        return VALUE_BANK["email"]
    if "name" in low or "title" in low or "heading" in low:
        return VALUE_BANK["name"] + ["Evacuation routes", "Assembly points"]
    if "street" in low:
        return VALUE_BANK["street"]
    if "postcode" in low:
        return VALUE_BANK["postcode"]
    return VALUE_BANK["text"]


# ------------------------------------------------------------------ state

def _build(app_name: str, start: Optional[str] = None) -> dict:
    """Screens and elements with ids, before a goal is attached."""
    app = APPS[app_name]
    keys = list(app["screens"])
    sid = {k: f"screen_{i + 1}" for i, k in enumerate(keys)}
    screens = [{"id": sid[k], "name": app["names"][k]} for k in keys]
    elements = []
    addr: Dict[tuple, str] = {}
    n = 0
    for k in keys:
        for i, (kind, label, value, extra) in enumerate(app["screens"][k]):
            n += 1
            eid = f"element_{n}"
            addr[(k, i)] = eid
            elements.append({
                "id": eid, "screen": sid[k], "kind": kind, "label": label,
                "value": value,
                "options": "|".join(extra.get("options", [])),
                "target": sid.get(extra.get("target", ""), ""),
                "required": bool(extra.get("required")),
                "done": False,
            })
    return {
        "entities": {
            "app": [{"id": "app_1", "screen": sid[start or app["start"]]}],
            "screen": screens,
            "element": elements,
        },
        "outbox": [],
        "payments": [],
        "nav": {sid[k]: [sid[t] for t in app["nav"][k]] for k in keys},
        "app_name": app_name,
        "screen_ids": sid,
        "addr": {f"{k}:{i}": v for (k, i), v in addr.items()},
        "turn": 0,
        "status": "playing",
        "log": [],
        "turn_budget": 3,
        "actions_this_turn": 0,
    }


SCENARIOS = {
    "settings_digest": {
        "app": "app_settings", "start": "profile", "max_turns": 15,
        "task": "Turn the weekly digest off, point the reply-to address at "
                "ops@acme.test, and save the notification settings.",
        "values": {"notifications:0": "off",
                   "notifications:2": "ops@acme.test"},
        "submitted": ["notifications:3"],
    },
    "settings_plan": {
        "app": "app_settings", "start": "notifications", "max_turns": 15,
        "task": "Move us onto the business plan and send the invoices to "
                "billing@acme.test, then save the billing page.",
        "values": {"billing:0": "business", "billing:1": "billing@acme.test"},
        "submitted": ["billing:2"],
    },
    "checkout_express": {
        "app": "app_checkout", "start": "cart", "max_turns": 15,
        "task": "Deliver to 14 Harbour Row, EC1 4RT by express, then place "
                "the order on invoice with reference PO-4471.",
        "values": {"shipping:0": "14 Harbour Row", "shipping:1": "EC1 4RT",
                   "shipping:2": "express", "payment:0": "invoice",
                   "payment:1": "PO-4471"},
        "submitted": ["shipping:3", "payment:2"],
    },
    "ticket_urgent": {
        "app": "app_ticket", "start": "new_ticket", "max_turns": 15,
        "task": "Raise an urgent ticket for engineering: subject \"Door badge "
                "reader down\", description \"Nobody can get into the east "
                "wing\", and let the watchers know.",
        "values": {"new_ticket:0": "Door badge reader down",
                   "new_ticket:1": "Nobody can get into the east wing",
                   "new_ticket:2": "urgent", "new_ticket:3": "engineering",
                   "new_ticket:4": "on"},
        "submitted": ["new_ticket:6"],
    },
    # held out
    "coursebuilder_element": {
        "app": "app_coursebuilder", "start": "outline", "max_turns": 18,
        "task": "In module 2, add a quiz element headed \"Check your route\" "
                "at the start, and save it.",
        "values": {"element:0": "quiz", "element:1": "Check your route",
                   "element:3": "start"},
        "submitted": ["element:4"],
    },
    "coursebuilder_rename": {
        "app": "app_coursebuilder", "start": "element", "max_turns": 18,
        "task": "Rename module 2 to \"Assembly points\" and save the module.",
        "values": {"module:1": "Assembly points"},
        "submitted": ["module:3"],
    },
}


def scenarios_for(world_name: str) -> List[str]:
    return sorted(k for k, s in SCENARIOS.items() if s["app"] == world_name)


def _attach_goal(state: dict, spec: dict) -> dict:
    addr = state["addr"]
    state["task"] = spec["task"]
    state["max_turns"] = spec["max_turns"]
    state["goal"] = {
        "screen": state["screen_ids"].get(spec.get("goal_screen", ""), ""),
        "values": {addr[k]: v for k, v in spec.get("values", {}).items()},
        "submitted": [addr[k] for k in spec.get("submitted", [])],
    }
    return state


def new_state(scenario: str = "settings_digest") -> dict:
    spec = SCENARIOS[scenario]
    return _attach_goal(_build(spec["app"], spec.get("start")), spec)


def sample_state(rng: random.Random, world: Optional[str] = None) -> dict:
    """A random job in one app: a screen picked at random, one to three of
    its inputs given new values, and its save button if it has one. The task
    sentence is templated from the same material, so nothing here needs a
    teacher model."""
    app_name = world or rng.choice(
        [a for a in APPS if a != "app_coursebuilder"])
    app = APPS[app_name]
    keys = list(app["screens"])
    fillable = [k for k in keys
                if any(kind in ("field", "select", "toggle")
                       for kind, *_ in app["screens"][k])]
    target_key = rng.choice(fillable)
    # a linear flow has dead ends (the checkout's confirmation screen has no
    # way back), so only start somewhere the job can actually be done from
    starts = [k for k in keys
              if k != target_key and _reaches(app, k, target_key)]
    start = rng.choice(starts or [target_key])
    state = _build(app_name, start)

    inputs = [(i, e) for i, e in enumerate(app["screens"][target_key])
              if e[0] in ("field", "select", "toggle")]
    rng.shuffle(inputs)
    chosen = inputs[:rng.randint(1, min(3, len(inputs)))]
    values, phrases = {}, []
    for i, (kind, label, value, extra) in chosen:
        if kind == "select":
            options = [o for o in extra["options"] if o != value]
            v = rng.choice(options or extra["options"])
            phrases.append(f'set {label.lower()} to {v}')
        elif kind == "toggle":
            v = "off" if value == "on" else "on"
            phrases.append(f'turn {label.lower()} {v}')
        else:
            v = rng.choice([x for x in _bank_for(label) if x != value])
            phrases.append(f'put "{v}" in {label.lower()}')
        values[f"{target_key}:{i}"] = v

    submits = [i for i, e in enumerate(app["screens"][target_key])
               if e[0] == "submit"]
    if submits:
        # a form will not save while one of its required fields is empty, so
        # a job that asks for a save has to name them all - otherwise the
        # task is unfinishable and the oracle is teaching a dead end
        for i, (kind, label, value, extra) in enumerate(
                app["screens"][target_key]):
            key = f"{target_key}:{i}"
            if kind != "field" or not extra.get("required") or value:
                continue
            if key in values:
                continue
            v = rng.choice(_bank_for(label))
            values[key] = v
            phrases.append(f'put "{v}" in {label.lower()}')

    spec = {
        "app": app_name, "start": start, "max_turns": 18,
        "task": f"On {app['names'][target_key]}: "
                + _join(phrases)
                + (", then save it." if submits else "."),
        "values": values,
        "submitted": [f"{target_key}:{submits[0]}"] if submits else [],
    }
    return _attach_goal(state, spec)


def _reaches(app: dict, start: str, goal: str) -> bool:
    seen, queue = {start}, [start]
    while queue:
        k = queue.pop()
        if k == goal:
            return True
        for nxt in app["nav"].get(k, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _join(parts: List[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# -------------------------------------------------------------- perception

def _app(state: dict) -> dict:
    return state["entities"]["app"][0]


def screen_by_id(state: dict, sid: str) -> dict:
    return next(s for s in state["entities"]["screen"] if s["id"] == sid)


def elements_on(state: dict, sid: str) -> List[dict]:
    return [e for e in state["entities"]["element"] if e["screen"] == sid]


def _describe(e: dict) -> str:
    if e["kind"] == "toggle":
        return f'"{e["label"]}" toggle, currently {e["value"]}'
    if e["kind"] == "select":
        return (f'"{e["label"]}" dropdown, currently {e["value"]}, choices: '
                + ", ".join(e["options"].split("|")))
    if e["kind"] == "field":
        shown = e["value"] or "empty"
        req = ", required" if e["required"] else ""
        return f'"{e["label"]}" text field, currently {shown}{req}'
    if e["kind"] == "submit":
        return f'"{e["label"]}" save button'
    if e["kind"] == "link":
        return f'"{e["label"]}" link'
    if e["kind"] == "button":
        return f'"{e["label"]}" button'
    return f'"{e["label"]}" label reading {e["value"]}'


def observe(state: dict, vision: Optional[int] = None) -> Observation:
    a = _app(state)
    here = screen_by_id(state, a["screen"])
    on_screen = elements_on(state, a["screen"])
    nav = state["nav"].get(a["screen"], [])

    constants: List[dict] = []
    for sid in nav:
        s = screen_by_id(state, sid)
        constants.append({"type": "ID:screen", "value": sid,
                          "desc": f"the {s['name']} screen"})
    for e in on_screen:
        constants.append({"type": "ID:element", "value": e["id"],
                          "desc": _describe(e)})
    for value in _value_constants(state, on_screen):
        constants.append(value)

    lines = "\n".join(f"  - {_describe(e)}" for e in on_screen) or "  (empty)"
    going = ", ".join(screen_by_id(state, s)["name"] for s in nav) or "nowhere"
    request = (
        f"{here['name']}, turn {state.get('turn', 0)}. {state.get('task', '')}\n"
        f"On this screen:\n{lines}\n"
        f"From here you can open: {going}.\n"
        f"Last turn: {last_turn(state)}.\n"
        f"Choose up to {state.get('turn_budget', 3)} actions for this turn."
    )
    return Observation(request=request, constants=constants)


def _value_constants(state: dict, on_screen: List[dict]) -> List[dict]:
    """The values the task names, plus a couple of near misses - the same
    rule the rest of the corpus follows: the request's literals become
    constants, and a table with only the right answer in it decides the task
    without the model reading anything."""
    wanted = list(dict.fromkeys(state["goal"]["values"].values()))
    out = [{"type": "STR", "value": v, "desc": f'the text "{v}"'}
           for v in wanted]
    have = set(wanted)
    for e in on_screen:
        for opt in (e["options"].split("|") if e["options"] else []):
            if opt and opt not in have:
                have.add(opt)
                out.append({"type": "STR", "value": opt,
                            "desc": f'the "{e["label"]}" choice {opt}'})
    return out


def legal_actions(state: dict) -> List[str]:
    obs = observe(state)
    index = {c["value"]: i for i, c in enumerate(obs.constants)}
    a = _app(state)
    out = []
    for sid in state["nav"].get(a["screen"], []):
        out.append(f"CALL @open ${index[sid]}\nSTOP\n")
    values = [c["value"] for c in obs.constants if c["type"] == "STR"]
    for e in elements_on(state, a["screen"]):
        ref = f"${index[e['id']]}"
        if e["kind"] in ("link", "button", "toggle"):
            out.append(f"CALL @click {ref}\nSTOP\n")
        elif e["kind"] == "field" and values:
            out.append(f"CALL @fill {ref} ${index[values[0]]}\nSTOP\n")
        elif e["kind"] == "select":
            opts = [o for o in e["options"].split("|") if o in index]
            if opts:
                out.append(f"CALL @fill {ref} ${index[opts[0]]}\nSTOP\n")
        elif e["kind"] == "submit":
            out.append(f"CALL @submit {ref}\nSTOP\n")
        out.append(f"CALL @read {ref} -> r0\nSTOP\n")
    return out


def outcome(state: dict) -> dict:
    goal = state["goal"]
    by_id = {e["id"]: e for e in state["entities"]["element"]}
    set_ok = sum(1 for k, v in goal["values"].items()
                 if by_id.get(k, {}).get("value") == v)
    saved = sum(1 for k in goal["submitted"] if by_id.get(k, {}).get("done"))
    return {
        "status": state.get("status", "playing"),
        "won": state.get("status") == "done",
        "dead": False,
        "turn": state.get("turn", 0),
        "funnel": {
            "any_value_set": set_ok > 0,
            "values_set": set_ok == len(goal["values"]),
            "saved": bool(goal["submitted"]) and saved == len(goal["submitted"]),
        },
    }


WORLDS = []
for _name in APPS:
    _w = _world(_name)
    _scenarios = scenarios_for(_name)
    _w["default_state"] = new_state(_scenarios[0]) if _scenarios \
        else _attach_goal(_build(_name), {"task": "", "max_turns": 15})
    WORLDS.append(_w)
