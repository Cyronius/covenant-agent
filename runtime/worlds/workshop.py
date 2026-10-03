"""Workshop: family A's crafting instance (TextCraft's mechanic, Crafter's
tech tree, our own items).

The agent is given an item to make. Raw materials are gathered from the
wilds; everything else is crafted by recipe, which takes counts of
ingredients and sometimes needs a station (workbench, kiln, loom, anvil)
standing in the workshop. Goals sit several recipes deep, so a turn is a
decision about what the job is still short of, not about where to walk.

What borrowed and from where: data/borrowed/crafter_textcraft/README.md.
Mechanics only; every item name and recipe here is ours.

The three tools all take one `ID:item`. Which one applies is decided by the
item's own description (a raw material is gathered, anything with a recipe is
crafted), so the tool choice needs the descriptions, not the signatures.

One module, two world names, as `pages.py` does:

  workshop   five recipe trees (lantern, writing desk, wool cloak, tea set,
             hunting bow) and their sub-goals - trainable
  boatyard   HELD OUT - the rowing-boat tree: hull plank, oar, tar, caulking
             and the slipway station appear nowhere in `workshop`

Rules live in `runtime/engines/workshop.js`.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from runtime.worlds.decision import Observation, last_turn

NOW = 1_760_000_000

TOOLS = [
    {
        "name": "gather",
        "desc": "Go out and gather a raw material; each trip brings back that "
                "material's usual amount. Only raw materials can be gathered "
                "- anything with a recipe has to be crafted.",
        "params": [{"name": "item", "type": "ID:item",
                    "desc": "the raw material to gather",
                    "field": ["item", "id"]}],
        "returns": "OBJ:item",
        "effects": ["mutates"],
        "impl": {"op": "engine", "module": "workshop", "fn": "gather"},
    },
    {
        "name": "craft",
        "desc": "Make one batch of an item from its recipe, using up the "
                "ingredients. Its station (workbench, kiln, loom, anvil) must "
                "already stand in the workshop; stations are never used up.",
        "params": [{"name": "item", "type": "ID:item",
                    "desc": "the item to make",
                    "field": ["item", "id"]}],
        "returns": "OBJ:item",
        "effects": ["mutates"],
        "impl": {"op": "engine", "module": "workshop", "fn": "craft"},
    },
    {
        "name": "inspect",
        "desc": "Read an item's recipe and how many you hold. Changes "
                "nothing, but still takes one of the turn's actions.",
        "params": [{"name": "item", "type": "ID:item",
                    "desc": "the item to look up",
                    "field": ["item", "id"]}],
        "returns": "STR",
        "effects": [],
        "impl": {"op": "engine", "module": "workshop", "fn": "inspect"},
    },
]

ENTITIES = {
    "item": {"id": "ID:item", "name": "STR", "have": "INT", "raw": "BOOL",
             "station": "BOOL", "gives": "INT"},
}


def _world(name: str) -> dict:
    return {"name": name, "now": NOW, "entities": dict(ENTITIES),
            "tools": [dict(t) for t in TOOLS],
            "post_hook": {"module": "workshop", "fn": "end_turn"}}


# ------------------------------------------------------------- the catalogue

# raw material -> how much one gathering trip brings back
RAW: Dict[str, int] = {
    "timber": 2, "resin": 2, "flax": 2, "stone": 3, "clay": 2, "sand": 2,
    "copper ore": 1, "iron ore": 1, "beeswax": 1, "wool": 2, "bone": 1,
}

STATIONS = {"workbench", "kiln", "loom", "anvil", "slipway"}

# item -> (makes, {ingredient: count}, station or "")
RECIPES: Dict[str, tuple] = {
    # shared basics
    "plank": (2, {"timber": 1}, ""),
    "twine": (1, {"flax": 2}, ""),
    "workbench": (1, {"plank": 4}, ""),
    "kiln": (1, {"stone": 3, "clay": 2}, "workbench"),
    "charcoal": (1, {"timber": 1}, "kiln"),
    # lantern
    "glass pane": (1, {"sand": 2}, "kiln"),
    "brass ingot": (1, {"copper ore": 1, "charcoal": 1}, "kiln"),
    "brass rivet": (4, {"brass ingot": 1}, "workbench"),
    "candle": (2, {"beeswax": 1, "twine": 1}, "workbench"),
    "lantern": (1, {"glass pane": 1, "brass rivet": 2, "candle": 1},
                "workbench"),
    # writing desk
    "iron ingot": (1, {"iron ore": 1, "charcoal": 1}, "kiln"),
    "anvil": (1, {"iron ingot": 2, "stone": 2}, "workbench"),
    "iron nail": (4, {"iron ingot": 1}, "anvil"),
    "varnish": (1, {"resin": 1, "beeswax": 1}, "kiln"),
    "writing desk": (1, {"plank": 3, "iron nail": 4, "varnish": 1},
                     "workbench"),
    # wool cloak
    "yarn": (1, {"wool": 2}, ""),
    "loom": (1, {"plank": 3, "twine": 2}, "workbench"),
    "cloth": (1, {"yarn": 2}, "loom"),
    "bone button": (3, {"bone": 1}, "workbench"),
    "wool cloak": (1, {"cloth": 2, "bone button": 2, "twine": 1},
                   "workbench"),
    # tea set
    "ash": (1, {"timber": 1}, "kiln"),
    "glaze": (2, {"ash": 1, "sand": 1}, ""),
    "clay cup": (1, {"clay": 1, "glaze": 1}, "kiln"),
    "teapot": (1, {"clay": 2, "glaze": 1}, "kiln"),
    "tea set": (1, {"teapot": 1, "clay cup": 2}, ""),
    # hunting bow
    "bow stave": (1, {"plank": 2, "resin": 1}, "workbench"),
    "bowstring": (1, {"twine": 3, "beeswax": 1}, ""),
    "hunting bow": (1, {"bow stave": 1, "bowstring": 1}, "workbench"),
    # boatyard only (held out)
    "tar": (1, {"resin": 2}, "kiln"),
    "hull plank": (2, {"plank": 2, "tar": 1}, "workbench"),
    "oar": (1, {"plank": 2, "twine": 1}, "workbench"),
    "caulking": (1, {"tar": 1, "flax": 2}, ""),
    "slipway": (1, {"stone": 4, "plank": 4}, "workbench"),
    "rowing boat": (1, {"hull plank": 4, "oar": 2, "caulking": 1},
                    "slipway"),
}

# world name -> the trees (goal roots) it draws jobs from
TREES = {
    "workshop": ["lantern", "writing desk", "wool cloak", "tea set",
                 "hunting bow"],
    "boatyard": ["rowing boat"],
}
BOATYARD_ONLY = {"tar", "hull plank", "oar", "caulking", "slipway",
                 "rowing boat"}


def tree_of(item: str) -> List[str]:
    """Every item the recipe tree under `item` touches, `item` first, stations
    included, in a stable depth-first order."""
    out: List[str] = []

    def walk(name: str) -> None:
        if name in out:
            return
        out.append(name)
        if name in RECIPES:
            _, needs, at = RECIPES[name]
            if at:
                walk(at)
            for ing in needs:
                walk(ing)

    walk(item)
    return out


def _catalogue(world_name: str) -> List[str]:
    """The items a world may show at all: the held-out tree's own items
    never appear in `workshop`."""
    names = list(RAW) + list(RECIPES)
    if world_name == "boatyard":
        return names
    return [n for n in names if n not in BOATYARD_ONLY]


# ------------------------------------------------------------------ state

def _state_from(spec: dict) -> dict:
    """spec: world, goal, count, have {name: n}, distractors [names],
    order [names] (the shown order; defaults to tree then distractors),
    max_turns."""
    world_name = spec.get("world", "workshop")
    tree = tree_of(spec["goal"])
    shown = list(spec.get("order") or (tree + [d for d in spec.get(
        "distractors", []) if d not in tree]))
    # every item the job's tree needs must be nameable; distractors on top
    for name in tree:
        if name not in shown:
            shown.append(name)
    # a distractor's own ingredients and station exist (so its recipe reads
    # whole and crafting it fails for a real reason) but are not constants
    names = list(shown)
    for name in shown:
        if name in RECIPES:
            _, needs, at = RECIPES[name]
            for dep in list(needs) + ([at] if at else []):
                if dep not in names:
                    names.append(dep)
    ids = {n: f"item_{i + 1}" for i, n in enumerate(names)}
    have = spec.get("have", {})
    items = [{"id": ids[n], "name": n, "have": int(have.get(n, 0)),
              "raw": n in RAW, "station": n in STATIONS,
              "gives": RAW[n] if n in RAW else RECIPES[n][0]}
             for n in names]
    # only shown items can be named, so only they need their recipe here
    recipes = {}
    for n in names:
        if n not in RECIPES or not all(
                d in ids for d in list(RECIPES[n][1]) + [RECIPES[n][2]] if d):
            continue
        makes, needs, at = RECIPES[n]
        recipes[ids[n]] = {"makes": makes,
                           "needs": [[ids[i], c] for i, c in needs.items()],
                           "at": ids[at] if at else ""}
    count = spec.get("count", 1)
    goal_name = spec["goal"]
    return {
        "entities": {"item": items},
        "outbox": [],
        "payments": [],
        "recipes": recipes,
        "crafted": {},
        "goal": {"item": ids[goal_name], "count": count},
        "shown": [ids[n] for n in shown],
        "world_name": world_name,
        "task": f"Make {count} {goal_name}.",
        "turn": 0,
        "status": "playing",
        "log": [],
        "turn_budget": 3,
        "actions_this_turn": 0,
        "max_turns": spec.get("max_turns")
        or (scratch_actions(goal_name, count) + 2) // 3 + 10,
    }


def scratch_actions(name: str, qty: int) -> int:
    """Actions to make `qty` of `name` from an empty shelf, counting a
    station once per recipe that needs it - an upper bound on the straight
    plan, which is all the turn cap needs."""
    if name in RAW:
        return -(-qty // RAW[name])
    makes, needs, at = RECIPES[name]
    batches = -(-qty // makes)
    return (batches + sum(scratch_actions(i, c * batches)
                          for i, c in needs.items())
            + (scratch_actions(at, 1) if at else 0))


SCENARIOS = {
    "lantern_from_scratch": {
        "world": "workshop", "goal": "lantern", "count": 1,
        "have": {"timber": 1}, "distractors": ["wool", "bone"],
    },
    "desk_with_bench": {
        "world": "workshop", "goal": "writing desk", "count": 1,
        "have": {"workbench": 1, "plank": 1, "stone": 2},
        "distractors": ["clay cup", "yarn"],
    },
    "cloak_half_done": {
        "world": "workshop", "goal": "wool cloak", "count": 1,
        "have": {"workbench": 1, "cloth": 1, "flax": 2},
        "distractors": ["candle"],
    },
    # held out
    "boat_from_scratch": {
        "world": "boatyard", "goal": "rowing boat", "count": 1,
        "have": {"timber": 2}, "distractors": ["wool"],
    },
    "boat_oars": {
        "world": "boatyard", "goal": "oar", "count": 2,
        "have": {"workbench": 1}, "distractors": ["glass pane", "resin"],
    },
}


def scenarios_for(world_name: str) -> List[str]:
    return sorted(k for k, s in SCENARIOS.items()
                  if s.get("world", "workshop") == world_name)


def new_state(scenario: str = "lantern_from_scratch") -> dict:
    return _state_from(SCENARIOS[scenario])


def sample_state(rng: random.Random, world: Optional[str] = None) -> dict:
    """A random job from this world's trees: the tree's root most of the
    time, otherwise one of its crafted intermediates in a small batch; a
    random head start (a workbench already standing, some materials on the
    shelf); and two or three items from elsewhere in the catalogue shown
    beside the job's own, so the right item has to be read for."""
    world_name = world or "workshop"
    root = rng.choice(TREES[world_name])
    tree = tree_of(root)
    crafted = [n for n in tree if n in RECIPES and n not in STATIONS
               and n != root and sum(1 for m in tree_of(n) if m in RECIPES
                                     and m not in STATIONS) >= 2]
    if world_name == "boatyard":
        # a sub-goal of the held-out tree has to be one of its own items,
        # or the exam would be a workshop job under another name
        crafted = [n for n in crafted if n in BOATYARD_ONLY]
    if crafted and rng.random() < 0.4:
        goal = rng.choice(crafted)
        count = rng.randint(1, max(2, RECIPES[goal][0] * 2))
    else:
        goal, count = root, 1
    tree = tree_of(goal)

    have: Dict[str, int] = {}
    if "workbench" in tree and rng.random() < 0.5:
        have["workbench"] = 1
        if "kiln" in tree and rng.random() < 0.4:
            have["kiln"] = 1
    for name in tree:
        if name == goal or name in STATIONS:
            continue
        if rng.random() < 0.25:
            have[name] = rng.randint(1, 3)

    pool = [n for n in _catalogue(world_name) if n not in tree]
    distractors = rng.sample(pool, rng.randint(2, 3))
    for d in distractors:
        if rng.random() < 0.5:
            have[d] = 1 if d in STATIONS else rng.randint(1, 3)
    order = tree + distractors
    rng.shuffle(order)
    spec = {"world": world_name, "goal": goal, "count": count, "have": have,
            "distractors": distractors, "order": order}
    return _state_from(spec)


# -------------------------------------------------------------- perception

def items(state: dict) -> List[dict]:
    return state["entities"]["item"]


def by_id(state: dict) -> Dict[str, dict]:
    return {i["id"]: i for i in items(state)}


def _qty(n: int, name: str) -> str:
    return f"{n} {name}"


def shortfall(state: dict, item_id: str) -> List[tuple]:
    """[(ingredient record, how many short)] for one batch of `item_id`."""
    recipe = state["recipes"].get(item_id)
    if not recipe:
        return []
    idx = by_id(state)
    return [(idx[i], c - idx[i]["have"]) for i, c in recipe["needs"]
            if idx[i]["have"] < c]


def station_missing(state: dict, item_id: str) -> Optional[dict]:
    recipe = state["recipes"].get(item_id)
    if not recipe or not recipe["at"]:
        return None
    st = by_id(state)[recipe["at"]]
    return None if st["have"] > 0 else st


def _item_desc(state: dict, item: dict) -> str:
    """One constant's description. Kept under 120 characters with no ". "
    before the end, because the tiny planner keeps a description's first
    sentence up to that cap (models/tiny/prep.py compact_desc) - the
    shortfall is the part that decides the move, so it comes early."""
    goal = state["goal"]
    tag = " (the job)" if item["id"] == goal["item"] else ""
    if item["raw"]:
        return (f"{item['name']}{tag}: raw, gather brings {item['gives']}; "
                f"you hold {item['have']}")
    recipe = state["recipes"][item["id"]]
    idx = by_id(state)
    ings = ", ".join(_qty(c, idx[i]["name"]) for i, c in recipe["needs"])
    at = f" at a {idx[recipe['at']]['name']}" if recipe["at"] else ""
    makes = f" (makes {recipe['makes']})" if recipe["makes"] > 1 else ""
    if item["station"]:
        if item["have"] > 0:
            return f"{item['name']}: a station, standing in the workshop"
        head = f"{item['name']}: a station, not built; needs {ings}{at}"
    else:
        hold = f"you hold {item['have']}; " if item["have"] else ""
        head = f"{item['name']}{tag}: {hold}needs {ings}{at}{makes}"
    short = shortfall(state, item["id"])
    missing = station_missing(state, item["id"])
    parts = []
    if short and len(short) == len(recipe["needs"]) and all(
            i["have"] == 0 for i, _ in short):
        parts.append("have none of it")
    elif short:
        parts.append("short " + ", ".join(_qty(n, i["name"])
                                          for i, n in short))
    if missing is not None:
        parts.append(f"no {missing['name']} yet")
    return f"{head}; " + ("; ".join(parts) if parts else "ready to craft")


def observe(state: dict, vision: Optional[int] = None, *, exits: bool = False,
            **_) -> Observation:
    """`exits` is accepted for the generator's brief mode
    (data/gen/episodes.py passes it to every world); here the constants
    always carry the recipes, so it changes nothing."""
    idx = by_id(state)
    goal = idx[state["goal"]["item"]]
    count = state["goal"]["count"]
    constants: List[dict] = [
        {"type": "ID:item", "value": iid, "desc": _item_desc(state, idx[iid])}
        for iid in state["shown"]
    ]

    holding = ", ".join(_qty(i["have"], i["name"]) for i in items(state)
                        if i["have"] > 0 and not i["station"]) or "nothing"
    standing = ", ".join(i["name"] for i in items(state)
                         if i["station"] and i["have"] > 0) or "none"
    recipes = "\n".join(
        f"  - {_item_desc(state, idx[iid])}" for iid in state["shown"]
        if not idx[iid]["raw"])
    raws = ", ".join(f"{idx[iid]['name']} ({idx[iid]['gives']} a trip)"
                     for iid in state["shown"] if idx[iid]["raw"])
    request = (
        f"Workshop, turn {state.get('turn', 0)}. {state.get('task', '')} "
        f"You hold {goal['have']} of {count}.\n"
        f"On the shelf: {holding}. Stations standing: {standing}.\n"
        f"Recipes:\n{recipes}\n"
        f"Raw materials you can gather: {raws or 'none'}.\n"
        f"Last turn: {last_turn(state)}.\n"
        f"Choose up to {state.get('turn_budget', 3)} actions for this turn."
    )
    brief = _brief(state, goal, count)
    return Observation(request=request, constants=constants, brief=brief)


# The tiny planner reads 128 request tokens; its tokenizer averages about 2.4
# characters a token on this text (runtime/worlds/rpg.py BRIEF_CHARS), so
# 280 characters leaves headroom. tests/test_workshop.py checks every brief
# it sees with the real counter (data/gen/brief_budget.py).
BRIEF_CHARS = 280


def _brief(state: dict, goal: dict, count: int) -> str:
    """One line: the job, what is standing, last turn. Recipes and holdings
    are in the constants. A failure from last turn is kept verbatim; other
    events are the first thing given up when the line runs long."""
    standing = ", ".join(i["name"] for i in items(state)
                         if i["station"] and i["have"] > 0) or "none"
    head = (f"Workshop, turn {state.get('turn', 0)}. Make {count} "
            f"{goal['name']}, you hold {goal['have']}. "
            f"Stations standing: {standing}. ")
    tail = f"Up to {state.get('turn_budget', 3)} actions."
    log = list(state.get("log") or [])
    failed = [e for e in log if str(e).startswith("failed:")]
    done = [e for e in log if not str(e).startswith("failed:")]

    def compose(keep_done: int) -> str:
        kept = done[:keep_done]
        more = len(done) - keep_done
        events = kept + ([f"and {more} more"] if more > 0 else []) + failed
        text = "; ".join(events) if events else "nothing yet"
        return f"{head}Last turn: {text}. {tail}"

    for keep in range(len(done), -1, -1):
        text = compose(keep)
        if len(text) <= BRIEF_CHARS:
            return text
    return text


def legal_actions(state: dict) -> List[str]:
    """One-call programs the engine accepts from here: gather any raw
    material shown, craft anything whose ingredients and station are in
    hand, inspect anything."""
    idx = by_id(state)
    index = {iid: n for n, iid in enumerate(state["shown"])}
    out = []
    for iid in state["shown"]:
        it = idx[iid]
        if it["raw"]:
            out.append(f"CALL @gather ${index[iid]}\nSTOP\n")
        elif (not shortfall(state, iid)
              and station_missing(state, iid) is None
              and not (it["station"] and it["have"] > 0)):
            out.append(f"CALL @craft ${index[iid]}\nSTOP\n")
    for iid in state["shown"]:
        out.append(f"CALL @inspect ${index[iid]} -> r0\nSTOP\n")
    return out


def outcome(state: dict) -> dict:
    idx = by_id(state)
    goal = idx[state["goal"]["item"]]
    stations = [i for i in items(state) if i["station"]]
    return {
        "status": state.get("status", "playing"),
        "won": state.get("status") == "done",
        "dead": False,
        "turn": state.get("turn", 0),
        "funnel": {
            "crafted_anything": any(state.get("crafted", {}).values()),
            "station_built": any(i["have"] > 0 and
                                 state.get("crafted", {}).get(i["id"])
                                 for i in stations),
            "goal_made": goal["have"] >= state["goal"]["count"],
        },
    }


WORLDS = []
for _name in TREES:
    _w = _world(_name)
    _w["default_state"] = new_state(scenarios_for(_name)[0])
    WORLDS.append(_w)
