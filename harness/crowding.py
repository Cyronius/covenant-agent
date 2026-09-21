"""E-crowded (S0): distractor tools mixed into a task's context.

crowd_world() returns a copy of a task's world with tools (and the entity
field declarations they reference) borrowed from donor domains. The
reference program never touches them; the model must pick the right tools
out of a 30-80 tool context instead of ~9. Donor tools keep their real
impls, so a wrong-but-well-typed call executes (and fails or no-ops
against the task's state) rather than being statically impossible —
matching real deployments, where calling the wrong tool is legal.

Name collisions (tool or entity) between the native world and donors, or
between donors, are skipped rather than renamed.

`foreign=` is the second distractor source: entity-free tools from outside
the themed vocabulary, which is how the imported open schemas enter the
corpus (`data/gen/open_pool.py`, and
`.claude/plans/imported-schemas-as-distractors.md` for why they are
distractors and not worlds). They carry no entities, so they can never
conflict on one; only the tool-name check applies.
"""
from __future__ import annotations

import copy
import random
from typing import List, Optional

# Foreign tools declare no entities, so the entity-merge and entity-conflict
# paths below are no-ops for them; this stands in for a donor world.
_NO_ENTITIES = {"entities": {}}


def crowd_world(world: dict, donors: List[dict], rng: random.Random,
                n_extra_tools: int,
                foreign: Optional[List[dict]] = None) -> dict:
    merged = {
        "name": world["name"], "now": world["now"],
        "entities": dict(world["entities"]),
        "enums": dict(world.get("enums", {})),
        "tools": list(world["tools"]),
        "default_state": world["default_state"],
    }
    tool_names = {t["name"] for t in merged["tools"]}
    native_entities = set(merged["entities"])

    candidates = []
    for donor in donors:
        for tool in donor["tools"]:
            candidates.append((donor, tool))
    for tool in foreign or ():
        candidates.append((_NO_ENTITIES, tool))
    rng.shuffle(candidates)

    added = 0
    for donor, tool in candidates:
        if added >= n_extra_tools:
            break
        if tool["name"] in tool_names:
            continue
        # entities this tool's signature references: ID:/OBJ: types AND
        # field annotations (legacy tools annotate scalar params with
        # ["entity", "fname"] — e.g. crm's invoice amount)
        ents = set()
        for p in tool["params"]:
            t = p["type"]
            if t.startswith("ID:"):
                ents.add(t[3:])
            if p.get("field"):
                ents.add(p["field"][0])
        ret = tool.get("returns") or ""
        for token in ret.replace("LIST ", "").split():
            if token.startswith("OBJ:") or token.startswith("ID:"):
                ents.add(token.split(":", 1)[1])
        # skip if any referenced entity collides with a native entity or a
        # same-named entity already merged from a *different* donor
        conflict = False
        for e in ents:
            if e in native_entities:
                conflict = True
                break
            if e in merged["entities"] and \
                    merged["entities"][e] != donor["entities"][e]:
                conflict = True
                break
        if conflict:
            continue
        for e in ents:
            if e not in merged["entities"]:
                merged["entities"][e] = dict(donor["entities"][e])
        merged["tools"].append(copy.deepcopy(tool))
        tool_names.add(tool["name"])
        added += 1
    return merged
