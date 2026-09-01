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
"""
from __future__ import annotations

import copy
import random
from typing import List


def crowd_world(world: dict, donors: List[dict], rng: random.Random,
                n_extra_tools: int) -> dict:
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
