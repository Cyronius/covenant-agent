"""Synthetic worlds (F3): schema + initial state + tool set with effects.

Each world module exposes WORLD, a plain dict:

  name           str
  entities       {entity: {field_name: type_str}}   ('id' is always ID:<entity>)
  tools          [ {name, desc, params, returns, effects, impl} ]
                 params: [{name, type, desc, required?, field?: [entity, fname]}]
                 impl: descriptor interpreted by runtime/sandbox.js
  default_state  {"entities": {entity: [records]}, "outbox": [], "payments": []}

`projects` is a held-out world (data/holdout/reserved.json): it never enters
training data; it exists for R5.
"""
from . import crm, kanban, projects

WORLDS = {w["name"]: w for w in (kanban.WORLD, crm.WORLD, projects.WORLD)}


def get_world(name: str) -> dict:
    return WORLDS[name]
