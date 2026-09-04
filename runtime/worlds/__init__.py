"""Synthetic worlds (F3): schema + initial state + tool set with effects.

Each world module exposes WORLD, a plain dict:

  name           str
  entities       {entity: {field_name: type_str}}   ('id' is always ID:<entity>)
  tools          [ {name, desc, params, returns, effects, impl} ]
                 params: [{name, type, desc, required?, field?: [entity, fname]}]
                 impl: descriptor interpreted by runtime/sandbox.js
  default_state  {"entities": {entity: [records]}, "outbox": [], "payments": []}
  post_hook?     {module, fn} — a runtime/engines/ function the sandbox runs
                 once after each program ends (the rpg world's enemy phase)

A state may carry top-level keys beyond entities/outbox/payments (the rpg
world's map, turn, status). They survive the sandbox round trip; note that
harness/metrics.py's normalize_state ignores them, so anything a task is
scored on by state equality must live under `entities`.

`projects` is a held-out world (data/holdout/reserved.json): it never enters
training data; it exists for R5. `coursebuilder` is held out the same way:
it is the E-real-sessions eval world (harness/real_suite.py). `rpg` is held
out too — a non-CRUD grid world used as a cross-domain decision probe
(.claude/plans/rpg-demo-app.md); its rules live in runtime/engines/rpg.js
and reach the sandbox through the `engine` impl op.
"""
from . import coursebuilder, crm, kanban, projects, rpg

WORLDS = {w["name"]: w for w in (kanban.WORLD, crm.WORLD, projects.WORLD,
                                 coursebuilder.WORLD, rpg.WORLD)}


def get_world(name: str) -> dict:
    return WORLDS[name]
