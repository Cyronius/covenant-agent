# Covenant Agent

Can a small neural network reliably construct correct executable programs
from an English request and arbitrary tool schemas? This directory is the
measurement infrastructure for that question — see [PLAN.md](PLAN.md) (the
governing document) and [results/foundation.md](results/foundation.md) for
current status.

**Status:** Foundation (F1–F5) complete — the exit criterion holds
(35/35 curriculum reference tasks at `goal_success=true`; the generator
produces valid data at every level). Next: R1 (IR fitness) and R2
(off-the-shelf baseline), per PLAN.md §11.

## Layout

| path | what |
|---|---|
| `PLAN.md` | the experimental plan; every change lands here first (§14) |
| `spec/agent_core.md` | Agent Core IR: grammar, types, effects, PAUSE protocol |
| `spec/examples/` | hand-written reference programs, Levels 0–10 (authoring form) |
| `core/` | parser → typechecker → effect checker → JS compiler (`pipeline.build` is the one entry point) |
| `runtime/sandbox.js` | Node `vm` sandbox: generic tool engine, effect gate, error injection, PAUSE |
| `runtime/worlds/` | kanban, crm, projects, coursebuilder, rpg (the last three are held out) |
| `runtime/engines/` | non-CRUD world rules the sandbox loads by name (`rpg.js`: the grid dungeon) |
| `client/` | demo apps: `kanban-ui`, `rpg-ui`, and `shared/` (planner, prompt, validate, inference) |
| `harness/` | task contexts + symbol assignment, runner, metrics, curriculum, authoring; `task_grammar.py` rebuilds the GBNF per task from the symbols that task declares |
| `data/gen/` | F4 backtranslation generator (worlds → programs → execution → English) |
| `data/holdout/reserved.json` | worlds/tools that never enter training |
| `results/` | one file per result; the harness is the only source of accuracy numbers |
| `tests/` | round-trip, property, diagnostic, sandbox tests |

## Quickstart

```bash
cd covenant-agent
python -m pytest tests/                    # 83 tests

# build the curriculum reference tasks + spec examples, then self-check
python -m harness.curriculum --out data/curriculum_tasks.jsonl --examples spec/examples
python -m harness.run --tasks data/curriculum_tasks.jsonl   # expect 35/35

# generate training data (never includes held-out worlds/tools)
python -m data.gen --level 3 --n 10000 --seed 42 --out data/L3.jsonl
python -m data.gen --level all --n 1100 --seed 7 --out data/mix.jsonl

# held-out worlds/tools for R5 evaluation only
python -m data.gen --level all --n 220 --holdout --out data/holdout/eval.jsonl
```

Requires Python 3.11+ (stdlib + pytest) and Node 18+ (stdlib only).

## The pipeline

```
request + tool schemas ──planner──▶ Agent Core text
  ──parse──▶ AST ──typecheck──▶ ──effects──▶ ──compile──▶ JS
  ──sandbox──▶ (PAUSE round-trips) ──▶ final state ──▶ metrics JSONL
```

A *planner* is any `(request, ctx, segment_idx, registers) -> program text`
callable; `harness.run.reference_planner` replays stored references, and
model planners (R2+) plug into the same seam.

Every task is a self-contained JSON row: per-request symbol assignment
(`T`/`F`/`C` symbols are always fresh), initial world state, expected final
state, reference program + call log, injections, tags, provenance.

## Working conventions

PLAN.md §14 applies verbatim: spec/parser/typechecker/compiler/generator/
examples change together with round-trip tests passing; the harness is the
only source of truth for success; failures are written to `results/` rather
than the bar being widened; no new IR primitives, architectures, or
experiments without a note in PLAN.md naming the motivating result.
