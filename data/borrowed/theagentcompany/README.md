# TheAgentCompany wording bank

**What it is.** TheAgentCompany simulates a small software company with
self-hosted GitLab, RocketChat (chat, with simulated coworkers), ownCloud
(files) and Plane (project tracker). Its 175 tasks are workplace requests
written as a colleague would ("collect feedback from everyone who attended…
write it to /workspace/meeting_feedback.xlsx"). This bank holds the task
wording and the services each task needs, for future files and multi-app
worlds (plan, kinds 6 and 8).

- **Source:** https://github.com/TheAgentCompany/TheAgentCompany
- **Revision:** git commit `98b68ef82a47690c316f42fddb05baafaab56851` (2025-11-17)
- **Licence:** MIT, Copyright (c) 2024 TheAgentCompany. Copy in `LICENSE.upstream`.
- **Pulled:** 2026-09-29, shallow git clone.

## What was taken (174 rows in `bank.jsonl`, all `kind: task`)

The full text of `workspaces/tasks/<task>/task.md` for every task except the
`example` template. `meta`:

- `task`: the directory name
- `category`: its prefix (sde 69, hr 29, pm 28, admin 15, ds 14, finance 12, ml 2, qa 2, research 2, bm 1)
- `services`: from `dependencies.yml` (gitlab, rocketchat, owncloud, plane)
- `services_mentioned`: services the text points at by URL port or name. It differs from `services` for 2 tasks.
- `has_npc_scenarios`: the task has a `scenarios.json` (a simulated coworker to talk to)

## Deliberately not taken

- The bundled repos, documents and spreadsheets the services are seeded with
  (`servers/`), and task-local data files.
- `checkpoints.md`, `evaluator.py`, `scenarios.json` contents (NPC profiles and
  goals), Dockerfiles and solutions.
- `servers/rocketchat/npc/task.md`, which is service config, not a task.

**Note:** the texts use the benchmark's host (`the-agent-company.com:<port>`),
its fictional employees (e.g. "Chen Xinyi", "mike.chen@agentcompany.com") and
names of the open-source repos it seeds (OpenHands, JanusGraph, sotopia). Swap
in our own entities before use. Some texts contain upstream template
placeholders such as `{CODEBASE_NAME}`.

## Attribution

> Task descriptions adapted from TheAgentCompany (Xu et al., "TheAgentCompany:
> Benchmarking LLM Agents on Consequential Real World Tasks", 2024),
> https://github.com/TheAgentCompany/TheAgentCompany, MIT License,
> Copyright (c) 2024 TheAgentCompany.
