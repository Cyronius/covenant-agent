# Borrowed material: sources, licences, attribution

Plan: `.claude/plans/borrowed-worlds.md`. Licence audit: 2026-09-29, against
each source's LICENSE file, dataset card and paper. It is not legal advice;
the YELLOW calls (scraped or crowd-written content) want counsel's sign-off
before anything ships.

What is borrowed is wording, task templates, action vocabularies and
mechanics. Every training row built from it is labelled by our own oracle
and executed in our sandbox; no benchmark's labels or trajectories are
used. Nothing here comes from a GPT-generated trajectory set (AgentInstruct,
AgentTraj, ToolBench), a gated set (GAIA, WorkArena instances) or a test
split (Mind2Web's).

Rules:
- every generated row's `provenance` names the source and the bank file;
- `python data/borrowed/canary_check.py` scans for known do-not-train
  markers (Mind2Web's and AppWorld's canary identifiers, "BENCHMARK DATA SHOULD
  NEVER APPEAR"); `data.gen.episodes` runs it on every file it writes and
  deletes the file on a hit;
- the corpus stays private;
- each folder keeps the upstream licence as `LICENSE.upstream` and says in
  its README what was taken and what was deliberately not.

The banks are rebuilt by `data/borrowed/build_banks.py` from clones at the
revisions each README records.

## Wording banks (not yet built into worlds)

| folder | source | licence | revision | taken |
|---|---|---|---|---|
| `alfworld/` | ALFWorld / ALFRED | MIT | `aaba6870` | 6 task types (+ sliced), 24 goal templates, 21 command templates, 51 observation strings, receptacle and object vocabulary with affordances. No scene layouts (room graphs; the house exam stays a transfer test) |
| `webarena/` | WebArena | Apache-2.0 | `dce04686` | 223 human intent templates (slots only; 18 naming real brands/places dropped, 44 naming WebArena's own sites flagged `meta.site_names_to_rewrite`), 12 browser actions. Taken from its test config: training on these contaminates any future WebArena score, which this project does not report |
| `osworld/` | OSWorld | Apache-2.0 | `b138d348` | 361 Ubuntu-set instructions with app folder. No Windows set, setup files or evaluators; 8 tasks sourced from GAIA or Mind2Web dropped. 7 from NL2Bash and 19 from SheetCopilot kept but unverified licences (`meta.upstream_dataset` filters them) |
| `mind2web/` | Mind2Web | CC-BY-4.0 (task text) | HF `main@17ece8eb` | 1,009 train-split tasks (`confirmed_task`, website, domain, action strings). Test split never downloaded; no HTML |
| `webshop/` | WebShop | MIT code; MTurk instructions | `64fa2a5c` | goal templates, `search[]`/`click[]`, buttons and page types, a 3,000-instruction sample with attributes. No product data or ids. YELLOW: some instructions name brands |
| `travelplanner/` | TravelPlanner | MIT (code only) | `e52c87f4` | 8 tool descriptions, hard-constraint types and values, commonsense checks and violation messages, plan fields. No queries (GPT-4 written), database or plans |
| `theagentcompany/` | TheAgentCompany | MIT | `98b68ef8` | 174 `task.md` texts with the services each needs. No seeded content, evaluators or coworker profiles |
| `bfcl/` | Berkeley Function Calling Leaderboard | Apache-2.0 | gorilla `6ea57973` | 562 hand-written function definitions for the distractor pool. No questions, answers or live categories |

## Attribution

- Adapted from ALFWorld (Shridhar et al., ICLR 2021), https://github.com/alfworld/alfworld, MIT License, Copyright (c) 2020 Mohit Shridhar and others; built on ALFRED (Shridhar et al., CVPR 2020).
- Adapted from WebArena (Zhou et al., ICLR 2024), https://github.com/web-arena-x/webarena, Apache License 2.0.
- Adapted from OSWorld (Xie et al., NeurIPS 2024 D&B), https://github.com/xlang-ai/OSWorld, Apache License 2.0, Copyright 2024 XLANG NLP Lab.
- Task descriptions from Mind2Web (Deng et al., NeurIPS 2023 D&B), train split, https://huggingface.co/datasets/osunlp/Mind2Web, CC BY 4.0; selected fields extracted and rewritten.
- Adapted from WebShop (Yao et al., NeurIPS 2022), https://github.com/princeton-nlp/WebShop, MIT License, Copyright (c) 2023 Princeton Natural Language Processing.
- Adapted from the TravelPlanner code (Xie et al., ICML 2024), https://github.com/OSU-NLP-Group/TravelPlanner, MIT License, Copyright (c) 2024 OSU Natural Language Processing.
- Adapted from TheAgentCompany (Xu et al., 2024), https://github.com/TheAgentCompany/TheAgentCompany, MIT License, Copyright (c) 2024 TheAgentCompany.
- Function definitions from the Berkeley Function Calling Leaderboard (Patil et al., Gorilla project), https://github.com/ShishirPatil/gorilla, Apache License 2.0.

## Built into worlds

| folder | source | licence | revision | world | taken |
|---|---|---|---|---|---|
| `babyai/` | BabyAI, Minigrid | BSD-3-Clause, Apache-2.0 | BabyAI `65fb0cb`, Minigrid `8ea099e` | `rooms` (trainable), `rooms_after` (held out) | mission sentences and connectives (`verifier.py`), description and article rules, which objects a mission may name (`levelgen.py`), colours, locked door keeps its key, one object carried, exploring for an unseen target (`bot.py`). Not taken: location phrases, turning, boxes that open |
| `crafter_textcraft/` | ADaPT (TextCraft), Crafter | MIT, MIT | ADaPT `ecdc4ab0`, Crafter `e04542a2` | `workshop` (trainable), `boatyard` (held out) | the crafting mechanic (gather, craft by recipe with counts, stations) and the tech-tree idea. Every item name and recipe is ours; no Minecraft data |
| `tau2/` | tau2-bench | MIT | `5bfa7e37` | `service_retail`, `service_airline` (trainable), `service_telecom` (held out) | train-split tasks, policies, tool names and descriptions, database extracts. Requests are our own templated briefs, not tau2's `reason_for_call`; test-split users are on `denylist.json` and never drawn |

- Mission grammar from BabyAI (Chevalier-Boisvert et al., ICLR 2019), https://github.com/mila-iqia/babyai, BSD 3-Clause License; rules after Minigrid, https://github.com/Farama-Foundation/Minigrid, Apache License 2.0.
- Crafting mechanic after TextCraft in ADaPT (Prasad et al., 2023), https://github.com/archiki/ADaPT, MIT License, and Crafter (Hafner, 2021), https://github.com/danijar/crafter, MIT License.
- Task scenarios, tool names and descriptions, policies and database records adapted from tau2-bench (Sierra Research), https://github.com/sierra-research/tau2-bench, MIT License.
