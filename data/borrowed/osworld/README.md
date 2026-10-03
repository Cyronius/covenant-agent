# OSWorld wording bank

**What it is.** OSWorld is a desktop-agent benchmark on a real Ubuntu VM: Chrome,
LibreOffice, GIMP, VLC, Thunderbird, VS Code, the shell, and tasks that span
several of them. Each task is a one-line user instruction. This bank holds those
instructions, for future files and multi-app worlds (plan, kinds 6 and 8).

- **Source:** https://github.com/xlang-ai/OSWorld
- **Revision:** git commit `b138d348256078fa634fc3b73567a7337c793e6b` (2026-09-15)
- **Licence:** Apache-2.0, Copyright 2024 XLANG NLP Lab. Copy in `LICENSE.upstream`. No NOTICE file upstream.
- **Pulled:** 2026-09-29, shallow git clone.

## What was taken (361 rows in `bank.jsonl`, all `kind: task`)

The `instruction` string of every task in `evaluation_examples/examples/<domain>/*.json`
(the Ubuntu set), minus 8 dropped (below). `meta`:

- `domain`: the folder (chrome 40, gimp 26, libreoffice_calc 47,
  libreoffice_impress 47, libreoffice_writer 23, multi_apps 99, os 24,
  thunderbird 15, vlc 17, vs_code 23)
- `id`, `related_apps`, `snapshot`
- `idea_source`: upstream's `source` field, usually the tutorial or forum URL the task was based on
- `infeasible`: the evaluator is `infeasible` (the right answer is to refuse). 27 tasks.
- `in_test_all`: listed in `test_all.json` (all are)
- `setup_uses_files`: the task's setup downloads or opens files (258 tasks). The instruction refers to them, but they were not taken.
- `upstream_dataset`: `NL2Bash` (7) or `SheetCopilot` (19) when upstream says the task came from those sets. **Their licences are not verified.** Filter on this field if that matters.

## Deliberately not taken

- `examples_windows/` (the gated Windows set).
- Task `config` (setup steps), `evaluator` blocks, the attached initial-state
  files (documents, spreadsheets, images), and trajectories.
- **8 tasks whose `source` names another benchmark we must not take from:**
  - 2 from GAIA (RED in the licence audit: gated, no reshare): `da52d699…` and `f918266a…`
  - 6 from Mind2Web, which could be Mind2Web test-split tasks: `0d8b7de3…`,
    `121ba48f…`, `59155008…`, `a728a36e…`, `f0b971a1…` and `f5d96daf…`

## Attribution

> Task instructions adapted from OSWorld (Xie et al., "OSWorld: Benchmarking
> Multimodal Agents for Open-Ended Tasks in Real Computer Environments",
> NeurIPS 2024 Datasets and Benchmarks), https://github.com/xlang-ai/OSWorld,
> Apache License 2.0, Copyright 2024 XLANG NLP Lab.
