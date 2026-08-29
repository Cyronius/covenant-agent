# Foundation (F1–F5) — status and exit-criterion evidence

**Date:** 2026-08-29 · **Generator version:** 0.1.0 · **Spec version:** 0.1.0

## Exit criterion (PLAN.md §2, F5)

> A hand-written reference program for every level runs through F2→F3 and
> produces `goal_success = true`, and F4 produces valid data at every level.

**Both halves hold.**

1. **Curriculum reference suite: 35/35 `goal_success`.**
   `python -m harness.curriculum --out data/curriculum_tasks.jsonl --examples spec/examples`
   then `python -m harness.run --tasks data/curriculum_tasks.jsonl` →
   `35/35 goal_success` (`results/curriculum_reference_metrics.jsonl`).
   The suite covers Levels 0–10 (3 tasks each, across all three worlds) plus
   two effect-gate tasks (`expected_status=effect_blocked`). Spot-checked
   mechanics: L8 retry shows the injected `RATE_LIMITED, RATE_LIMITED, ok`
   call log; L10 tasks do a real PAUSE round-trip (`pauses=1`) with the
   continuation segment typed from the returned registers; gate tasks block
   with state untouched.

2. **F4 produces valid data at every level.**
   110 tasks (`--level all`, 10/level) and 22 held-out-world tasks all replay
   through the harness at `goal_success = true` (110/110, 22/22). A
   1,000-task weighted run (Levels 2–8 heavy) completed with 5 resamples and
   100% valid references. Property tests compile 20 random samples per level
   on every `pytest` run.

Test suite: **83 passed** (`python -m pytest tests/`), including round-trips
of every curriculum program, per-diagnostic typechecker tests, compiler
determinism, effect gate, error injection, and PAUSE round-trip tests.

## Preliminary R1-style measurements (NOT the R1 result)

From the 1,000-task weighted run (`results/r1_preliminary.json`):

| metric | value |
|---|---|
| generation validity (parse+typecheck+compile+execute) | 100% (5 resamples/1000) |
| instructions per task | mean 6.1 · p50 6 · p95 14 |
| Agent Core tokens (whitespace) | p50 25 · p95 62 |
| AC/JS token ratio | median **0.63** · mean 0.58 · p95 0.73 |

Caveats, recorded per PLAN.md §14 ("do not quietly widen the bar"):

- R1's setup calls for reference programs authored by a strong model on
  sampled tasks and hand-written JS equivalents. The ratio above compares
  against **our own compiled JS**, which is `rt.*`-call-heavy — hand-written
  idiomatic JS would likely be *shorter*, making the true ratio worse, while
  a real subword tokenizer would compress JS boilerplate heavily, making it
  better. The R1 pass bar (median ≤ 40% of the JS equivalent) is therefore
  **not yet demonstrated**; R1 must run with its stated setup before anything
  downstream of it proceeds.
- The 95%-compile-cleanly half of R1's bar is about a *model-authored*
  program distribution, not the generator's (which is valid by construction).

## Deviations from the plan, and why

1. **Teacher model for English (F4).** Requests are currently produced by a
   template renderer (`data/gen/english.py`, three styles + adversarial
   phrasing material, tagged `teacher=template-v1`). The plan specifies a
   teacher model writing N paraphrases per program; `--teacher` is the hook.
   **Flag:** template-only phrasing diversity is *not* sufficient for R6's
   OOD suite; wire a real teacher before training corpora are cut.
2. **`ASC`/`DESC`** are in the grammar as `SORT` modifiers (spec §3 note):
   without a direction, ordinal tasks ("second oldest") are inexpressible.
3. **`UNBOUND` also covers unknown `C` symbols** (spec §9): the diagnostic
   vocabulary has no `UNKNOWN_CONST`, and a constant with no binding in the
   task context is exactly "unbound symbol".
4. **Held-out reservations** (`data/holdout/reserved.json`): the `projects`
   world plus tool families `kanban.create_card`, `crm.charge_customer`,
   `crm.set_priority`. Adversarial tool *names* (R5) are moot at the input
   level — tools are already presented as bare `T` symbols with descriptions;
   the R5 adversarial arm should instead perturb *descriptions*.
5. **Known generator artifacts** (acceptable noise for now, revisit before
   R3-scale corpora): occasional no-op tasks (e.g. "close the closed
   tickets"), and template English that can read stiffly. Both are valid
   programs with exact labels.

## What Foundation explicitly does not include

Model training (gated on R1/R2), the R2 baseline runs (needs the Qwen-class
checkpoints and target hardware), constrained decoding (R3; the grammar and
schema constraints exist in `core/`, the decoder shim does not), and the R8
trace converter.

## Next actions in plan order

1. **R1** as specified: 1,000 sampled tasks, strong-model-authored reference
   programs (few-shot), ≥100 hand-checked, JS equivalents hand-written,
   tokens counted with the actual tokenizer that R3 will use. Gate: nothing
   downstream until it passes.
2. **R2** in parallel: pick the current smallest Qwen-class instruct models,
   define the latency/size target in `results/R2.md` *before* running.
3. R7's runtime gate already exists and is exercised; R8 trace collection can
   start any time.
