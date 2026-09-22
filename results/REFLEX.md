# The demo suites have a floor, and on one of them it is above the model

**Date:** 2026-09-21. **Instruments:** `harness/reflex_baseline.py` (the
null) and `harness/filter_check.py` (the per-row flag), both model-free and
GPU-free — every number below reproduces on a laptop in under a minute.
**Plan:** `.claude/plans/agent-loop-and-ir-review.md` step 2a.

Two changes moved the Database Analyst demo before anyone asked what it
would score untouched: the conjunct cap on `/db_prompt` (35 → 55%) and
`static_repair` deleting the dead conjunct (41 → 48%). Neither says how much
of that is the model, because the alternative was never scored. This is that
score: a planner with no comprehension, no model and no grammar, writing the
corpus's modal skeleton — list → `FILTER` → act — every single time.

97.2% of training rows that call a list tool then `FILTER`
(`general-agent-plan.md` §2B), so that skeleton is what the corpus teaches.
What it earns unaided is the line every demo number has to clear, the same
job `chance_tool_sig` does for grounding (`results/R9.md` §3).

---

## 1. The null

```bash
python -m harness.reflex_baseline --tasks data/holdout/e_db_requests.jsonl
python -m harness.reflex_baseline --tasks data/holdout/e_demo_requests.jsonl
```

| suite | tasks | `goal_success` | `return_match` | `compile_ok` | `filter_padded` | `correct_abstain` |
|---|---|---|---|---|---|---|
| `e_db_requests` | 75 | **80.0%** | 16.7% (10/60) | 100.0% | 45.3% | 0/5 |
| `e_demo_requests` | 70 | 14.3% | 0.0% (0/10) | 100.0% | 70.0% | 0/5 |

Rows: `results/logs/reflex_e_db_requests.jsonl`,
`results/logs/reflex_e_demo_requests.jsonl`.

## 2. `goal_success` on the Analyst demo is unfalsifiable, and now it is measured

**A planner that cannot read scores 80.0% `goal_success` on
`e_db_requests`.** Not because it answers anything — it answers almost
nothing — but because **65 of the 75 tasks expect an unchanged state**, and
a program that returns the wrong list changes nothing either.
`harness/metrics.py:103-109` has said this in a comment since
`return_match` was added ("a whole read-only world … is unfalsifiable under
it"); this is the number behind the comment.

So no `goal_success` figure on `e_db_requests` supports any claim about the
Analyst demo, and the 35–55% that motivated two fixes was never that column
— it is **`return_match`**, which is the honest one. Against it the null is
**16.7%**, and a demo at 35–55% is genuinely two to three times the floor.
That is a real margin, and it is smaller than a reading against zero
suggests.

On `e_demo_requests` (kanban, where only 15 of 70 tasks are read-only)
`goal_success` keeps its meaning: the null is **14.3%** against a demo that
reads ~42% (`results/logs/eval_s2_levels.sh`'s table). Real signal, on a
column that can fail.

**What the reflex never does:** abstain. 0 of 5 abstain tasks in each suite,
by construction — it has no branch that declines. Any `correct_abstain`
above zero is the model, all of it.

## 3. `filter_padded`, and its own floor

`harness/filter_check.py` flags a `FILTER` clause when the request names
neither its field nor the value it compares against — the padding the
corpus teaches (`delinquent EQ true AND delinquent EQ false` on a
one-predicate request).

The reflex pads **45.3%** (db) and **70.0%** (demo) of tasks, which is what
a planner that always fills the slot should look like.

The matcher ignores function words, and that was not cosmetic: without a
stopword list, `"the completed status"` read as grounded by a request
saying `"what cards are there"` — because `"the"` is inside `"there"` — and
the demo suite's reflex rate read 35.7% instead of 70.0%. A lenient
substring match needs the stoplist or it launders padding as grounding.

**The column does not read zero on correct programs, and the floor is the
point.** On reference programs, which are correct by construction:

| suite | refs with a `FILTER` | flagged | rate |
|---|---|---|---|
| `e_demo_requests` | 15 | 0 | 0.0% |
| `e_db_requests` | 40 | 5 | 12.5% |
| `curriculum_tasks` | 24 | 4 | 16.7% |

Every one of those nine is semantic inference the wording never spells:
"show me the overdue invoices" filters `NOT paid EQ true`; "whoever has the
most unfinished cards **on the board**" filters `archived EQ false`;
`L9_projects_cleanup`'s "clean up the old projects" is *tagged ambiguous*
precisely because "old" names no cutoff. A model's rate means something
against 0–17%, not against 0.

## 4. Where the corpus teaches it: level 9

`data.gen` now prints this share with every file it writes, beside the
signature-uniqueness line. Generated fresh, 40 rows per level:

| level | what it is | rows filtering on something unnamed |
|---|---|---|
| 2 | filtering, explicit | 5.0% |
| 3 | multiple constraints | 2.5% |
| 9 | **ambiguous scope** | **60.0%** |
| 11 | abstain | 0.0% |

Level 9 is `"Clean up the old projects."`, `"Tidy up the board."`,
`"Deal with the stale open tickets."` — tasks whose whole design is that
the request underspecifies and the reference picks a criterion anyway. It
is doing what it was built to do; the finding is that **it is the one place
the corpus teaches "invent a filter clause", and it teaches it at 60%.**
Any recipe change aimed at the padding reflex (step 2b, step 3) should
start there rather than at levels 2 and 3, which are already clean.

## 5. The answering recipes, and what the instrument caught in them (step 2b)

`sample_lookup` (L21, `data/gen/programs.py`) is the family the corpus had
none of: list → optional single filter → one of `COUNT` / `SORT`+`FIRST` /
`MAP`+`sum`/`avg`/`max` → `RETURN`. Measured on its own rows, 60 per run:

| | before (same 200-row mix, no L21) | with L21 at ~10% |
|---|---|---|
| rows ending in `RETURN` | 11.0% | **19.5%** |
| rows that call a list tool and then do **not** filter | 5.0% | **13.3%** |

On the L21 rows alone: 100% end in `RETURN`, 42% never filter, 60/60
references execute green and 60/60 `return_match`.

Those baselines are not the 7.6% / 2.8% the plan quotes — those were
measured on the 29k-row training corpus, this is a 200-row sample at
different level weights. The *direction and size* of the move is the
claim, not a like-for-like delta against that corpus.

**`report_filter_grounding` caught a real defect in the first version of
the recipe, on its first batch.** The `first` arm sampled a filter clause
and then asked a question that never mentioned it — "What is the oldest
open ticket" against `FILTER priority LT 2` — because that arm's English
is the sort's own superlative phrase and drops clause material. The family
would have taught exactly the padding it exists to cure. The arm now takes
no clause; the share went 23.1% → 5.7% (hand-written worlds) and 0.0%
(themed), inside the 0–17% reference floor above. This is the instrument
paying for itself two hours after it was built.

## 6. The compute arm had almost nothing to fold, and now it does

`sum`/`avg`/`max` need a `LIST INT`, so the arm needs an entity with an
`INT` field. When 2b was built there were almost none. Measured across the
tree:

- `crm`: `ticket.priority`, `invoice.amount`
- `kanban`, `projects`: none
- the 143 generated themes: **none** — every themed entity field was `STR`,
  `BOOL`, `TIME` or `ID` (the same inventory `spec/agent_core.md` §12 cites
  for retiring `CONTAINS`'s list arm)

So the arm fired on the hand-written worlds (12 of 60 rows, 20%) and
essentially not at all on a themed corpus (1 of 60, 1.7%). A themed
training corpus would carry almost no aggregate-answer rows however the
recipe was weighted, and step 6's "totals/averages via compute tools" axis
would read near-empty.

**Closed 2026-09-21 (same day), in worldgen rather than in the recipe.** A
theme may now declare one numeric field — `child.number` with a `field`, a
`noun` (the words a request uses for it), and a `min`/`max` range
(`data/gen/THEME_SCHEMA.md`). The compiler types it `INT`, the state
generator fills every record with a value in range, and the profile carries
the `noun` so the question reads as English rather than as a field name.
All 143 themes were given one, hand-authored per domain: a quantity that
domain's software would really record per record (`chair_minutes` for a
dental treatment plan, `haul_miles` for a trucking haul, `word_count` for a
translation job).

| | before | after |
|---|---|---|
| themed child entities with an `INT` field | 0 / 143 | **143 / 143** |
| aggregate rows among themed L21 rows | 1.7% | **26.3%** (79/300) |
| themed L21 references that execute green | 60/60 | 60/60 |
| `return_match` on themed L21 references | 60/60 | 60/60 |

Two decisions inside that number:

- **The arms are weighted now, not uniform** (`count` 3, `list` 3, `first`
  2, each compute arm 1). Once every theme has a numeric field, a uniform
  pick over six arms makes **half** of this family aggregate questions,
  and the two shapes the family exists for are `count` and `list` — which
  are also the arms that carry the empty filter slot. Measured over 3,000
  in-process draws the mix comes out `count` 28.3% / `list` 26.5% /
  `first` 19.9% / compute 25.2%, against a design of 27.3 / 27.3 / 18.2 /
  27.3, with no failed draws.
- **Nothing sets the field and no filter reads it.** No tool takes it as an
  argument, and the profile exposes no numeric filter kind, so this change
  adds an answerable question and no new action. A numeric `FILTER`
  ("jobs over 5,000 words") would need a new filter kind, its own English,
  and its own grounding check; it is not here.

The English needed one fix the first batch exposed: a `noun` written as a
plural count ("trays in the batch") is ungrammatical in the frame the
question uses ("what is the total *trays in the batch* across the grow
batches"), so every such noun is a singular measure ("tray count"), and the
`largest` arm now says "largest" or "highest" — "the highest chair time in
minutes" reads, "the largest" does not.

## 7. What this does not say

- Nothing about a model. No model was run; `reflex_baseline` is arithmetic
  over the task's own declared symbols.
- Nothing about whether the demos are *fixable* — that is step 2b (answering
  recipes), whose effect is now attributable because these numbers exist
  first.
- `filter_padded` is a heuristic over request wording, not a proof. It is
  recorded, never gated, and no number already in `results/` moves because
  of it.
