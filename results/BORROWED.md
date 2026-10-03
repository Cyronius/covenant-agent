# Borrowed worlds: what was built, and the bar before any retrain

**Date:** 2026-09-29. **Plan:** `.claude/plans/borrowed-worlds.md` (steps 1-5
done; step 6, the retrain, ran 2026-09-30: `results/R24.md`).
**Spend:** $0.

## What was built

| world | kind | borrowed from | trainable rows | held-out exam |
|---|---|---|---|---|
| `workshop` | crafting (gather, craft by recipe, stations) | TextCraft / Crafter mechanics, our own items | 916 turns (`data/wsb_episodes.jsonl`) | `boatyard` tree: 267 turns, `data/holdout/e_brief_boatyard.jsonl` |
| `rooms` | BabyAI-style rooms, doors, keys, missions | BabyAI mission grammar, Minigrid rules | 933 turns (`data/rmb_episodes.jsonl`) | " after you" missions: 202 turns, `data/holdout/e_brief_rooms_after.jsonl` |
| `service_retail`, `service_airline` | customer service under a written policy | tau2-bench train split | 400 + 400 rows (`data/service_*.jsonl`), 37-40% ABORT on a named policy rule | `service_telecom`: 93 rows, `data/holdout/e_service_telecom.jsonl` |
| family A + C, re-rendered | warehouse, elevator, cards, three page apps | (ours) | 6,385 turns (`data/famb_episodes.jsonl`) | `house`: 445 turns, `data/holdout/e_brief_house.jsonl` |

Every row is labelled by our own oracle (or, for the service worlds, tau2's
expected writes re-derived against our world) and executed in our sandbox.
Every request is a one-line brief within the tiny planner's 128-token budget
(longest: 128 in retail, 110 in family A, 100 in workshop, 94 in rooms), with
the situation in the constants' descriptions. Signature-unique reference
calls: 0% in workshop and rooms (every tool shares its argument type with a
sibling), 3.6% retail and 0% airline with decoys. The canary check passes on
every file. Sources and attribution: `data/borrowed/SOURCES.md`.

Wording banks for the second batch (household, pages, files, trip planning,
distractor tools) are in `data/borrowed/` from ALFWorld, WebArena, OSWorld,
Mind2Web (train split), WebShop, TravelPlanner, TheAgentCompany and BFCL.
Nothing is built from them yet.

## The bar: `tiny:clt_RD` on every new kind

Live episodes (`harness.rpg_suite --planner tiny --exits`, 6 episodes x 20
turns) and task files (`play.py`, backoff 3):

| world | compiled | won / goal | oracle |
|---|---|---|---|
| warehouse_robot, elevator, cards, 3 page apps | 0/120 each | 0/6 each | - |
| house (held out) | 0/120 | 0/6 | - |
| workshop | 100/120 | 0/6 | 6/6 |
| boatyard (held out) | 97/120 | 0/6 | 6/6 |
| rooms | 120/120 | 0/6 | 6/6 |
| rooms_after (held out) | 0/120 | 0/6 | 6/6 |
| service_retail (150 rows) | 0/150 | 0/150 | every reference replays |
| service_airline (150 rows) | 0/150 | 0/150 | every reference replays |
| service_telecom (held out, 93) | 0/93 | 0/93 | every reference replays |

Where it compiles nothing, it writes the record worlds' list-filter-loop
shape - `CALL T4 -> r0`, a `FILTER` with conditions it cannot parse, a
`FOREACH` - the same template the dungeon runs showed (`results/RPG.md` §"Tiny planner"). It compiles in
workshop and rooms, whose tools all take one argument of a shared type and
whose constants spell the situation out, but it wins nothing there either.
So the bar is 0 everywhere, and any retrain that moves it is visible.

## Found on the way

- **The service contexts do not fit the tiny layout.** Up to 51 fields and
  35 constants against the clt layout's 28 and 20; `play.py` crashed on every
  row until it got `--max-field` (added, same reasoning as `--max-const`:
  pointer slots are this task's own line vectors). A retrain cache sizes its
  layout from its corpus, so the mixes below carry the wider layout.
- **Holding out a tool is not a transfer test.** The rooms world first held
  out put-next-to missions, which left `put_next_to` a tool no training row
  ever calls. It now holds out the " after you" connective instead - the
  sentence names the steps in the reverse of the order they happen - so the
  exam tests reading, with every tool trained.

## Step 6, prepared

`results/logs/mix_brw.py` writes three 30,000-row mixes of clt_train with the
new rows at 10.0%, 20.0% and 30.1% (`data/brw{10,20,30}_train.jsonl`), and
`results/logs/gen_brw.sh` preps their caches with gen_clt.sh's flags and the
same regression exam. All three caches are built
(`models/tiny/data_cache_brw{10,20,30}`, reader tables included): layout 51
fields and 29-35 constants, no request or line over its budget (longest
request 128/128, longest line 78/112, longest program 53/64 slots).
Training and scoring are the pod step.

The pod ran 2026-09-30; results in `results/R24.md`.
