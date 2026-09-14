# Task families — teaching the agent more than one shape of job

Plan: `.claude/plans/archive/task-families.md`. Corpus: `results/logs/gen_families.sh`.
Exams: `results/logs/eval_families.sh`. Nothing here has been retrained on
yet; this documents what was built, what it was checked against, and the
results that came back before any training: the decoy gap in §2, the
per-turn baselines in §3 and the scheduling probe in §4. Both of those corrected something I had
written down wrong first, which is noted where it happened.

## 1. The finding this rests on

The 0.8B generalizes across domains and not across task families.
E-foreign, 500 tasks in 143 themes it never trained on: 95.0% (`R6.md` §0.1).
E-rpg, one task family it never trained on: 0/6, 0/6, 0/6 for S2R, S3 and S4
(`RPG.md`). Every row of the 62,841-row corpus is the same family — find
records, filter them, act on them — so the dungeon asks for something the
corpus never showed, and the model writes a list-filter-act template with no
list to filter.

The fix is not a bigger corpus of the same shape. It is the rest of the map,
with the same held-out discipline the domains already have: several
instances in training, at least one reserved, scored on its own exam.

## 2. What was built

| family | what it teaches | where |
|---|---|---|
| A observe → act | partial view, small action set, choose by description, adapt next turn | `runtime/worlds/{warehouse,elevator,cards,house}.py`, engines in `runtime/engines/`, oracles in `harness/oracles/` |
| B tool selection by description | pick among tools with the same signature whose only difference is the description | `harness/decoys.py`, `data.gen --decoys MIN:MAX` |
| C UI navigation and form filling | family A on a screen; the goal is a form's end state | `runtime/worlds/pages.py`, `runtime/engines/page.js`, `harness/oracles/pages.py` |
| D ask, then act | stop when a required value is missing, then act on the answer | `data/gen/askact.py` |
| E recovery by error code | the continuation the code you were just shown calls for | `data/gen/recovery.py` |
| G scheduling | *not built* — it is an IR question, and §4 answers it | `runtime/worlds/scheduling.py`, `harness/schedule_probe.py` |

`harness/decision.py` is the registry that pairs each decision world with its
oracle and says which are reserved. `harness/rpg_suite.py` now takes
`--world` and plays any of them; the default is still `rpg`, so every stored
E-rpg command means what it always did.

### The four decision worlds vary surface and topology on purpose

A second grid dungeon with orcs instead of goblins would test the domain,
which we already know transfers. So:

| world | topology | rendering | actions | what it adds |
|---|---|---|---|---|
| warehouse_robot | grid | prose ("the pad is 3 bays east 4 bays north") | drive, lift, set_down, charge | no glyphs anywhere; a battery that makes the shortest route to the tote often the wrong first move |
| elevator | 1-D | a table of floors and calls | go_to, open_doors, hold | **waiting is a legal, sometimes right action**, and every argument is a number off a table rather than an entity id |
| cards | none | four numbers | hit, stand, double_down | the smallest possible instance of "read the situation" |
| house | room graph | prose, ways out by name | go, take, use, open | **held out** — no grid at all, so the family score is not grid → grid |

The dungeon stays held out, unchanged. The house joins it, and so does the
coursebuilder-shaped page app (`app_coursebuilder`), built to the shape of
the real builder's tools (`data/schemas/mobi_frontend_tools.json`: add an
element with a type, props and a position). `data/holdout/reserved.json`
carries all of them.

### The episodes teach off-path states, not the golden path

This is the part that matters more than the worlds. An oracle plays
perfectly and a model is off that path from its first wrong move; the RPG
showed it never comes back. So `data/gen/episodes.py` does not record a
clean play-through. At a quarter of turns it executes something other than
the oracle's move — a legal move the oracle would not have chosen, or (30%
of those) an outright illegal one, so the next observation opens with a real
`Last turn: failed: cannot drive north: a rack is in the way` — and then
asks the oracle what it would do *from the state that produced*. Every turn
is labelled from the state the episode is actually in.

The illegal detours are what put family E's problem inside family A: R4
found the react rule inert (the model does not read what happened last
turn), and this is the corpus that shows it what reading it looks like.

### Decoys are description-only siblings, and the A/B is paired

`--decoys 2:4` gives every mutating tool in a theme two to four siblings
with its exact signature and a neighbouring description, banked by effect so
that a copy operation never turns up carrying `[SEND]`. 15% are named
`foo17` / `operation_93` (R5's own), so the name cannot carry the choice at
all. Decoys are `noop` impls: a model that picks one is scored wrong on the
task's state rather than corrupting the world some other way.

The decoy RNG is seeded separately from the draw, so `--decoys` produces the
*same* world, state, request and program as the run without it. On a paired
8-task check the request and state matched 8/8 while the tool table went
from 18 tools to 42. That is what makes "E-known minus E-known-with-decoys"
a number rather than two unrelated scores.

### The number, 2026-09-12: 9.25 points, and the model never notices

S4 typed on both halves of the pair, 400 tasks each, same seed and same
programs (`results/logs/famB_{plain,decoy}_s4.*`):

| | goal | compiles |
|---|---|---|
| e_known_plain | **97.75%** | 99.5% |
| e_known_decoy | **88.5%** | 99.0% |

39 tasks lost, 2 gained. **37 of the 39 compiled and executed cleanly** -
the model called a sibling that does nothing and reported success. Compile
rate barely moves, so this is not a syntax or symbol-table effect: given
two tools with the same signature and a neighbouring description, the model
picks by something other than the description about one time in eleven, and
nothing downstream catches it. The losses sit at L12 (9), L8 (6), L15 (5),
L9 and L4 (4 each) - the levels whose action is a send or an update, which
is where the sibling banks are densest.

That is the one-bit answer family B was built for, and it says the family
earns its 6,000 rows.


## 3. A metric that scores decisions, not episodes — and the first thing it said

`0/6` cannot separate "cannot decide" from "cannot plan". `harness/rpg_suite.py`
now asks the oracle, at every model turn, what it would do from the *same*
state, and scores the model's turn against it.

**The plan's version of this metric is gameable, and the first baseline run
proved it.** §4 of the plan asked for "whether the model's first call names
the oracle's tool, and whether its argument matches". S3 scores 98% on that
on the dungeon while winning nothing, because it opens with the oracle's move
and then keeps going — it calls `move` north, south, east and west every
single turn, spending the whole budget to end up where it started. So the
report carries three numbers and the one to read is the last:

```
$ python -m harness.rpg_suite --world rpg       --model baselines/qwen/models/qwen3.5-0.8b-s3-q8.gguf --episodes 3
  won            0/3 (0%)
  funnel         key 0/3  door 0/3  exit 0/3
  per-turn       first tool 59/60 (98%)   first tool+arg 59/60 (98%)   whole turn 0/60 (0%)
  calls          192 (59 invalid)
  turns          60 (1 did not compile, 0 abstained, 15 hit the action budget)
```

That is a genuinely different diagnosis from `0/6`. **On the dungeon, S3 is
not failing to decide the first move — it decides it correctly 59 times out
of 60 and then cannot stop.** A third of its calls are invalid and a quarter
of its turns burn the whole action budget.

The same checkpoint on the held-out prose house says something else again:

```
  won            0/3 (0%)
  funnel         lamp_lit 0/3  way_open 0/3  arrived 0/3
  per-turn       first tool 33/60 (55%)   first tool+arg 0/60 (0%)   whole turn 0/60 (0%)
  calls          36 (36 invalid)
  turns          60 (24 did not compile, 0 abstained, 0 hit the action budget)
```

Here it *does* fail to decide: the right tool a bit over half the time, the
right argument never, 24 of 60 turns not compiling at all, and every single
call that ran illegal. Grid or no grid, a room graph named in prose is
further outside what it knows than the dungeon is.

### The row S5 has to beat: S4 typed, 2026-09-12

The same probe on the checkpoint the retrain starts from, each world on the
surface that checkpoint was trained on (`results/logs/perturn.sh`, 3
episodes, 120 turns a world, the Vulkan server):

| | first tool | first tool+arg | whole turn | turns that did not compile |
|---|---|---|---|---|
| dungeon (rpg) | **1%** | 1% | 0% | 109/120 |
| house | 49% | 0% | 0% | 61/120 |
| coursebuilder app | 93% | 46% | **12%** | 0/120 |

S3's 98% on the dungeon is not a bar S4 fell under by being worse at
deciding - S4 mostly does not produce a legal turn there. It writes
`CALL T2 S0 -> r0` then `CALL T3 r0.F5 -> r1`, reading a field off something
that has none (`UNKNOWN_FIELD r0 F5`), which is the same unbound-register
habit `results/RPG.md` found after the NULL fix closed the easy way out.

The app is the first non-zero whole-turn score any checkpoint has scored on
any of these worlds: 14 of 120 turns exactly right, every turn compiling.
It is also the world closest to the corpus's own shape - named tools over a
form, no spatial state - which is the point family C was built to make.
Three baselines, then, and they are three different problems: produce a
legal turn at all on the dungeon, bind an argument on the house, and stop
over-acting on the app.

Reading the oracle against itself gives 100% on all three numbers for every
world, which is the check that they measure agreement and not something else
(`tests/test_decision_worlds.py`, alongside a planner that appends one extra
move to the oracle's own turn and is caught only by the whole-turn score).

## 4. Family G: one gap, and it is narrower than it first looked

The plan reserved scheduling as an IR question before a data question. The
probe (`python -m harness.schedule_probe`) hand-writes the canonical asks
against a small held-out world and runs each through the real pipeline.
**Two of the three fit under spec 0.5.0; the third runs and does the wrong
thing.**

The first pass of this section said the field comparison was "not
expressible at all". That was wrong, and writing out the `FOREACH` forms is
what showed it: both of the then-failing asks could be *performed*, and
what neither could do was produce the matching set as a value. The gap was
one thing — `FILTER`'s clause was `field cmp operand` while an `IF`'s
condition is `operand cmp operand`, and nothing in the spec said why the
predicate was the weaker of the two. Half closed on 2026-09-11, the rest on
2026-09-12; all three asks now fit.

- *"Book the cheapest free room that seats eight"* — **fits**. One list and
  a predicate that fits `FILTER`'s clause form, so `FILTER` + `SORT` +
  `FIRST` is the whole job. L4's shape with a two-clause predicate.
- *"Which shifts still need cover?"* — **fits, since 0.5.0**. `FILTER r0
  F_covered LT F_needs -> r1` then `RETURN r1`. Before 0.5.0 the same
  comparison was legal only as `IF r1.F_covered LT r1.F_needs` inside a
  `FOREACH`, which can act on each under-covered shift and never count, sort
  or return them — and this ask is a question, so the value *is* the answer.
  The change is spec §3 (`clause` takes a field on the right), §4's
  `FILTER` row and §12; ~40 lines across parser, typecheck, compile and the
  two grammars; no new instruction, no new type, nothing in the sandbox.
- *"Take the earliest hour free in both calendars"* — **fits, since
  0.6.0**. `MAP` projects the other calendar to a list of times and `IN`
  takes that list on the right of a `FILTER` clause, so the intersection is
  a register: `FILTER r4 F_start IN r2 -> r5`, then `SORT` + `FIRST` + the
  call. Before that the only membership form was `CONTAINS` with the list on
  the left, which an `IF` admits and a `FILTER` clause cannot, so the
  intersection got walked and never bound — "the earliest" was unreachable
  and the program took both mutually free hours.

**On §12's tests.** The widening names no composite, so the four tests
apply by analogy. Tests 1, 2 and 4 hold (the class was unserved because it
was inexpressible; the workaround was not long but incapable; nothing new
in the type system). Test 3 — "a capable model fails to find the composite,
measured" — was not run, and on reflection should not be: there was no long
form for the 27B to find or miss, so a run could only restate the probe.
The model question that *is* open is adoption — whether a model taught the
form emits it for "which are understaffed" — and that is answered by S5's
corpus and eval, not by a pre-change measurement.

**The membership half landed as `IN`** (spec 0.6.0, plan §3b), and with it
the overlap question §5 of the plan raised: `IN` and `CONTAINS` doing the
same job with the operands swapped is family B's confusion by construction,
so `CONTAINS`'s list arm went rather than gaining a mirror. `CONTAINS` is
substring on `STR` again. Nothing depended on the arm — no entity field in
any world is `LIST`-typed (141 theme worlds built through
`data.gen.domains`, plus the 14 registered ones) and `CONTAINS` appears in
0 of 56,000 S4c references, 0 of 56,000 S4 and 0 of the family corpus. `python -m harness.schedule_probe` now prints 3/3.

**`IN` has corpus rows; the field comparison cannot.** An IR form with no
rows teaching it is a form the model never emits (`MOST` only landed because
R4 shipped recipes with it), so L19 of the S5 corpus is membership: every
parent that has, or has no, child matching a predicate, as a set. `MAP` the
matching children to their parent ids, one `FILTER ... IN` over the parent
list, then act. The `FOREACH`-over-children form it replaces acts once per
child, so a parent with three matching children gets messaged three times.

The field comparison has no recipe, and the reason is the worlds, not the
generator. Every theme entity `data.gen.domains` builds is `STR`, `BOOL`,
`TIME` or `ID` - **no theme world has a single `INT` or `FLOAT` field** - and
the two times a child carries are one event time and one creation time, so
"departed before it was planned" is not a request anyone makes. 77 entities
have two `TIME` fields and no pair among them is comparable in English.
Teaching `FILTER r0 F1 LT F2` needs a second comparable field on the child,
a due/actual pair or a numeric capacity/used pair, which is a theme-surface
change touching all 141 worlds and every level's clause sampling. Not folded
into S5 silently: owner's call.

## 5. What the corpus looks like

18,324 rows in the classic arm, 18,267 in the typed one:

    bash results/logs/gen_families.sh classic
    bash results/logs/gen_families.sh typed

R6 §0 showed a typed-trained model
scored on classic prompts looks broadly broken, so the two arms never share
a file — including their exams.

| | rows (classic) | |
|---|---|---|
| A decision episodes | 4,366 | warehouse_robot 1,809 · cards 1,364 · elevator 1,193 |
| C page episodes | 1,958 | checkout 731 · settings 648 · ticket 579 |
| B decoyed tasks | 6,000 | levels 0–18, every mutating tool given 2–4 siblings |
| D ask / act / complete | 3,000 | 1,000 draws × 3 rows |
| E recovery | 3,000 | 1,000 each of NOT_FOUND, PERMISSION_DENIED, RATE_LIMITED |

1,560 turns were played off the oracle's path, and 379 turns open with a
real `Last turn: failed: …` — those are the ones that teach the thing R4
found missing.

**Watch the abstain rate on the mix.** Family D is an ask row and family E a
third `PERMISSION_DENIED`, so the family corpus is 14.6% abort against S4c's
10.7%. Mixed one-for-one that comes to 11.7%, a point above where S4c sits.
That is deliberate and it is also the direction S2 went wrong in
(over-abstention, `S2.md`), which is why family D writes a third row — the
same job stated in full, needing no question — for every ask/act pair. Left
as pairs alone the family would have been 50% abort and the mix 12.3%.

Every reference in every family executes: `python -m harness.run --tasks
<file> --domains data/gen/themes` replays 18,324/18,324 and 18,267/18,267 —
both arms green — which is the check
that a row's `expected_state` is a derivation rather than an assertion.
`harness/run.py` gained `--domains` so that check can be run on a themed
corpus from the CLI at all.

### Two bugs the build found, worth writing down

**A theme was already shadowing a world.** `data/gen/themes/gen_warehouse.json`
is a domain called `warehouse`, and `register_domains` overwrites
`WORLDS[name]`, so the decision world of that name silently became a CRUD
theme the moment any generator registered themes — the corpus kept saying
`world: warehouse` while `warehouse` meant something else. The decision world
is now `warehouse_robot`, and `register_theme` raises rather than shadow a
world whose rules live in an engine. (`coursebuilder` shadows the same way
and always has, but that one is deliberate: the theme pack is how the
generator makes coursebuilder-shaped tasks.)

**Sampled draws have to be winnable.** Three separate ones slipped through
and were caught by running the oracle over them rather than by reading the
code: a page job that named one required field and left another empty, so
the form could never save; a warehouse draw whose battery could not reach
the charging pad; and a card shoe short enough at a big enough stake to go
broke under perfect play. An unwinnable draw is a training row that teaches a
dead end, so `tests/test_decision_worlds.py` now plays the oracle straight
through six sampled draws per world and asserts every one finishes.

## 6. What to run next

The plan's order still holds, with G removed from it:

1. ~~**Family B alone, on the current checkpoint.**~~ Done: 9.25 points,
   §2. Description-reading does not survive siblings, and the failure is
   silent, so the family stays the size it was built.
2. **The per-turn baseline on S4.** S3's is in §3; S4's is the row the
   retrain actually has to beat, and it is one `eval_families.sh` away.
3. ~~**One S5 retrain**~~ Done 2026-09-13: `results/S5.md`. The decoy gap
   closed (9.25 -> 0), the abstain control went 83.75 -> 99.2, E-known
   cleared its bar at 97.2, `IN` is adopted 30/30 - and family E did not
   transfer at all on two of its three error codes, which is the finding
   that matters. The original wording follows.

   One S5 retrain with the family corpus mixed into S4's, scored on the
   six suites in R6 §0.1 (must not regress), the dungeon, the house, the
   coursebuilder app, and the decoy gap. Each family has its own exam, so a
   regression is attributable — which is what the classic-control problem in
   R6 §0 cost us last time. The corpus is `results/logs/gen_s5.sh`: S4's
   mix plus the families, the two R6 §0.2 recipe fixes, and L19.
4. **Whether theme entities get a second comparable field** - without one
   the 0.5.0 field-vs-field clause has no corpus rows and no way to get
   any (section 4). `IN` needed no such change and is in S5 as L19.
