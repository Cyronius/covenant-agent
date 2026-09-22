# CENSUS — what real requests ask for that the language cannot say

**Date:** 2026-09-21 · **Plan:** `.claude/plans/agent-loop-and-ir-review.md`
step 2c · **Instrument:** `harness/expressibility_census.py` · **Cost:** none,
no model was run

Step 2c exists because the admission rule (`spec/agent_core.md` §11) asks
four things of a candidate instruction and only one of them — is it
expressible today — can be answered at a desk. `CONTAINS`'s list arm is the
reminder of what skipping the other three costs: admitted on intuition,
retired unused. The candidates are the ones already on record in
`results/R1.md`'s holding pen.

**This is the offline half of the census.** Two of the instrument's three
passes need no GPU and are reported here. The third — the 27B writing its
best program for each real request, so the failures can be labelled — has
not been run; it needs a rented pod and the owner has not been asked for
this specific spend. §6 has the recipe.

## 1. What was counted, and how to read it

| pass | what it reads | what it means |
|---|---|---|
| `patterns` | the wording of 1,544 real mobi session requests and 6,893 imported human-phrased ones | a **lower bound** per candidate: wording that only that candidate answers. Paraphrases are missed. |
| `workarounds` | our own reference programs | whether anything is actually **paying** the design tax the candidate would remove |

Every probe is deliberately narrow, and every match in this note was read by
hand — the per-candidate precision below is that reading, not an estimate.
The `label` pass is what turns a lower bound into a demand number, because a
model that cannot say something says something else, and that is visible
where wording is not.

Sources: `data/holdout/e_real_sessions.jsonl` (234 reviewed, committed) and
`data/real_sessions/b2_pool.jsonl` (1,310, scrubbed, gitignored — raw
sessions carry PII); `data/open_pairs/{glaive_12k,hermes_full,toolace_4k}`
(6,893, the three sources `data/gen/open_pool.py` counts).

## 2. Real product requests (n=1,544)

70 of the 1,544 carry the product's own continuity line ("Earlier modules in
this run updated 6 titles. Apply the same intent and tone to this module."),
which reads exactly like a human follow-up. The instrument counts it
separately rather than dropping it — it is real input the agent gets, just
not a person typing.

| candidate | rows | share | typed only | genuine, by hand |
|---|---|---|---|---|
| `take` | 5 | 0.3% | 5 (0.3%) | **4** |
| `unique` | 0 | 0.0% | 0 | 0 |
| `group_join` | 2 | 0.1% | 2 (0.1%) | **0** |
| `predicate_group` | 2 | 0.1% | 2 (0.1%) | **0** |
| `followup` | 93 | 6.0% | 23 (1.6%) | **~20** |
| `aggregate_tool` | 9 | 0.6% | 9 (0.6%) | **0** |

The four genuine `take` requests are all list prefixes over data the agent
assembles itself: *"Duplicate the first 2 elements in this lesson on every
lesson in this course"*, the same with 3, *"there is nothing on the back of
the last 2 cards"*, *"Give me the top 10 issues that operators see with
[vendor] equipment"*. The fifth match is lesson prose ("the six most common
hazards"), not a request.

Both `group_join` matches are `FOREACH`, not tables ("throughout each
lesson's content"). Both `predicate_group` matches are "except where" inside
a long instruction, not a negated conjunction. The nine `aggregate_tool`
matches are pricing questions about the product ("How much would LEAD cost
for my locations?") — not one is an aggregate over records.

The 23 typed `followup` rows are the opposite: nearly all genuine, and they
are the same two shapes over and over. **A correction of the previous
turn** — "make it cartoon-style instead of realistic", "change the audience
to parents instead of students", "please update this image to say 14 days
instead of 21 days" (13 rows carry *instead*) — and **a deictic reference to
something the previous turn produced** — "add an image element to the right
of this one", "can we make this one bullet points", "Let's do this one" (6
rows). That is step 5's demand (new English mid-task) and the
`AMBIGUOUS`/`NEEDS_INFO` case the resumable `ABORT` is for, in the same
place.

## 3. Imported requests (n=6,893)

| candidate | rows | share | what the hand pass found |
|---|---|---|---|
| `take` | 105 | 1.5% | in a random sample of 26: **0** post-hoc list prefixes |
| `unique` | 3 | 0.0% | 0 ("two distinct tasks") |
| `group_join` | 18 | 0.3% | **~10** genuine |
| `predicate_group` | 0 | 0.0% | — |
| `followup` | 24 | 0.3% | mostly within-request ("the same customer's savings account") |
| `aggregate_tool` | 498 | 7.2% | scalar arithmetic, not list aggregates |

**The `take` finding is the one that matters, and it inverts the candidate.**
Of 26 sampled matches, 24 are a *tool argument*: "top 30 GitHub
repositories", "the first 15 webcams", "the last 5 matches for season
12345", "the first four pages of the client list", "limited to the last 10".
The API being called takes the count; nothing lists a set and then keeps a
prefix of it. The remaining two are false positives ("the first 10 terms" of
a Fibonacci sequence, "page 5 of the PDF"). So on this corpus the demand
`TAKE` would serve is served by the tool, and `TAKE`'s own demand is at most
a handful of rows.

**The 7.2% aggregate share is a calculator, not a `sum`.** "It was priced at
$100 but there's a 20% discount, how much will it cost", "splitting a bill,
the total is $150 and we are 5 people". These are arithmetic over two
constants, which the spec deliberately keeps out of the language
(`spec/agent_core.md` §11's admission test, and §12's tool-first rule) — so
if this demand is ever served it is served by a calculator tool, not by
`sum`/`avg` over a field and not by an opcode. 1g's compute tools answer a
different question than this corpus asks.

`group_join`'s ~10 genuine rows are the shape Tier C parked: "total metrics
and the detailed statistics per function", "total spending per customer",
"the number of eco-friendly hotels for each location", "total size for each
drive".

## 4. Who is paying the design tax (`workarounds`)

| file | segments | `group_join` | `take` |
|---|---|---|---|
| `data/curriculum_tasks.jsonl` | 46 | 2 (4.3%) | 0 |
| `data/fam_tasks.jsonl` (every 9th of 18,444) | 2,086 | 36 (1.7%) | 0 |

The join workaround is real and in use — iterate the outer list, filter the
inner one by the loop variable, act:

```
CALL T17 -> r0
CALL T9 -> r1
LET r0 -> r2
FOREACH r2 -> r3
  FILTER r1 F17 EQ r3.F13 AND F7 LT NOW -> r4
  FOREACH r4 -> r5
    CALL T16 r5 -> r6
STOP
```

Nothing anywhere hand-writes a list prefix (two `SELECT`s off one sorted
register): 0 of 2,132 segments. Consistent with the corpus having no
generated request that phrases a top-N.

## 5. What this says about each candidate

- **`TAKE`** — the only Tier A item with any real-request demand at all, and
  it is **4 rows in 1,544 (0.26%)**. On the imported corpus the apparent
  1.5% is the tool's own limit argument. Nothing in the tree works around
  its absence. On this evidence it does not clear the admission rule's
  first test; the `label` pass could still change that, since a model asked
  for "the three most overdue" has to do *something*, and what it does is
  not visible in wording.
- **`UNIQUE`** — **zero** genuine matches in 8,437 requests. The weakest
  candidate on the list, as R1 already suspected.
- **`GROUP`/`JOIN`** — ~10 genuine requests in the imported corpus (0.15%),
  and the three-line workaround is used in 1.7% of training segments, so
  something is paying. Still parked: the blocker is a pair type, and the
  workaround is not failing.
- **`predicate_group`** — **zero** genuine matches. Its case was always the
  design-tax ladder rather than demand, and the ladder rung is still unrun.
- **New English mid-task (step 5) and the resumable `ABORT`** — the one
  shape with unambiguous demand: **23 typed real requests (1.6%)** that only
  make sense against the previous turn, plus the 70 the product itself
  injects. This is the strongest census result in the note, and it is
  evidence for a step the plan already has rather than for a new opcode.
- **Scalar arithmetic as a tool** — 7.2% of imported requests. Not an
  opcode, not `sum` over a field: a calculator. Worth a decision, not a
  grammar change.

## 6. The pass that is not run, and what it costs

```
python -m data.gen ...                       # or reuse the census sources
python baselines/qwen/run_a.py --model <27B gguf> --tasks <census tasks> \
    --think 512 --out results/logs/census_27b.jsonl
python -m harness.expressibility_census label \
    --run results/logs/census_27b.jsonl --tasks <census tasks>
```

The `label` pass classifies every row of a finished run as `expressed` /
`worked_around` / `abort_unsupported` / `did_not_compile` and joins the
non-expressed ones with the wording labels above. It calls no model itself,
so the pod is needed only for the middle command: the plan sizes it at one
pod hour (~$1-3 at 4090 rates, per R9's $3.18 for a larger run).

**Not run, on purpose.** The owner authorized "one or two pod hours" for 2c
in principle during the 2026-09-21 decision round; that is not a go-ahead
for a specific spend, and the offline half was worth reporting first
precisely because it changes what the model pass is for: with `UNIQUE` at
zero and `TAKE` at four rows, the interesting question the pod would answer
is no longer "how much demand" but "what does the model do instead when the
language cannot say it" — which is the same question step 3 is about.

## 7. What this does not say

- Nothing about a model. No model was run. Every number here is wording or
  reference-program structure.
- The pattern pass is a lower bound and its probes are narrow by design. A
  request that asks for a top-N without a number word ("just the worst
  offenders") is invisible to it.
- The real-request sources are one product's sessions (course authoring), so
  the absence of `group_join` demand there is that domain's absence, not
  every domain's. The imported corpus is where analyst-shaped asks live, and
  that is where `group_join` does show up.
- `aggregate_tool` is counted for sizing answering recipes, not as a
  candidate: 1g already made those tools.
