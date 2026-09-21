# Grounding is unmeasured: the signature identifies the tool in every suite

**Date:** 2026-09-19. **Instrument:** `harness/signature_uniqueness.py`, which
reads reference programs and task contexts only — no model, no GPU. Every
number below is reproducible in about a minute on a laptop.

`results/R3.md` §3 reported that in `s5_plain` the typed signature identifies
the called tool uniquely in 100% of reference `CALL`s, and used it to argue
that R3's failure was symbol binding rather than English. That was correct.
The other implication was not drawn, and it is the larger one: **a suite in
which the signature is unique cannot distinguish tool grounding from type
inference.** This file measures how far that property spreads. It spreads to
everything except the two suites built with decoys.

---

## 1. The measurement

For each reference `CALL`, is the called tool's signature shared with any other
tool declared in that task? Three columns, each a stricter claim:

- **full** — unique by the whole signature, per-request field symbols included.
- **stripped** — unique by types, arity and effects, field symbols removed.
- **effect** — unique by effect class alone.

| file | tasks | calls | tools/task | full | stripped | effect |
|---|---|---|---|---|---|---|
| `s5_plain` (tiny-model corpus) | 2000 | 3785 | 17.9 | **100.0%** | 94.0% | 19.4% |
| `s5_tasks` (main corpus) | 2000 | 3785 | 17.9 | **100.0%** | 94.0% | 19.4% |
| `s5_crowded` | 2000 | 3831 | 55.7 | **100.0%** | 90.9% | 2.2% |
| `s5_holdout` | 2000 | 3778 | 17.9 | **100.0%** | 93.1% | 19.2% |
| `e_known_plain` | 400 | 755 | 17.9 | **100.0%** | **100.0%** | 17.7% |
| `e_foreign` | 500 | 1186 | 9.0 | **100.0%** | **100.0%** | 28.5% |
| `e_crowded` | 300 | 708 | 55.2 | **100.0%** | **100.0%** | 0.0% |
| `e_crowded_v2` | 300 | 674 | 54.3 | **100.0%** | **100.0%** | 1.8% |
| `e_ood_english` | 200 | 470 | 9.0 | **100.0%** | **100.0%** | 26.6% |
| `r2_holdout` | 100 | 224 | 8.0 | **100.0%** | **100.0%** | 26.8% |
| `e_demo_requests` | 70 | 100 | 11.0 | **100.0%** | **100.0%** | 25.0% |
| `e_db_requests` | 75 | 75 | 13.0 | **100.0%** | **100.0%** | 0.0% |
| `r1_tasks` | 1000 | 2356 | 10.0 | **100.0%** | **100.0%** | 17.9% |
| `curriculum_tasks` | 43 | 93 | 11.1 | **100.0%** | **100.0%** | 19.4% |
| `e_known_decoy` | 400 | 755 | 40.4 | 68.3% | 68.3% | 0.0% |
| `fam_decoy` | 2000 | 3693 | 40.3 | 69.6% | 50.7% | 0.0% |

Three things to read off it.

**Every evaluation suite is 100% unique, and 100% even with field symbols
stripped** — stricter than the training corpus, which sits at 90–94% stripped.
The exams are easier on this axis than the corpus that trains for them.

**Crowding does not touch it.** `e_crowded` puts 55 tools in front of the model
instead of 18 and stays at 100.0%/100.0%. Burying the right tool among foreign
ones adds candidates without adding *collisions*: the foreign tools have
different shapes, so the signature still picks the answer out of 55 as cleanly
as out of 18. Whatever E-crowded's 96.3% measures, it is not description
reading.

**Only the decoy-built suites pose the discrimination.** `harness/decoys.py`
gives a mutating tool two to four siblings with the same signature and a
neighbouring description, so by construction the signature stops being
sufficient: 68.3% and 69.6%.

## 2. What a 100% suite can and cannot show

Stated precisely, because the strong version of this claim is false.

100% unique does **not** mean the task is easy or that the model is cheating.
The model still has to read the request, work out that the job needs
`(ID:customer, STR) -> None [SEND]`, and find the tool with that shape. That is
real work, and it is what `results/R7.md` shows the structural encoder doing at
99.6%.

What it means is narrower and exact: **no task in these suites requires a
description to choose between candidate tools.** A policy of "infer the type
shape from English, then take the unique tool with that shape" is sufficient
everywhere. So no accuracy number from these suites supports a claim about
reading descriptions or grounding in schemas, and a model that reads
descriptions and one that does not are indistinguishable on all of them.

## 3. What this invalidates

- **H4 ("the model can ground unseen tools from schemas rather than memorize
  APIs") is unmeasured.** Not refuted — unmeasured. No suite in the tree that
  has been run against it can separate the hypothesis from its alternative.
- **R5 (`PLAN.md` §7) cannot fail for the right reason as written.** Its pass
  bar is held-out and adversarial-named tools within 5 points of known tools.
  `r2_holdout` is 100.0%/100.0%: a signature matcher scores the same on it as
  on `e_known_plain`, and adversarial names (`foo17`, `operation_93`) cost such
  a model nothing, because it was never using the name. R5 as specified would
  be passed by a model with no grounding at all.
- **`results/R7.md`'s transfer claim is a binding claim, not a grounding
  claim.** "Binding transfers to unseen declarations" is what the numbers
  support. The stronger reading — that the encoder has learned to read schema
  lines — does not follow, and the 99.6% is exactly what a type matcher scores.
- **The pretrained-encoder question is affected in both directions.** R3
  withdrew it because "it solves a problem the corpus does not pose"; R7 §5 put
  it back within reach. The corpus still does not pose that problem, so a
  pretrained encoder cannot be evaluated on these suites either.

`results/R7.md` §4's seed instability is not explained by this file, and should
not be read as if it were. What this file establishes is that the suites cannot
tell the two candidate explanations apart.

## 4. The real tool sets do collide

`harness/signature_uniqueness.py --schema`, over the schemas in `data/schemas/`:

| tool set | form | tools | unique | largest colliding group |
|---|---|---|---|---|
| `coursebuilder_tools` | Agent Core (`params`/`returns`/`effects`) | 36 | **77.8%** | 2 |
| `mobi_frontend_tools` — module-scoped | JSON Schema | 25 | **28.0%** | 7 |
| `mobi_frontend_tools` — unbound chat | JSON Schema | 58 | **34.5%** | 9 |
| `mobi_backend_tools` | JSON Schema | 7 | 100.0% | 1 |

**The coursebuilder row is the comparable one.** It is declared in Agent Core
form, so 77.8% against the corpus's 100% is apples to apples. Its collisions
are the family-B shape exactly: `apply_to_lessons` and
`generate_lesson_content` are both `(ID:module, STR) -> None [WRITE]`;
`get_element_schema` and `search_help` are both `(STR) -> STR [READ]`.

**The mobi frontend rows are a floor, not the figure.** JSON Schema carries
only `string`/`number`/`object`, which is coarser than a typed Agent Core
signature with entity ids, so an Agent Core rendering of the same 25 tools
would land above 28.0%. How far above is unmeasured. What the row does show is
the shape of the problem: seven tools sharing `(string)` —
`delete_module`, `get_element_details`, `delete_element`, `select_element`,
`duplicate_element`, `get_sub_items`, `get_element_schema` — are tools whose
*effects differ wildly* and whose only separator, at that level of typing, is
the description. `mobi_backend_tools` at 100% is 7 tools, too few to read.

This is a difference in kind between what the corpus teaches and what the
product presents, and it is a plausible contributor to the real-request gap
(`route_write_match` 53.75%, `results/S5.md`). **It has not been shown to be
the cause.** Establishing that needs a failure breakdown of the real-session
misses against tool collisions in their contexts, which is not done here.

## 5. The fix, piloted: the rate is not the knob, and READ is the wall

200 tasks per cell, the S5 surface (`--symbols typed --enums --kinds
--domains data/gen/themes`), levels 0/2/3/4/5/11/18, seed 777, 365 reference
`CALL`s. Laptop, about 15 seconds a cell.

| decoys per mutating tool | tools/task | full | types-only |
|---|---|---|---|
| none | 17.8 | 100.0% | 92.3% |
| 1:2 | 30.7 | 59.2% | 46.6% |
| 2:4 | 39.9 | 59.2% | 46.3% |
| 3:6 | 47.4 | 60.0% | 47.1% |
| **1:2, after the fix below** | **30.7** | **46.6%** | **46.6%** |

**The rate is the wrong knob.** Tripling the siblings per tool moves full
uniqueness by 0.8 points — inside noise — while nearly tripling the context.
Whatever the corpus rollout uses, it should use the low rate; the plan's
"sweep the rate later" is answered.

**A defect in the decoy device, found and fixed.** At 1:2 the breakdown by
effect class was WRITE 2.7% unique, DELETE 2.5%, **SEND 97.9%** — and SEND was
2.1% unique with field symbols stripped. So SEND decoys collided in *shape* and
not in the full signature. The cause: `harness/context.py` keyed an unlinked
tool param (a message body, a note — anything with no entity field behind it)
by tool *name*, so a decoy's free-text slot read `S=F6` against the original's
`S=F5` and the two lines stayed distinguishable without reading a word of
either description. A decoy now carries `decoy_of` and takes its original's
symbol for unlinked params. SEND goes 97.9% → 2.1% unique and the file goes
59.2% → 46.6%, with full and types-only now equal, which is the sign that no
symbol identity is left to leak. Non-decoy generation is byte-for-byte
unchanged (verified on the same seed); decoyed corpora regenerate differently,
so `fam_decoy.jsonl` and `e_known_decoy.jsonl` as stored predate the fix.

**The residual is READ, and it is 45% of all calls.** With every mutating call
colliding, the file sits at 46.6% because READ tools are never decoyed:
`harness/decoys.py` targets `MUTATING = {WRITE, DELETE, SEND, PAY}` only. 165
of 365 reference calls are READ and all 165 are signature-unique at every rate.
So the ≤50% ceiling is now reachable, but the list-and-filter half of every
program still requires no description at all — and READ collisions are exactly
what the real frontend schema has most of (seven tools sharing `(string)`,
§4).

Extending decoys to READ is not a one-line change, and the reason is worth
recording rather than discovering later. A decoy READ tool has to return
something of the right type; an empty `LIST OBJ:x` is the natural noop. But an
empty list is also the legitimate trigger for the check-then-decline recipe, so
a model that picks a decoy list tool, sees nothing and emits
`ABORT NOT_FOUND` would be **scored as a correct abstain** while having chosen
the wrong tool. That needs a scoring decision before the generator change, not
after.

*(Resolved the following day — §8. The scoring decision was smaller than it
looked: the call log always distinguished the two runs and was simply never
read.)*

## 6. What follows

The corpus and the suites are the defect, so they are what changes. The plan is
`.claude/plans/binding-transfer-signature-shortcut.md`; `PLAN.md` §14 carries
the amendment.

Landed with this file: the statistic is printed for every file `data.gen`
writes, a 100% file warns, `--require-collisions PCT` makes it fatal, and the
`decoy_of` fix takes a decoyed corpus to 46.6%. Still open: decoys in the main
generation path and the exams (a corpus regen), the READ extension and the
abstain-scoring decision it needs (§5), and R5's restatement against a suite
that can fail it, which lands with the retrain that motivates it.

*(The READ extension and its scoring decision landed 2026-09-20 — §8. The
corpus regen and R5's restatement are still open.)*

## 7. Reproducing

```bash
python -m harness.signature_uniqueness --limit 2000 \
  data/s5_plain.jsonl data/s5_tasks.jsonl data/s5_crowded.jsonl \
  data/s5_holdout.jsonl data/fam_decoy.jsonl \
  data/holdout/e_known_plain.jsonl data/holdout/e_foreign.jsonl \
  data/holdout/e_crowded.jsonl data/holdout/e_crowded_v2.jsonl \
  data/holdout/e_ood_english.jsonl data/holdout/r2_holdout.jsonl \
  data/holdout/e_known_decoy.jsonl data/holdout/e_demo_requests.jsonl \
  data/holdout/e_db_requests.jsonl data/r1_tasks.jsonl data/curriculum_tasks.jsonl

python -m harness.signature_uniqueness --schema --examples data/schemas/*.json
```

## 8. The wall came down, and the chance line was the real defect (2026-09-20)

§5 left READ at 100% unique and blamed a scoring question: a decoy READ's
natural noop return is an empty list, an empty list is also the legitimate
trigger for check-then-decline, so a model that picks a decoy list tool, sees
nothing and emits `ABORT NOT_FOUND` scores a correct abstain having chosen the
wrong tool.

**The scoring question was smaller than it looked.** The outcome cannot
separate those two runs, but the call log always could, and it was simply never
read. A decoy's name has been in `task["provenance"]["decoys"]` since the
family was built. Landed:

- `decoy_called` in `harness/metrics.py` — did this run call a signature
  sibling that is never in a reference program? Recorded, not gated, following
  `abort_referent_match` directly above it, so every number already in
  `results/` keeps its meaning. `None` on an undecoyed task, so it cannot
  dilute a mean over a suite it does not apply to. Report
  `correct_abstain and not decoy_called` where grounding is the point.
- The same tally in `models/tiny/evaluate.py`, so a pod run prints it.

**A second defect, found while wiring the first.** `compare_calls` scores
`same_tool` against `chance_tool = 1/len(tools)` — a uniform pick over every
declared tool. That is the wrong null for a decoyed suite. A model that infers
the type shape and reads no description has already narrowed to the tools
sharing that signature, and it picks inside *that* group. `chance_tool_sig`
(`models/tiny/diagnose.py`) is uniform inside the collision group, and the two
lines are far apart:

| holdout variant | tools/task | full unique | `chance_tool` | `chance_tool_sig` |
|---|---|---|---|---|
| no decoys | 17.9 | 100.0% | 5.7% | **100.0%** |
| decoys, mutating only | 30.7 | 53.1% | 3.3% | **73.4%** |
| decoys, READ + EXTERNAL too | 41.5 | **2.6%** | 2.4% | **43.5%** |

Read the mutating-only row and the cost of the omission is plain: a model
scoring 80% `same_tool` on it reads as twenty-four times chance, when a pure
shape matcher that never opens a description already scores **73.4%**. Almost
all of that is READ — 48.9% of calls, every one in a collision group of one,
contributing 48.9 points on its own.

**The extension.** `harness/decoys.py` now banks by return shape as well as
effect, because "list every X" and "fetch the one X" are not interchangeable
descriptions and a description that does not fit its own return type is a
tell. `_bank_key` gives `READ:LIST`, `READ:OBJ`, `READ:STR` and
`EXTERNAL:STR` beside the four mutating banks. What a decoy hands back:

  `LIST OBJ:x`  `[]`, so a FOREACH over it does nothing
  `STR`         `""`, which fails the state check against real generated text
  `OBJ:x`, `ID:x`  a `NOT_FOUND` ToolError

The last one is the case worth stating. A getter decoy that hands back the
real record **works**, and a decoy that works is the type shortcut with extra
steps — so the mutating branch that returns "I changed nothing, here is your
record" is explicitly not taken for a READ. There is no empty record of the
declared type and `null` is not one either, so it raises.

Measured on the shipped corpus, `data/s5_holdout_decoy.jsonl` — 8,000 tasks
over the 42 reserved worlds, 15,189 reference calls, 41.5 tools per task. The
call count is identical to the plain holdout's, which is the check that the
decoys are purely additive: same seeds, same worlds, same states, same
requests, same reference programs, more tools declared. Every effect class now
collides, at a mean collision group of 2.4 tools. (Breakdown over the first
2,000 tasks, 3,778 calls.)

| effect | calls | share | unique before | unique after |
|---|---|---|---|---|
| READ | 1840 | 48.7% | 100.0% | **0.5%** |
| WRITE | 846 | 22.4% | 4.8% | 6.1% |
| SEND | 720 | 19.1% | 0.0% | 1.2% |
| DELETE | 264 | 7.0% | 4.0% | 3.0% |
| EXTERNAL | 108 | 2.9% | 100.0% | **10.2%** |

EXTERNAL does not go to zero: a name collision with an existing tool drops a
decoy silently (`taken`), and with only four entries in the `EXTERNAL:STR`
bank there is less room to recover than the READ banks have. 2.9% of calls at
10.2% unique is 0.3 points of the total, so it is recorded rather than chased.

200/200 reference programs execute green with no decoy called on the pilot,
which is the check that matters: the family is only worth anything if the
*reference* never needs the discrimination it poses.

**Compaction does not collapse the distinction.** `models/tiny/prep.py`
trims every schema line's description to its first sentence and then to 60
characters before the model sees it, and the description is the only thing
separating a decoy from its original — so a bank entry whose first 60
characters match its neighbour's would make the task unanswerable rather than
hard. Measured over the pilot's 378 reference calls: **0** called tools share
a full line with another tool, and **0** share a compacted one.

**What this does not settle.** The training corpus still has no decoys —
`s5_plain` is undecoyed and so is every cache built from it. A pod run against
the decoyed holdout therefore measures a model that never trained on
discrimination being asked to discriminate. That is a real measurement and it
is the one the grounding question asks, but it is not R8's measurement, and
the two should not be read as a single trend line. The corpus regen (§6)
remains the deeper fix.
