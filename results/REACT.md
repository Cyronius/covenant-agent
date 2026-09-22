# REACT — making the observation matter (step 3)

**Date:** 2026-09-22 · **Plan:** `.claude/plans/agent-loop-and-ir-review.md`
steps 3a (§1-3, the episode corpus) and 3b (§4, the task corpus) ·
**Instruments:** `report_reaction` in `data/gen/episodes.py` and
`report_segments` in `data/gen/__main__.py`, both printed with every file they
describe · **Cost:** none, no model was run

In the dungeon the 0.8B is told `Last turn: failed:` every turn and repeats
the failed move 103 times out of 105 (`.claude/plans/general-agent-plan.md`
Tier 1 item 1). The cause is in the generator, not the model. A failed move
does not change the board, so the oracle's next label is simply the plan it
already had — and because the failure was a *random* illegal move, that plan
has no reason to be a response to it. The corpus never showed the model a
turn where reading the failure changed what to do.

## 1. What the turn after a failure looks like, measured

`report_reaction` reads a written episode file and, for every turn whose
observation opens with `Last turn: failed: ...`, compares the label with the
move that produced the failure:

- **repeated** — the label *is* the move that just failed.
- **same object, another way** — the label acts on something the failed move
  named, with a different verb.
- **answers it** — that, or the refusal names the verb the label uses ("the
  north way is shut - open it first", answered by `CALL @open`). The verb has
  to differ from the one that failed, or a world whose refusal echoes the
  verb it refused ("cannot drive to shelf 0") would score every retry as an
  answer. That correction matters: before it, the arm below read 51.7%
  instead of 20.7%.

Two seeds, 12 episodes per world, every trainable decision and page world:

| | failures | caused by an aimed move | **answers the failure** |
|---|---|---|---|
| before (`--predictable 0`) | 59 | 0 | **6 (10.2%)** |
| after (the new default) | 55 | 17 (30.9%) | **22 (40.0%)** |

**Nothing repeats.** 0 of 114 post-failure labels is the move that just
failed, in either arm — so the corpus was never teaching repetition
directly, which is what `general-agent-plan.md`'s "zero of 373 rows share a
token with the failure" says from the other side. What it was teaching is
**indifference**: the label ignored the failure 90% of the time.

**Every aimed failure is answered: 17 of 17.** A random illegal failure is
answered 7 times in 59 (11.9%). That gap is the whole result.

## 2. What an aimed move is

Three ways to be wrong on purpose, measured separately because they are not
worth the same:

| branch | what it is | answered |
|---|---|---|
| `same_obj` | the right object, a verb that does not apply to it yet — walk into the way that is shut | **10/10** |
| `on_board` | a verb error on something the board really has — click an element on a screen that is not open | **7/7** |
| `same_verb` | the right verb aimed at the wrong object — drive to the shelf that is out of reach | **0/15** |

`same_verb` is the one intuition suggests first and it teaches nothing: the
refusal is about the target, the next label is the oracle's original target,
and the failure carries no information the model needed. It is now tried
last and, with verification on, never survives.

## 3. Why the aimed failures are answered 17 times out of 17

Because the generator plays the candidate before committing to it. For each
aimed candidate it deep-copies the state, runs the move through the real
engine, checks that it actually failed, asks the oracle what it would do
from there, and keeps the candidate only if that move answers the refusal —
a different verb, named by the reason or aimed at the same object. Up to
eight candidates, then it gives up and plays a random illegal move rather
than an aim that would not teach anything.

That is why the default is `--predictable 1.0`: trying always costs a few
sandbox runs on 7.5% of turns (offpath 0.25 × illegal 0.3), and when no aim
would be answered the fallback is exactly the old behaviour. `--predictable
0` reproduces the pre-2026-09-22 corpora.

Cost: 12 episodes × 7 worlds is 72 seconds with verification against about
55 without.

## 4. Observe, then decide (step 3b)

The other half of "make the observation matter" is on the task side, and it
had the same shape of defect. One recipe produced `PAUSE` — L10, a
filter-then-act program cut in half — so what the corpus taught was *pause
when the request says report back*, not *pause because you cannot know yet*.
Everywhere a decision genuinely depends on data, the reference decided
anyway, in one shot, with `IF`.

Three levels have a real data-dependent decision, and each now has a
two-segment form whose second half could not have been written before the
first ran:

| level | one-shot form | two-segment form |
|---|---|---|
| 5, branch | `GET` the field, `IF r1 EQ v` then act else act | `GET` the field, `PAUSE`; the second segment is **only the branch the observed value calls for** |
| 7, notify-if-any | `PARALLEL` read, `FILTER`, `IF NOT EMPTY` notify | `FILTER`, `PAUSE`; the message when something matched, a bare `STOP` when nothing did |
| 11, check-then-decline | list, `FILTER`, `IF EMPTY` abort else act | list, `FILTER`, `PAUSE`; `ABORT NOT_FOUND` or `FIRST` + act, whichever the rows call for |

Measured on the same 240-row mix over twelve levels, generated twice:

| | rows that pause and decide |
|---|---|
| before | **20/240 (8.3%)** — all of them L10, the one recipe that always pauses |
| after | **31/240 (12.9%)** — L5 6/20, L7 2/20, L11 3/20, L10 20/20 |

Every reference still replays: **240/240 `goal_success`**, and the two-segment
rows report exactly 2 segments through the harness.

**The pause must not be a tell.** If every paused first segment ended in
`ABORT NOT_FOUND`, a model could answer "not found" from the shape of the
program it had just written rather than from the rows it got back — the same
context-not-words shortcut `results/S2.md` found behind the demo's
over-abstention. So a third of L11's rows now name a record that **is**
there, in both the one-shot and the two-segment form, and the two-segment
ones split roughly evenly between a decline and an action. The acting rows
carry no `abort` tag, so `gen_one` marks them `expected_status: ok`.

Ten tests in `tests/test_two_segment.py` pin this, including that the branch
surviving in L5's second segment is the one the record's own field value
calls for, and that both endings appear in L11's paused rows.

**Not done, and it is the reason this is 12.9% rather than the third of the
corpus `reactive-execution.md` §6 sized:** the error levels (8 and 9). Their
reactive form is not a `PAUSE` at all — it needs the harness's
error-feedback path, the one `results/R4.md` found never fired in 150 runs,
so it is a design question about that path rather than another recipe. The
levels that shouldn't pause still don't: a filter-then-act row (2, 3) and an
argmax row (4) have nothing to observe, and teaching them to pause would be
teaching the gratuitous pause R4 measured at −4.

## 5. What this does not say

- **No model was run.** This is a corpus property. The model-side gate —
  `repeat_after_failure` on the dungeon and house exams, 98% and 34% today —
  needs a GPU run and rides with step 6's one re-baseline. That metric does
  not exist yet either.
- The 40% is not a ceiling anyone should read as a target. It is bounded by
  how often a verified aim exists at all (30.9% of failures here), which
  depends on the world: a world whose refusals are instructive ("open it
  first") produces them and blackjack produces none, because nothing in it
  can fail.
- `answers the failure` is a wording heuristic over the refusal text plus an
  argument comparison. It is recorded, never gated, and it undercounts: a
  page app's answer acts on the screen rather than on the element that
  failed, and only the verb-name half sees it.
- Nothing about levels 5–11's two-segment references (step 3b), which is the
  other half of "make the observation matter" and is not built.
