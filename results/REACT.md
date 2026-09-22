# REACT — making a failure worth learning from (step 3a)

**Date:** 2026-09-22 · **Plan:** `.claude/plans/agent-loop-and-ir-review.md`
step 3a · **Instrument:** `data/gen/episodes.py`'s `report_reaction`, printed
with every episode file · **Cost:** none, no model was run

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

## 4. What this does not say

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
