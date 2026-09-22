# SECOND TURN — the tiny model can take one now

**Date:** 2026-09-22 · **Plan:** `.claude/plans/agent-loop-and-ir-review.md`
step 4 · **Cost:** none, CPU only · **Gate 1: met. Gate 2 not met at CPU scale
and not yet run properly — §3 prices it at ~$0.37.**

The loop already existed in three places (`agent-loop-and-ir-review.md` §1a),
and for the 0.8B and the 27B the owner's claim was already false — they do get
their registers back. For the model that ships in the browser it was exactly
true, and not for a subtle reason:

- `models/tiny/corpus.py` admitted **single-segment tasks only**, so a task
  that pauses was not in the training data at all.
- `models/tiny/prep.py` exited on a `REGISTERS:` section rather than parse it.
- Region A — the encoder's context — had **nowhere to put** what a register
  holds.

So the model had seen one kind of row its whole life: a whole program, written
from nothing.

## 1. What changed

**A paused task is now one training row per segment.** The continuation's
input is the same context plus a register region, and its target is only the
part that starts there. The first segment's row is unchanged.

**A bound register is rendered, not dumped.** `harness/context.py`'s
`render_register`: the symbol, its type, and a bounded view of the value —
`r1 LIST OBJ:card n=9 [card_0 card_1 card_2 ...]`, `r3 INT 4`,
`r2 OBJ:card card_3`. Nine kanban cards in full are 3.6 KB against a 64-token
line cap, and `run_a.build_prompt` caps the 27B's own JSON dump at 1,500
characters for the same reason.

**The registers come from running the program, not from guessing.**
`corpus._replay` plays the reference through the real sandbox — the same path
`harness/run.py` takes — and hands back the context and register values each
segment actually starts from. A task whose reference will not replay produces
**no rows at all** rather than rows with an invented register state.

**In the encoder the region sits between the constants and the request, with
its own tag.** The pointer bank does not move: a canvas slot still points at a
tool, a field or a constant, and `r1` was always an output row
(`canvas.Layout`'s 32 register rows). What was missing was any way to *read*
what `r1` holds.

**`play.py` drives the loop.** Sample a segment, hand it to `run_task`, let
the sandbox execute it, re-serialize the context with the registers it bound,
encode that with `prep.encode_one`, sample again. The plan specified this as a
mode of `evaluate.py`; it is its own file because `evaluate.py` scores cached
rows one at a time and a second turn cannot come from a cache — the registers
depend on what the model itself just did.

## 2. Gate 1: does the continuation work at all

The plan's first gate is the three L10 curriculum tasks passing end to end,
"does the continuation work, not accuracy" — they were filtered out of this
model's training and evaluation entirely.

A 0.9M-parameter model (d=128, 2+2 layers), trained on CPU for 400 epochs over
the 48-task curriculum, which is **memorization, not generalization** — the
point is the mechanism:

| | result |
|---|---|
| paused tasks that got a second segment at all | **3/3** |
| paused tasks that reached the goal | **2/3** |
| whole curriculum, goal | 38/43 (88.4%) |
| reference round trip through the cache, continuation rows included | **15/15** |

The one failure is worth more than the two passes. On `L10_kanban_pause_archive`
the model wrote the **first segment again** after the pause — four times, until
the harness's planner cap — instead of the continuation:

```
seg0: CALL T10 -> r0 ; FILTER r0 F3 LT NOW AND F4 EQ C0 -> r1 ; PAUSE
seg1: (the same)
seg2: (the same)
seg3: (the same)
```

That is the failure mode the register region exists to cure, still visible in
a model that has seen four continuation rows in its life. The other two tasks
continue correctly: `FOREACH r1 -> r2 ; CALL T11 r2 ; STOP` against a register
region that says `r1` holds four cards.

## 3. Gate 2 is not run, and it needs a pod

The plan's second gate is: on the same 42-world holdout R9 scored (control arm
70.4% plain, 46.0% decoyed), goal success on two-segment tasks within five
points of one-segment tasks of the same level. That is a trained-model
measurement on a real corpus.

**It costs about a dollar, not the retrain.** R9's own receipts: 6.5
GPU-hours for both arms x three seeds, 12 epochs, batch 64, on a RunPod
community 4090 at $0.34/hr — **$2.21 for six runs**, so one arm-seed is ~1.08
hours and **~$0.37** (double it if it falls through to a secure pod). An
arm-seed is 4,812 steps over ~25k rows at the real model size: not a degraded
run, but exactly the run R9 read 70.4% / 46.0% off.

| | what it buys | cost |
|---|---|---|
| CPU | does the mechanism hold on unseen worlds at all (§2 and below) | free |
| one arm-seed + the register region | **gate 2**, plus the real VRAM figure for the new region | **~$0.37** |
| R9's full shape (2 arms x 3 seeds) | only if the region interacts with the arm comparison | ~$2.21 |
| step 6's retrain over the 62.8k-row family corpus | everything deferred into one rebuild | the ~$16-32 estimate |

So the $16-32 figure the plan quotes is the *last* row. Gate 2 does not have
to wait for it, and the region's VRAM cost should be measured before it does:
R9 found the current model at 18.9 GB of 24 at batch 64, and `pod.md` has been
wrong in the optimistic direction twice.

Before any of that there is ~20 minutes of free laptop work: the corpus has to
be rebuilt with continuation rows (`prep.py`'s replay costs 2m17 per 3k rows),
and the episode shards regenerated if step 3a's aimed detours are wanted in
the same measurement.

## 4. The free tier, run: unseen worlds, real loop

Before spending the $0.37, the same question on CPU. A 3,000-row themed
corpus generated for this (26.2% of its rows two-segment: L5 156/437, L7
64/327, L10 437/437, L11 128/436), six worlds carved out, a 0.9M model, and
`play.py` driving the real loop over the **170 tasks of the six worlds the
model never saw** — 55 of them paused.

Scored at epoch 3 of 14, which is why the absolute numbers are low (R9's full
run reads 70.4% on unseen worlds; this is a fifth the width and a quarter of
the schedule):

| | goal |
|---|---|
| one-segment tasks | 32/115 = **27.8%** |
| two-segment tasks | 13/55 = **23.6%** |
| *conditioned on a compilable first program* | one-segment 32/41 = **78.0%**, two-segment 10/21 = **47.6%** |

**The mechanism holds on worlds the model has never seen.** 21 of 55 paused
tasks got a second segment, and the continuations are real programs written
against the register region, not memorized text — this one is from a world
carved out of training:

```
seg0: CALL T6 -> r0 ; FILTER r0 F23 EQ C0 -> r1 ; PAUSE
seg1: ABORT NOT_FOUND C0
```

It looked, saw nothing matched, and declined.

**The gate-2 comparison does not pass at this scale, and the 4.2-point
unconditional gap should not be quoted as if it did.** Both arms are near the
floor, which flatters the difference; conditioned on getting a compilable
first program the two-segment tasks are ~30 points worse. Per level, the gap
is concentrated exactly where the decision is real: L11's one-shot form (`IF
EMPTY` → abort or act, writable blind) scores 92.9%, its two-segment form
(look, then decide) 40.0%. L5 is level: 12.5% one-shot against 14.3%
two-segment.

That is the honest read of a weak model, and it is the argument for the
$0.37: a real-size, full-schedule arm-seed is what tells you whether
two-segment tasks are harder *for a trained planner* or only for this one.

## 5. What this does not say

- **Nothing about accuracy.** A 0.9M model memorizing 41 examples says the
  data path and the architecture work. It says nothing about whether a trained
  planner uses the register region well, which is gate 2.
- The register rendering is a **choice**, not a measurement: ids and a count
  for a list, the scalar for a scalar. Whether a continuation needs more than
  the ids (a field value, say) is open, and the cap is where to look first if
  gate 2 comes in low.
- `--max-reg 8` covers every reference in the tree today; prep refuses a task
  with more rather than truncating.
- Nothing about the browser. Latency doubles per pause, which the plan sizes
  at a second or two for the 0.8B and less for this model, unmeasured here.
