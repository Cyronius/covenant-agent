# E-rpg — the planner plays a held-out grid dungeon

Plan: `.claude/plans/rpg-demo-app.md`. World: `runtime/worlds/rpg.py`, rules
in `runtime/engines/rpg.js`, runner `harness/rpg_suite.py`, demo
`client/rpg-ui`. Reserved in `data/holdout/reserved.json` — never trained on.

## What this measures, and what it does not

Every other world in this repo is CRUD over records. This one is a
turn-based dungeon: each turn the model is shown a 5×5 window of its
surroundings plus HP, inventory, what it saw earlier and what happened last
turn, and it emits a program using five tools — `move(direction)`,
`attack(enemy)`, `pick_up(item)`, `use_item(item)`, `interact(door)`. The
program is compiled and executed, enemies answer, and the next turn starts
from the result.

**It is a decision probe, not an IR-competence probe.** The programs a good
player writes here are straight-line `CALL Tn Ck` lines — none of the
`FILTER`/`FOREACH`/`SORT` machinery the corpus is built around. There is no
read tool, so nothing to filter. What is being asked is narrower and more
basic: *given what you can see, pick the right tool and give it the right
argument.*

Scoring is a terminal predicate plus a progress funnel (key → door → stairs),
not `goal_success` state equality: a game has many winning play-throughs.

**The scenario is winnable from the observation alone.** The scripted player
(`harness/rpg_oracle.py`) wins every episode in 12 turns with zero illegal
actions, and it can only ever name constants that the same observation
exposed. So a failure below is a failure to decide, not a prompt that hides
the answer.

## Result: the tuned planner does not play

3 episodes, seeds 0–2, one map, 20-turn cap, 3 actions per turn, grammar on,
greedy decoding, `qwen3.5-0.8b-s2r-q8.gguf` (the newest tuned checkpoint,
unpruned).

| | oracle | S2R 0.8B |
|---|---|---|
| won | 3/3 | 0/3 |
| died | 0/3 | 1/3 |
| reached the key | 3/3 | 0/3 |
| opened the door | 3/3 | 0/3 |
| turns to win (p50) | 12 | — |
| turns that compiled | 36/36 | 23/55 |
| illegal actions | 0 of 78 calls | 21 of 65 calls |

It never gets far enough to make a spatial mistake. **Thirty-two of 55 turns
fail in the typechecker**, before anything executes, and the reason is the
same every time (all 32 are `TYPE_ERROR`): it picked a tool whose argument
type does not match the constant it passed. Of the 23 turns that did run, a
third of the calls were illegal moves into walls, and 14 turns spent their
whole action budget going nowhere. It reached neither the key nor the door
in any episode, and was killed by a goblin in one.

**What it actually writes.** Turn after turn, whatever the situation, it
emits one shape:

```
CALL T3 C0
CALL T3 C1
CALL T3 C2
CALL T3 C3
CALL T2 C4
STOP
```

That is: *enumerate every constant in the list, one call each.* 41 of 55
programs have exactly five `CALL` lines and 10 more have four — the count
tracks the number of constants on offer, not the situation. The tool symbol
varies between turns (`T0`, `T2`, `T3`, `T4` all appear in that slot across
episodes) but the pattern does not. Since `C0`–`C3` are the four direction
strings, most of those picks do not typecheck.

When the enumeration happens to land on `move`, the program is legal and the
model walks north, south, east and west — back where it started, having spent
the turn and let the goblins take a step. Every call it ever made was `move`
(65 of 65): the other four tools only ever appeared with arguments that
failed to typecheck, so `attack`, `pick_up`, `use_item` and `interact` were
never once executed. It also never used `IF`, `FOREACH`, `TRY` or `PARALLEL`
— no attempt at conditional play, just the template.

So it is not reading the tool descriptions. It is reproducing a template
whose real content is "there are five constants, so make five calls". That
is the same failure `results/S2.md` records on the real-session suite —
"reproduces the S1 recipe skeleton whatever the request says, picks tools by
position habit" — reproduced here in a domain built to expose it, where the
correct program is as simple as programs get.

The honest reading: **on this evidence the current 0.8B planner cannot make
sequential decisions in an unfamiliar domain, and the blocker is not
perception or planning but basic schema-grounded tool selection.** That
matters for what to test in a candidate replacement: the first thing to
measure in a bigger Qwen, an LFM2.5-instruct, or anything else is whether it
picks the right tool for a described situation at all — not whether it can
reason about a grid.

### A spot check on the untuned base: it fails differently

Two turns of `Qwen3.5-0.8B-Q8_0` on its own chat template (a code-path check,
not a scored arm — the real arm is in `eval_rpg.sh`) fail for a different
reason, which is worth knowing before choosing a replacement model. It does
not enumerate constants. It copies the shape of the worked example in the
`SYSTEM` prompt — `CALL` a list tool, `FILTER`, then `FOREACH` — into a world
that has no list tool and nothing to filter, and then repeats the loop body
until it hits the 250-token cap (`finish_reason: length` on both turns,
against `stop` on all 55 tuned-model turns):

```
CALL T0 -> r0
FILTER r0 F1 EQ C4 -> r1
FOREACH r1 -> r2
  CALL T0 r2.F1
  CALL T0 r2.F1
  ... (to the token limit)
```

So the tuned model has learned to stop cleanly and to respect the constant
table, and has not learned to read a tool description; the base model has
neither. Both are anchored to the one worked example they are shown, which
is the obvious next lever to test: the `SYSTEM` prompt's only demonstration
is a kanban list-filter-loop, and every world since has been the same shape.

## Re-run 2026-09-10: S3 and S4, and nobody plays yet

Six episodes (seeds 0-5), same map, 20-turn cap, 3 actions a turn, grammar
on, greedy. Each checkpoint on the surface it was trained on: S4 with
`--symbols typed --enums --kinds`, S2R and S3 classic. Unpruned GGUFs, as
below. Generated through LM Studio's bundled Vulkan `llama-server` against
this box's AMD iGPU (`results/logs/rpg_gpu.sh`) - the in-process CPU path
takes ~23 s a turn, this takes ~3 s, and the two return the same programs
token for token (checked on the S4 arm's first three turns).

| | oracle | S4 typed | S3 | S2R |
|---|---|---|---|---|
| won | 6/6 | 0/6 | 0/6 | 0/6 |
| died | 0/6 | **6/6** | 0/6 | 1/6 |
| reached the key | 6/6 | 0/6 | 0/6 | 0/6 |
| turns played | 72 | 74 | 120 | 120 |
| turns that compiled | 72/72 | 41/74 | **116/120** | 32/120 |
| calls (failed) | 156 (0) | 66 (35) | 378 (116) | 74 (27) |
| calls with a NULL arg | 0 | **25** | 0 | 0 |
| tools ever executed | move, attack, pick_up, interact | move, interact, use_item | move | move |
| CALLs per turn (mode) | - | 3 | 4 | 5 |

**Nothing reaches the key.** Three checkpoints, three failure modes, one
outcome: the funnel's first gate is still shut, and only the scripted player
ever opens a door. Whatever the corpus taught between S2R and S4, it did not
teach sequential decisions in this world.

**The failure moved, twice.** S2R could not compile: 88 of its 120 turns died
in the typechecker. S3 compiles almost everything (116 of 120) and still gets
nowhere, because every one of its 378 calls is `move` and it emits four or
five of them a turn - north, south, east and west, back where it started.
That is the constant-enumeration template of the original run, reproduced on
a newer checkpoint with a cleaner typechecker record. S4 typed writes a
shorter template: three calls, one per constant *letter* on offer, 65 of its
74 turns. The letters reorganised the habit instead of breaking it.

**S4 is the first checkpoint to call anything but `move`** - `interact` 23
times and `use_item` twice. That is the typed slots doing what they were
designed to do, steering the tool choice by argument type. It bought
nothing: every one of those 25 calls passed NULL and failed, while all 41
`move` calls carried a real direction. Not one non-`move` call in the run
had an argument, which is the next paragraph.

**S4 dies in every episode, and that is not a worse planner.** S3 spends each
turn moving in all four directions, so it never leaves the safe corner and
never dies; S4 spends its one real action moving consistently in one
direction, travels, and meets the goblins. Both are lost. Only one of them
is lost somewhere dangerous.

### The NULL hole: a legal call that can never work

25 of S4's 66 calls passed NULL for a required argument, and all of them
failed at runtime. This is not the model inventing syntax. `interact` takes
one `ID:door`; on a turn with no door in sight, no door constant exists, no
register is bound, and the typed operand class for that slot reduces to

```
op-ID-door ::= reg | reg "." cf-ID-door | "NULL"
```

so NULL is the only operand the grammar can emit there. The typechecker
agrees: `core/typecheck.py:44` treats NULL as compatible with every type, on
both surfaces, so `CALL <interact> NULL` compiles clean and fails in the
sandbox as `NOT_FOUND`. Verified directly, classic and typed alike.

The same shape is already on the record elsewhere: `results/R6.md` §0.2's
L18 misses in the abstain control are `CALL T11 I0 NULL` - a note id the
model could only have got by listing and filtering first. A tool whose
required slot has no admissible operand should be unreachable, not reachable
with a placeholder.

**Fixed the same day** (`.claude/plans/null-required-slot.md`): a literal
NULL in a required slot is now `MISSING_ARG` in the typechecker, and typed
required slots are built without NULL in the grammar. Optional slots keep
it, and so do comparisons and `SET`, where testing against null is the
point. Where the corpus teaches the right form the fix pays: L18 goes 16/30
to 17/30 with all 14 NULL programs gone and nothing broken (`R6.md` §0.4).

On this suite it changes where the turn dies and nothing else. Re-run, same
six seeds:

| S4 typed, 6 episodes | before | after |
|---|---|---|
| won / reached the key | 0/6, 0/6 | 0/6, 0/6 |
| died | 6/6 | 0/6 |
| turns that compiled | 41/74 | 13/120 |
| calls (failed) | 66 (35) | 16 (11) |
| required-slot NULL calls | 25 | **0** |

The NULLs are gone - the 11 that remain are `use_item`'s optional target
slot, which is legal. In their place the model writes an unbound register
field (`CALL T r0.F5`), so 107 of 120 turns now fail in the typechecker,
`UNBOUND r0` 77 times. It stops dying only because it barely moves: 5 `move`
calls against 41. E-rpg runs with no repair round, so here the fix trades
silent waste for visible waste. That is the right trade and it is not a
planner improvement: this suite's problem was never NULL, it is the one
template.

### What this does not settle

The untuned arms (`Qwen3.5-0.8B`, `Qwen3.5-2B`, both on their own chat
template) are still unrun - the HTTP path here builds only the hand-rolled
qwen markup, and mixing templates through it would measure the wrapper. They
remain the arm that separates "the tuning taught this template" from "a
0.8B base is equally lost", and `results/logs/eval_rpg.sh` still scripts
them for the in-process path.

Also unchanged from the original run: the FIELDS block is mostly noise here,
the `SYSTEM` prompt's only worked example is still a kanban list-filter-loop,
and no arm has ever used `IF`, `FOREACH`, `TRY` or `PARALLEL` - S4 included.
The one worked example remains the obvious lever nobody has pulled.

## What was held fixed

Nothing was tuned to make this easier or harder:

- Same `SYSTEM` prompt as every other suite (`baselines/qwen/run_a.py`),
  including its kanban worked example. No RPG-specific coaching.
- Same GBNF grammar. Same greedy decoding, `max_tokens` 250.
- Same `serialize_context` prompt format the model was trained on. The turn's
  prompt is ~1,300 input tokens: 5 tools, 23 fields, 4–12 constants,
  and a ~250-token observation.

Two properties of the setup worth naming, neither adjusted for this run:

- **The FIELDS block is mostly noise here** — `build_context` emits every
  entity field, so the model sees `enemy.x`, `item.held` and 21 others it can
  never use, because no tool returns an object to read. That is ~300 tokens
  of distractor. It is also exactly the condition `E-crowded` measures, so it
  was left alone.
- **Pruned-vocabulary checkpoints were deliberately not run.** Their
  vocabulary was trimmed to the SFT corpus plus a general-English floor
  (`results/S2.md` §B4), so map glyphs and words like "goblin" fall back to
  segmentations they never saw; scoring them here would measure the trim.

## Not yet run

The untuned arms (`Qwen3.5-0.8B` and `Qwen3.5-2B` on their own chat
template) are still outstanding - see the re-run section above. A frontier
planner would separate the picture further; it was left out because only a
third-party API key was present on this machine and sending the prompts
there was not part of what was agreed.

Latency is not a headline in either run. The original 29.5 s median was
measured against three concurrent llama.cpp evaluations; the re-run's ~4 s
mean is an iGPU number and says nothing about a deployment target.

## Reproducing

```
python -m pytest tests/test_rpg.py            # rules, turn budget, perception
python -m harness.rpg_suite --planner oracle --episodes 3   # 3/3 won
python -m harness.rpg_suite --model baselines/qwen/models/qwen3.5-0.8b-s2r-q8.gguf \
    --episodes 3 --max-turns 20 --out results/logs/<tag>_e_rpg.jsonl
python -m harness.rpg_suite --report results/logs/<tag>_e_rpg.jsonl

# any non-Qwen GGUF: --template chat applies the model's own chat template
python -m harness.rpg_suite --model <some.gguf> --template chat --episodes 3

# a checkpoint trained on the spec 0.4.0 surface has to be scored on it
python -m harness.rpg_suite --model .../qwen3.5-0.8b-s4-q8-fixed.gguf \
    --symbols typed --enums --kinds --episodes 6

# GPU on this box: LM Studio's bundled Vulkan server, ~3 s a turn (~23 s in
# process, which has no CUDA card and a CPU-only llama-cpp-python)
llama-server.exe -m <gguf> -c 4096 -ngl 99 --port 8078
python -m harness.rpg_suite --server http://127.0.0.1:8078 --model <gguf> ...
bash results/logs/rpg_gpu.sh 6 20          # oracle + S4 + S3 + S2R
python results/logs/rpg_breakdown.py results/logs/*_e_rpg.jsonl
```

The S4 GGUFs both needed `baselines/qwen/fix_gguf_layer_arrays.py` before
they would load at all (`results/R6.md` §3); the unpruned one was fixed on
2026-09-10 for this run and is `qwen3.5-0.8b-s4-q8-fixed.gguf`.

Each row carries the git SHA, model, template, grammar condition, seed and
action cap, plus a per-turn log with the observation, the raw program, the
call log and the finish reason — enough to read any episode back move by
move. The same game is playable in a browser (`client/rpg-ui`), which runs
the identical rules through the identical sandbox.
