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

The comparison arms are scripted in `results/logs/eval_rpg.sh` (oracle, both
tuned checkpoints, the untuned 0.8B and the untuned 2B, same seeds and cap)
but were not executed: three other evaluations were occupying this machine's
CPU, and adding a multi-hour sweep would have slowed them and made the
latency column meaningless. Run it when the box is free:

```
bash results/logs/eval_rpg.sh 3 20
```

The arm that matters most is the untuned 0.8B on its own chat template: it
separates "the tuning taught this template" from "the base model is equally
lost". A frontier planner would separate it further; it was left out because
only a third-party API key was present on this machine and sending the
prompts there was not part of what was agreed.

Latency is not reported as a headline for the same reason: the 29.5 s median
generation time in the run above was measured against three concurrent
llama.cpp evaluations and says nothing useful.

## Reproducing

```
python -m pytest tests/test_rpg.py            # rules, turn budget, perception
python -m harness.rpg_suite --planner oracle --episodes 3   # 3/3 won
python -m harness.rpg_suite --model baselines/qwen/models/qwen3.5-0.8b-s2r-q8.gguf \
    --episodes 3 --max-turns 20 --out results/logs/<tag>_e_rpg.jsonl
python -m harness.rpg_suite --report results/logs/<tag>_e_rpg.jsonl
```

Each row carries the git SHA, model, template, grammar condition, seed and
action cap, plus a per-turn log with the observation, the raw program, the
call log and the finish reason — enough to read any episode back move by
move. The same game is playable in a browser (`client/rpg-ui`), which runs
the identical rules through the identical sandbox.
