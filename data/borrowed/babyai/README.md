# BabyAI mission grammar (borrowed)

Used by `runtime/worlds/rooms.py` (the `rooms` and `rooms_after` decision
worlds). Plan: `.claude/plans/borrowed-worlds.md`, "3. Grid rooms".

## Sources

| Source | URL | Licence | Commit |
|---|---|---|---|
| BabyAI | https://github.com/mila-iqia/babyai | BSD-3-Clause | `65fb0cb6f816532a65014bf034a758244e2d5ae7` (2023-03-06) |
| Minigrid | https://github.com/Farama-Foundation/Minigrid | Apache-2.0 | `8ea099e114b7d6465afabc76957e3d689663534c` |

Both licences allow reuse with attribution; no upstream code is copied, only
the vocabulary, the sentence templates and the rules below, re-implemented.

## What was taken (`grammar.json`)

- **Mission sentences**: `go to X`, `pick up X`, `open X`, `put X next to Y`
  (BabyAI `babyai/levels/verifier.py:249, 288, 319, 367`).
- **Connectives**: `A, then B`, `A after you B`, `A and B`
  (`verifier.py:440, 481, 527`), with BabyAI's non-strict ordering: doing
  the second part first is not a failure, it simply does not count until the
  first part is done (`verifier.py:433-545`, `strict=False`).
- **Object descriptions**: `[colour] type`, colour optional
  (`levelgen.py:367`); article `the` when exactly one object in the level
  matches, else `a` (`verifier.py:88-92`).
- **Which objects each mission may name** (`levelgen.py:411-423`): go to any
  object or door; pick up a ball, box or key; open a door; put a ball, box or
  key next to anything.
- **Vocabulary**: the six colours (Minigrid `minigrid/core/constants.py:8-17`),
  the object types (`verifier.py:7, 10`).
- **"Next to"**: one step apart horizontally or vertically, never diagonally
  (`verifier.py:28-38`).
- **World rules**: a locked door opens only while carrying the key of its
  colour, and the key stays in hand (Minigrid
  `minigrid/core/world_object.py:184-194`); one carried object at a time
  (`minigrid/minigrid_env.py:561-573`).
- **Oracle idea**: breadth-first search to the target, and when the target
  has not been seen, go through the nearest unopened door (BabyAI
  `babyai/bot.py:452-500`, `ExploreSubgoal`). Re-implemented in
  `harness/oracles/rooms.py`.

## What was not taken

- Location phrases (`on your left`, `in front of you`): our agent has no
  facing direction.
- Turn left / turn right / forward: our moves are the four compass steps,
  as in the dungeon.
- Boxes that open to reveal their contents, and closing doors.
- BabyAI's own layouts and level classes; our layouts are sampled in
  `rooms.py` (1x2, 1x3 or 2x2 rooms).
