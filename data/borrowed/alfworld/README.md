# ALFWorld wording bank

**What it is.** ALFWorld is a text version of the ALFRED household benchmark: an
agent is given a one-line goal ("put a clean mug in coffeemachine") and acts
with short text commands over receptacles and objects. This bank holds its
task types, goal templates, command templates, observation wording and
object/receptacle vocabulary, for a future `kitchen` world (plan, kind 4).

- **Source:** https://github.com/alfworld/alfworld
- **Revision:** git commit `aaba6870f86c5be6a08a491f32a50b906227bc3e` (2026-02-08)
- **Licence:** MIT, Copyright (c) 2020 Mohit Shridhar and others. Copy in `LICENSE.upstream`.
- **Pulled:** 2026-09-29, shallow git clone.

## What was taken (232 rows in `bank.jsonl`)

| kind | rows | from |
|---|---|---|
| `task_type` | 12 | `alfworld/gen/goal_library.py`: the six ALFWorld task types plus their `_slice` variants. `meta` holds the templates, the PDDL goal, and the room types each type is valid in (`constants.GOALS_VALID`). |
| `task_template` | 24 | The goal-description templates for those 12 variants (`put a clean {obj} in {recep}`), with slot names. ALFWorld fills these in `agents/utils/misc.py:get_templated_task_desc`. |
| `action_template` | 21 | Command templates from `alfworld/data/alfred.twl2` (`go to`, `open`, `close`, `take {o} from {r}`, `move {o} to {r}`, `heat/clean/cool {o} with {r}`, `slice`, `use`, `examine`, `look`, `inventory`, `help`). |
| `feedback_template` | 51 | Observation strings from the same grammar ("You arrive at {r.name}.", "The {r.name} is closed."), with the condition under which each fires. Upstream debug prefixes such as `PickupObjectFromReceptacleObject:` are kept verbatim. |
| `receptacle` | 39 | Receptacle classes from `constants.RECEPTACLES` plus Sink and Bathtub, with `movable`, `openable` and `accepts` (the objects `VAL_RECEPTACLE_OBJECTS` allows in each). |
| `object` | 85 | Object classes (including sliced forms), with `pickupable`, affordances (heatable, coolable, cleanable, toggleable, sliceable) and `can_go_in`. |

The six task types: pick and place (`pick_and_place_simple`), pick two
(`pick_two_obj_and_place`), examine in light (`look_at_obj_in_light`), and clean,
heat or cool then place (`pick_{clean,heat,cool}_then_place_in_recep`).

## Deliberately not taken

- ALFRED's 7th type (`pick_and_place_with_movable_recep`) and the extra types in
  `goal_library.py` that ALFWorld does not use (place-all, pick-three, and so on).
- ALFRED's human-written goal annotations (`traj_data.json`, a separate download)
  and the scene layouts (`gen/layouts/*.json`). The layouts are room graphs, and
  the plan keeps room navigation out of training so the `house` world stays a
  transfer test.
- Expert trajectories, PDDL problem files and game files.

## Attribution

> Task templates, command templates and vocabulary adapted from ALFWorld
> (Shridhar et al., "ALFWorld: Aligning Text and Embodied Environments for
> Interactive Learning", ICLR 2021), https://github.com/alfworld/alfworld,
> MIT License, Copyright (c) 2020 Mohit Shridhar and others. Built on ALFRED
> (Shridhar et al., CVPR 2020).
