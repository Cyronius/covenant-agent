# TextCraft and Crafter crafting mechanics (borrowed)

Used by `runtime/worlds/workshop.py` (the `workshop` decision world and its
held-out `boatyard` tree), `runtime/engines/workshop.js` and
`harness/oracles/workshop.py`. Plan: `.claude/plans/borrowed-worlds.md`,
"2. Crafting".

## Sources

| Source | URL | Licence | Commit |
|---|---|---|---|
| ADaPT (TextCraft) | https://github.com/archiki/ADaPT | MIT | `ecdc4ab0030b4be9be122622d8ea78f8c59c44c4` (2024-01-03) |
| Crafter | https://github.com/danijar/crafter | MIT | `e04542a2159f1aad3d4c5ad52e8185717380ee3a` (2023-12-13) |

Both licence texts are kept in `LICENSE.upstream`. Only mechanics were
taken, re-implemented. No upstream code is copied and
no upstream data is used. TextCraft's recipes are Minecraft's (loaded from a
Minecraft data directory, `TextCraft/env.py:16`); none of them are here. All
item names, recipes, counts and yields in `workshop.py` are ours.

## What was taken

From TextCraft (`TextCraft/env.py`, `TextCraft/crafting_tree.py`):

- **The two verbs.** `get N item` for base materials and `craft X using
  ingredients` for everything else (`env.py:19-20`). Ours are `gather` and
  `craft`, and they take one item each; the recipe is looked up rather than
  written out by the agent.
- **Counted recipes and an inventory.** A recipe consumes counts of its
  ingredients and yields a count of its output; crafting fails when the
  inventory is short (`env.py:44`, `has_items` at `env.py:90`). Ours says
  what is short and what to do about it ("short 4 plank for workbench - craft
  plank first").
- **Goals several recipes deep, with the recipes shown.** TextCraft's
  observation lists the relevant crafting commands and the goal
  (`env.py:161`), and goals are picked by recipe depth
  (`crafting_tree.py:275`, `get_min_depth`). Ours puts each recipe in an
  item constant's description, and samples either a tree's root or one of
  its deeper intermediates.
- **Distractor recipes.** TextCraft's recipe set is the goal's tree plus
  recipes from elsewhere that share its inputs (`crafting_tree.py:317`,
  `create_recipe_set`). Ours shows two or three items from other trees
  beside the job's own.

The oracle is ours: TextCraft's agent is a language model, so there is no
scripted solver to borrow. `harness/oracles/workshop.py` is a recursive
recipe planner over the tree.

From Crafter (`crafter/data.yaml`, `make:` and `place:`):

- **The tech tree with stations.** Some recipes need a station nearby
  (`nearby: [table]`, `nearby: [table, furnace]`), and the stations are
  themselves built from gathered materials (table from wood, furnace from
  stone). Ours: workbench, kiln, loom and anvil, plus the held-out tree's
  slipway, each crafted from materials and never used up.

## Not taken

- Minecraft item names, recipe data, tags or counts (TextCraft).
- Crafter's grid, survival stats, creatures and achievements.
