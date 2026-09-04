# Dungeon Agent — the planner plays a game

A turn-based grid dungeon driven by the same small planner that runs the
kanban demo. Each turn the model is shown a 5×5 window of its surroundings
and writes an Agent Core program; the program is compiled and executed
against the real rules, enemies answer, and the next turn starts from
whatever state that left.

The point is not the game. Every other world in this repo is CRUD over
records — list, filter, update. This one needs perception, a plan across
turns, and a decision that depends on what was just seen, and it is
**held out of training** (`data/holdout/reserved.json`), so it measures
whether the architecture carries to a domain the corpus never described.
See `.claude/plans/rpg-demo-app.md` and `results/RPG.md`.

## Run it

Two processes, both from the repo root:

```
python server/dev_server.py --port 8080
```

```
cd client/rpg-ui
npm install
npm run dev
```

Open the Vite URL (typically `http://localhost:5173`). Vite proxies
`/rpg_new`, `/rpg_prompt`, `/validate`, `/plan`, `/models`, and
`/agent_core.gbnf` to the Python server and sends the COOP/COEP headers
wllama's multi-threaded WASM path needs. To point at a server on another
port, set `COVENANT_BACKEND=http://localhost:8082`.

**Step** plays one turn. **Auto-play** keeps going until the stairs, death,
or the turn cap. **New game** re-deals.

## What is real

- **The model.** Real grammar-constrained generation, either in the browser
  (wllama, WASM or WebGPU) or on the server's CPU (`POST /plan`,
  llama.cpp). The header picks the backend and the checkpoint; both are
  load-time settings, so switching either reloads.
- **The program.** Real parse → typecheck → effect check → compile
  (`core/`), then real execution in the Node sandbox. A program that does
  not typecheck fails the turn, and you can read it in the log.
- **The rules.** `runtime/engines/rpg.js`, run inside the sandbox — the same
  code the offline suite (`harness/rpg_suite.py`) executes, so what you
  watch here and what `results/RPG.md` reports are the same game.
- **The perception.** The observation text and the constants come from
  `rpg.observe` on the server. The dashed box on the map is exactly the
  window that went into the prompt.

## What is authored

The dungeon itself: one hand-drawn 12×9 map, two goblins, a potion, a key,
a locked door, and the stairs (`runtime/worlds/rpg.py`, `SCENARIOS`). No
randomness anywhere — same program, same result — which is what makes two
models comparable on the same seeds.

The art is drawn in code as small pixel arrays (`DungeonView.tsx`). No
sprite sheets and nothing lifted from a published game.

## The rules, briefly

Five tools: `move(direction)`, `attack(enemy)`, `pick_up(item)`,
`use_item(item)`, `interact(door)`.

- Three actions per turn. A fourth is refused (`RATE_LIMITED`), and an
  illegal action still spends one, so wrapping moves in `TRY` buys no free
  probing of the map.
- Enemies act **once per turn, after the whole program** — not between
  calls. Acting between calls would invalidate the snapshot the model
  planned from and turn a fair plan into a failure.
- A goblin within three tiles closes in; adjacent, it hits for 2. The player
  has 10 HP and hits for 3, so a goblin dies in one blow.
- The locked door needs the key, which the key opens and is consumed by.
- Stepping onto the stairs wins; 0 HP ends it.

## Reading the turn log

Each entry shows the calls with their real arguments (a struck-through call
failed, with the error code), the events the rules produced, and two
collapsed panels: the generated program verbatim, and the exact text the
model was shown. When it walks into a wall, those two panels tell you
whether it could see the wall.
