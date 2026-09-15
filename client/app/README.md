# Agent Core Demos — a real agent, three fake back ends

One merged app, replacing what used to be separate `client/kanban-ui` and
`client/rpg-ui` builds. `/` is a menu; `/kanban`, `/rpg`, and `/db` are
three independent demos of the same real thing: a small planner writes an
Agent Core program, the program is compiled and sandboxed for real, and the
data it acts on is fake. See `.claude/plans/example-host-and-new-worlds.md`
for why this merged and what changed to make that safe.

## Run it

Two processes, both from the repo root:

```
python server/dev_server.py --port 8080
```

```
cd client/app
npm install
npm run dev
```

Open the Vite URL it prints (typically `http://localhost:5173`). Vite
proxies `/validate`, `/kanban_prompt`, `/rpg_new`, `/rpg_prompt`,
`/db_prompt`, `/db_new`, `/apps`, `/models/*`, and `/agent_core.gbnf` to the
running Python server and sends the same COOP/COEP headers `dev_server.py`
does, which wllama's multi-threaded WASM path needs. `COVENANT_BACKEND`
points at a different server (another port, another checkpoint) without
editing `vite.config.ts`.

## The three demos

**Kanban Board** (`/kanban`, `src/worlds/kanban/`) — free-typed requests
against a fake team board: "archive the overdue cards assigned to Bob",
"message Priya about her overdue card." The full pipeline is real: in-browser
grammar-constrained generation, the real `core/` compile pipeline, real
sandboxed execution, and a real approval gate that runs the whole program
unapproved first and previews *everything* it would do before a DELETE/SEND
gets a click — see `.claude/plans/preview-run-approval-gate.md` and
`.claude/plans/host-preflight-and-bulk-gate.md`. The board itself
(`data/board.ts`) is invented.

**Dungeon Agent** (`/rpg`, `src/worlds/rpg/`) — a turn-based grid dungeon.
Every other world here is CRUD over records; this one needs perception, a
plan across turns, and a decision that depends on what was just seen — and
it's held out of training (`data/holdout/reserved.json`) so it measures
whether the architecture carries to a domain the corpus never described.
See `.claude/plans/rpg-demo-app.md` and `results/RPG.md`.

**Database Analyst** (`/db`, `src/worlds/db/`) — ask questions of a fake
CRM (customers, tickets, invoices, staff): "which customers are
delinquent", "close ticket 2", "list open tickets sorted by priority."
Unlike the other two, the fake data isn't authored in TypeScript at all —
`runtime/worlds/crm.py` already had a `default_state`, so `POST /db_new`
just hands the client a copy of it, the same way `/rpg_new` hands over a
dungeon it didn't build.

## Why this is one app now, and what that took

Merging two full-page apps into client-routed pages isn't free — two things
had to be built deliberately, not assumed:

- **Model lifecycle.** Each world's hook (`useAgentRun.ts`, `useDbRun.ts`,
  `useDungeonRun.ts`) now has a dedicated unmount-cleanup effect that awaits
  `planner.unload()`. Without it, navigating from one world to another
  while a model is loaded would try to open a second OPFS access handle on
  the next world's model file while the first is still open — the same
  failure `useAgentRun.ts`'s StrictMode comment already documents, just
  triggered by routing instead of a double-invoke.
- **CSS scoping.** Every world's stylesheet has its top-level selectors
  prefixed under a wrapper class (`.world-kanban`, `.world-rpg`,
  `.world-db`) — kanban and rpg both defined `.chrome` with different rules
  before this, a real collision once both stylesheets can be in one page at
  once. `react-router` + `React.lazy` per route also means each world's CSS
  chunk only loads when its route is visited.

Each world still owns its own components and styling — no shared
`ChatPanel`/`Message`/board-or-dungeon-or-table component — matching the
convention `InferenceControls.tsx`'s own comment already established
(logic lives in `client/shared/*.ts`; markup stays per-world).

## Adding a fourth world

1. A `runtime/worlds/<name>.py` `WORLD` dict — entities, tools, effects. If
   it needs non-CRUD rules (a grid, a turn order), a `runtime/engines/`
   module and a `post_hook`, the way `rpg`/`warehouse`/`cards` do.
2. A server handler pair mirroring `handle_db_new`/`handle_db_prompt` (or
   `handle_kanban_prompt` if the world has no server-owned state) in
   `server/dev_server.py`, registered in `do_POST`'s handler table and (for
   the static route) `APPS`.
3. `src/worlds/<name>/` — `App.tsx` (root wrapped in `.world-<name>`, its
   own `styles.css` imported there so it code-splits with the route), a
   hook following `useDbRun.ts`'s shape, and a lazy route added to
   `router.tsx`.
