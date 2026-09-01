# Kanban Board Demo — real agent, fake board

React + TypeScript + Vite build of the original design mockup
(`client/design/understory.html`). See
`.claude/plans/understory-kanban-frontend.md` for how this evolved — it
started as a scripted 3-message demo and ended up here: no preloaded
conversation, type anything.

**What's real:** in-browser LLM inference (wllama/WASM, grammar-constrained),
the full `core/` parse/typecheck/effects/compile pipeline, sandboxed
execution (`runtime/sandbox.js`), and the approval gate — it's the real
runtime effect gate genuinely blocking an unapproved DELETE/SEND/PAY call,
not a UI simulation of one.

**What's fake:** the board (`src/data/board.ts`) — invented cards and
people, not a real team's data. `send_message` and friends are already
no-ops in this sandbox regardless (it's an eval harness, not a live
system).

## Run it

Two processes, both from the repo root:

```
python client/poc/server/dev_server.py --port 8080
```

```
cd client/kanban-ui
npm install
npm run dev
```

Open the Vite URL it prints (typically `http://localhost:5173`). Vite
proxies `/validate`, `/kanban_prompt`, `/models/*`, and `/agent_core.gbnf`
to the Python server (see `vite.config.ts`) and sends the same
`Cross-Origin-Opener-Policy`/`Cross-Origin-Embedder-Policy` headers
`dev_server.py` does, which wllama's multi-threaded WASM path needs.

The model starts loading immediately — no button. First run pulls the
~800MB GGUF from the local dev server; cached after that (5–15s loads
locally).

## How it actually works

Type anything (e.g. *"archive the overdue cards assigned to Bob"*, *"set
card 3 to done"*, *"message Priya about her overdue card"*). Each request:

1. `POST /kanban_prompt` — builds a fresh, grammar-matched
   `TOOLS`/`FIELDS`/`CONSTANTS` context for the `kanban` world from your
   current board (`client/poc/server/dev_server.py`'s
   `handle_kanban_prompt`, using `harness.context.build_context()` — the
   same function the offline curriculum generator uses, not a
   reimplementation).
2. Real grammar-constrained generation (`src/lib/llm.ts`, ported from
   `client/poc/src/llm.js`).
3. `POST /validate` — real parse/typecheck/effects/compile
   (`core.pipeline.build`) and real sandboxed execution
   (`harness.run.run_sandbox`), run **unapproved first**. If it touches a
   DELETE/SEND/PAY call, the real effect gate (`runtime/sandbox.js`)
   returns `EFFECT_BLOCKED` — the UI renders that as an approval gate, and
   only resends the *same* generated program with the approval flag flipped
   once a real click grants it.

Tool-call detail lines and the closing summary are derived from the real
`calls` log and a before/after board diff (`src/lib/describe.ts`,
`summarize()` in the hook) — not canned text.

Each generated program is shown in the chat, collapsed under "Show generated
program" — real Agent Core source, not a paraphrase. A stats line
(`tokens · time · tok/s`) follows once the turn resolves, computed from
wllama's own `usage`/timing numbers, not measured client-side around the
whole request.

### CPU only toggle

The header checkbox forces WASM-only decoding (`n_gpu_layers: 0`), skipping
WebGPU offload entirely — see `createPlanner()`/`loadPlanner()` in
`src/lib/llm.ts`/`src/hooks/useAgentRun.ts`. It's a real lever, not
guaranteed to be faster: WebGPU compute-shader dispatch for a small
quantized model can be slower than plain multi-threaded WASM on a given
laptop, so treat it as "try both," not "the fast option." Flipping it
reloads the model from scratch (`n_gpu_layers` is a load-time setting), so
it's disabled while a request is in flight or the model is already loading.
There is no server-side inference path — generation always runs in the
browser; this only changes which in-browser backend it uses.

### Why only cards the request actually names get a shortcut constant

`constants_from_board()` in `dev_server.py` doesn't dump the whole board as
constants on every request — it only includes a card if the request names
it by number (`"card 3"`, `"#3"`) or a shared title word.
`relevant_cards()` there explains why in detail, but the short version: a
bulk/rule request like *"archive the overdue cards assigned to Bob"* has
**zero** card constants in the real curriculum data (`data/curriculum_tasks.jsonl`)
— the reference solution lists and filters. Dumping all 14 cards as
constants for every free-typed request was a real, confirmed-by-testing
regression: `"Set card 3 to done."` picked the wrong card, consistently,
against both the tuned 0.8B model and the larger 2B checkpoint — a much
harder disambiguation problem than either was tuned to solve. Filtering
constants down to what the request plausibly means fixed it.

### The one real constraint

The grammar can only ever emit a pre-declared constant as a literal — the
model can't compose new text on the fly (spec: "every literal value must
be a C symbol"). That's fine for naming an existing card/user/status, but
a `send_message` body has to come from somewhere: `GENERIC_MESSAGES` in
`dev_server.py` is a small fixed bank of message bodies the model picks
from. Real limitation, not hidden — you'll see one of those five sentences
in any message it sends, not a custom-composed one.

## Known gaps vs. the original mockup

No fly-to-archive-drawer / delete animations — those were hand-timed
against a scripted sequence in `understory.html`; this build's timing
comes from real network + inference latency instead. Cards/messages get a
plain fade-in; the board otherwise just re-renders from real state.
