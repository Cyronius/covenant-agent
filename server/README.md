# server — dev server

Dev-only local server behind the merged demo app (`client/app` — kanban,
rpg, db) and the browser-inference stand-in
(`.claude/plans/browser-inference-standin.md`). Pure Python stdlib —
`http.server.ThreadingHTTPServer`, no Flask/FastAPI, no new dependencies.

It does these things:

1. **Static file server** for the built app — `client/app/dist`
   (index.html, `assets/*.js`, `vendor/wllama/*.wasm`, ...), with an SPA
   fallback to `index.html` for any unmatched GET so `react-router`'s
   client-side routes (`/kanban`, `/rpg`, `/db`) work on a hard refresh —
   plus the large GGUF model file served in place from
   `baselines/qwen/models/` (never copied — it's ~0.8–2GB and gitignored)
   with `Range:` request support, and the grammar file at
   `baselines/qwen/agent_core.gbnf`.
2. **`POST /validate`** — runs generated Agent Core program text through
   the existing Python pipeline (`core.pipeline.build`) and sandbox
   (`harness.run.run_sandbox`), the same path `harness/run.py`'s
   `run_task()` uses. This is the dev-shortcut round-trip described in
   scope item 4(a) of the plan — not a reimplementation of
   parse/typecheck/compile/execute. Accepts either a `task_id` (looked up
   in `data/curriculum_tasks.jsonl`) or an inline `context`/`world`/`now`
   (as returned by `/kanban_prompt` below) — see the `/validate` section.
3. **`POST /rpg_new` and `POST /rpg_prompt`** — the grid-RPG demo
   (`client/app/src/worlds/rpg`, plan `.claude/plans/rpg-demo-app.md`). `/rpg_new
   {scenario?}` deals a fresh dungeon and returns `{state}`; the server owns
   the map so the client never carries a second copy. `/rpg_prompt {state}`
   is that world's analogue of `/kanban_prompt`: instead of a typed request,
   the request text *is* the rendered observation (`runtime/worlds/rpg.py`'s
   `observe`), and the constants are the things currently in view. It
   returns `{input_text, context, grammar, system, world: "rpg", now, state,
   observation}` (`grammar`/`system` as for `/kanban_prompt` above) —
   feed `input_text` to `buildFullPrompt()`, send `context`/`world`/`now` to
   `/validate`, and thread the **returned** `state` forward (its `memory`
   has been advanced by the act of observing). `observation` carries the
   window and the relative offsets so a UI can draw the same fog the prompt
   describes rather than reimplementing the rule.

4. **`GET /models` and model switching** — `GET /models` lists the GGUFs in
   `baselines/qwen/models/` as `{default, loaded, models: [{name, template,
   tuned, size_mb}]}`. `template` says how a checkpoint wants to be
   prompted: `qwen` for our own tuned ones (the client builds the ChatML
   markup they were SFT'd on, byte-identical to the browser path), `chat`
   for anything else (send `{system, user}` and llama-cpp-python applies the
   GGUF's own `tokenizer.chat_template`). `POST /plan {"model": name}`
   switches checkpoint; one planner is loaded at a time and the previous is
   dropped, since a 2B Q8 is ~2.5 GB.

5. **`POST /kanban_prompt`** — for `client/app/src/worlds/kanban`'s free-typed chat:
   given `{"request": "<anything>", "state": <a kanban board>}`, builds a
   fresh `TOOLS`/`FIELDS`/`CONSTANTS` context for the `kanban` world with
   `harness.context.build_context()`/`serialize_context()` (the same
   functions the offline curriculum generator uses) instead of looking up
   a fixed task — every card/user on the supplied board becomes a
   referenceable constant. Returns
   `{"input_text", "context", "world": "kanban", "now", "grammar", "system", "preflight"}`;
   feed `input_text` and `system` to `buildFullPrompt()`, pass `grammar` to
   `generate()`, and send `context`/`world`/`now` straight through to
   `/validate` in place of `task_id`.

   **`grammar` and `system` are not optional extras.** The context is
   serialized in the typed surface (spec 0.4.0 letters: `S0`, `I0`, `D0`)
   because that is what every checkpoint since S3 trained on, and `grammar`
   is built for *this* symbol table, so each `CALL` slot admits only
   type-compatible constants. Using the static `/agent_core.gbnf` with this
   prompt puts the model back where the demo used to be: it can spell
   `CALL T3 <user id> <title> <date>`, which typechecks as three errors and
   runs nothing. `POST /plan` takes the same `grammar` string and caches the
   compiled form. See `.claude/plans/demo-typed-surface.md`.

   **`preflight`** is non-null when the request addresses somebody the board
   doesn't have (`{"status": "unknown_person" | "ambiguous_person", "name",
   "message", "people", "suggestion"?, "candidates"?}`). A client should
   answer with `message` and not generate: resolving named people is a board
   lookup the host can't get wrong, and a model asked to assign to a person
   who doesn't exist picks one who does — "assign all issues to cyrus" put
   four cards on Bob. The prompt is still built and returned, so nothing
   about the surface changes. See
   `.claude/plans/host-preflight-and-bulk-gate.md`.

6. **`POST /db_new` and `POST /db_prompt`** — the Database Analyst demo
   (`client/app/src/worlds/db`). Unlike kanban's board, the crm world
   (`runtime/worlds/crm.py`) already has a `default_state`, so `/db_new {}`
   just returns `{state: <a copy of it>}` — no client-authored fake data to
   keep in sync. `/db_prompt {request, state}` is `/kanban_prompt`'s
   analogue against the `crm` world: `{input_text, context, grammar,
   system, world: "crm", now}`. No `preflight` — nothing in this world
   assigns to a named person the way `assign_card` does.

7. **`GET /apps`** — `{"apps": [{slug, path, title, blurb}, ...]}`, the
   registry `client/app/src/pages/Home.tsx`'s landing page renders. Add a
   world here (and to the static router's nothing-to-do-since-`react-router`-
   owns-it fallback) when a fourth demo lands.

## Run

```
python server/dev_server.py --port 8080
```

Runnable from any working directory — it resolves the repo root relative
to its own file location (`server/dev_server.py` → repo root is
3 levels up), the same convention `harness/run.py` uses.

Startup prints the resolved repo root, static root, models dir, grammar
file path, and how many tasks were indexed from
`data/curriculum_tasks.jsonl`.

`index.html` and `src/*` are owned by a parallel workstream and may not
exist yet — a request for a missing static file 404s cleanly instead of
crashing the server.

## Static routes

- `GET /` and every path under it → `client/app/dist/<path>`, falling back
  to `client/app/dist/index.html` when the path isn't a real built file —
  `react-router` owns `/kanban`, `/rpg`, `/db` client-side, so a refresh on
  any of them needs this fallback to work (directory traversal outside
  `client/app/dist/` is rejected with 403)
- `GET /models/<filename>` → `baselines/qwen/models/<filename>`, served in
  place with full `Range:` support (`206 Partial Content` +
  `Content-Range`/`Accept-Ranges`, or a normal `200` whole-file response for
  a non-range GET)
- `GET /agent_core.gbnf` → `baselines/qwen/agent_core.gbnf` as `text/plain`

Content-Type is set by extension: `.wasm` → `application/wasm`, `.js`/
`.mjs` → `text/javascript`, `.json` → `application/json`, `.html` →
`text/html`, `.gbnf`/unknown → `text/plain`, `.gguf` →
`application/octet-stream`.

## `POST /validate`

### Request

```jsonc
{
  "task_id": "L0_kanban_delete",   // EITHER this: id in data/curriculum_tasks.jsonl
  // ...OR both of these instead (as returned by POST /kanban_prompt):
  "context": { "tools": [...], "fields": [...], "constants": [...] },
  "world": "kanban",
  "now": 1760000000,               // optional with inline context; defaults to the world's own "now"
  "text": "CALL T9 C0\nSTOP\n",    // required: generated Agent Core program text for this segment
  "state": null,                   // world state to run against; required with inline context, optional (defaults to the task's own) with task_id
  "registers": null,               // register bindings carried into this segment, or null/{} for the first segment
  "pause_types": null,             // see "Continuing after a PAUSE" below — omit/null on the first segment
  "approval": null                 // optional: true/false overrides the default approval token; omit/null to use the task's own default (or false with inline context, which has none)
}
```

- `approval`: the runtime effect gate (`runtime/sandbox.js`) blocks any `DELETE`/`SEND`/`PAY` call unless the run carries an approval token — see `spec/agent_core.md` §7. That token is normally the task's own stored `approval` field, but a client can override it per request: send `false` to deliberately run unapproved (the real gate then returns `"status": "effect_blocked"` for the first destructive call, not a client-side simulation of one), then re-send the same `text`/`state` with `true` once a person has actually approved it.
- **Preview runs.** On the inline-context path for the `kanban` world, the
  server sets the sandbox's `preview` flag and `bulk_write_limit`
  (`BULK_WRITE_LIMIT`, 2). An unapproved preview run does **not** halt at the
  first destructive call: it runs to the end against the state clone it
  already works on, and if it made any `DELETE`/`SEND`/`PAY` call — or more
  writes than the limit — the result is `"status": "effect_blocked"` with
  `error: {code: "DESTRUCTIVE" | "BULK_WRITE", effect, count, calls: [{tool, name, args, effect}]}`.
  `calls` is everything the program would do, exact because it completed;
  re-send with `approval: true` to apply it. This exists because the halting
  gate previewed one call and handed back a token good for all of them: one
  click on `delete_card #2` deleted seven cards (2026-09-14). Not set for
  `task_id` runs or for the RPG (where every tool is a `WRITE` and moving
  three squares is an ordinary turn), and `harness/run.py` builds its own
  payload — so spec §7's halt-at-the-first-call is still what every scored
  run gets.
- `state`: the client threads this forward turn to turn, mirroring how
  `run_task()` threads `state` through segments. Send `null` (or omit) on
  the very first segment to use the task's stored initial state; on every
  later segment, send back exactly the `final_state` from the previous
  response.
- `registers`: same idea — `null`/`{}` on the first segment, otherwise the
  previous response's `registers`.

### Response

```jsonc
{
  "status": "ok",              // "ok" | "paused" | "static_error" | "effect_blocked" | "error" | "server_error" | ...
  "diagnostics": [],           // rendered diagnostic strings from core.pipeline.build (parse/typecheck/effect errors)
  "final_state": { ... },      // world state after this segment; feed back as next request's "state"
  "registers": { ... },        // register bindings after this segment; feed back as next request's "registers"
  "calls": [ ... ],            // tool call log for this segment
  "return_value": null,        // sandbox return value, if any
  "pause_envs": null,          // only non-null when status == "paused" — see below
  "error": null                // sandbox error object ({code, message}), also the blocked call on "effect_blocked"
}
```

If the world declares a `post_hook` (`runtime/worlds/__init__.py`), the
server adds it to the sandbox payload, so the world's per-turn phase — the
RPG's enemy turn — runs once after the program ends. The demo and the
offline suite therefore advance the game through exactly the same code.

If `core.pipeline.build` doesn't compile (parse/typecheck/effect errors),
the response is `{"status": "static_error", "diagnostics": [...], ...other fields null/empty...}`
— this is a normal, expected outcome (HTTP 200), not a server error.

An unhandled exception is caught and returned as HTTP 500 with
`{"status": "server_error", "error": {"code": "SERVER_ERROR", "message": "..."}, "traceback": "..."}`
— the process never crashes on a bad request.

### Continuing after a PAUSE

This server is **stateless across requests** — it does not remember
anything from one `/validate` call to the next. When a segment's program
ends in `PAUSE`, its response has `status: "paused"` and a `pause_envs`
field: a list of `{register_name: type_string}` dicts (one BuildResult, so
in practice a 0- or 1-element list — mirrors `BuildResult.pause_envs`).
That's the same info `harness/run.py`'s `run_task()` uses internally to
re-type registers via `parse_type()` before building the next segment's
`TaskContext.initial_registers`.

Since the server has no session state, **the client must echo this back**:
on the request for the *next* segment, send the previous response's
`pause_envs[0]` (the dict, not the list) as `pause_types`. The server then
reconstructs `ctx.initial_registers = {r: parse_type(t) for r, t in
pause_types.items() if r in registers}` exactly as `run_task()` does after
a pause.

Concretely, a two-segment task:

1. `POST /validate {task_id, text: segment_0, state: null, registers: null}`
   → `{status: "paused", final_state: S1, registers: R1, pause_envs: [{"r0": "LIST OBJ:card", "r1": "LIST OBJ:card"}], ...}`
2. `POST /validate {task_id, text: segment_1, state: S1, registers: R1, pause_types: {"r0": "LIST OBJ:card", "r1": "LIST OBJ:card"}}`
   → `{status: "ok", final_state: S2, registers: null, ...}`

If a segment does *not* pause, `pause_envs` is `null` in the response and
the next request (if any — e.g. a fresh task) should omit `pause_types`.

## Verified end-to-end (curl)

Sanity-tested against a real task from `data/curriculum_tasks.jsonl` using
its known-good reference program
(`task["reference"]["segments"][0]`), with the server running locally:

**Single-segment task** — `L0_kanban_delete`, reference program
`"CALL T9 C0\nSTOP\n"` (`delete_card` on `card_4`):

```
curl -s -X POST http://127.0.0.1:8080/validate \
  -H "Content-Type: application/json" \
  --data '{"task_id":"L0_kanban_delete","text":"CALL T9 C0\nSTOP\n","state":null,"registers":null}'
```

Response: `status: "ok"`, `calls: [{"tool":"T9","name":"delete_card","args":["card_4"],"ok":true,"error":null}]`,
and `final_state` with `card_4` removed from the `card` entity list. Passed.

**Two-segment PAUSE task** — `L10_kanban_pause_archive`, reference program
segments `["CALL T1 -> r0\nFILTER r0 F0 LT NOW AND F10 EQ C0 -> r1\nPAUSE\n",
"FOREACH r1 -> r2\n  CALL T2 r2 -> r3\nSTOP\n"]`:

1. First request (`state: null, registers: null`) → `status: "paused"`,
   `pause_envs: [{"r0": "LIST OBJ:card", "r1": "LIST OBJ:card"}]`.
2. Second request, feeding back `state`, `registers`, and
   `pause_types: {"r0": "LIST OBJ:card", "r1": "LIST OBJ:card"}` from the
   first response → `status: "ok"`, with the four matching cards
   (`card_1`, `card_3`, `card_4`, `card_6`) archived in `final_state` and
   four `archive_card` calls logged.

Both runs passed — full parse/typecheck/compile/execute round trip,
including the PAUSE continuation contract, confirmed working without a
browser in the loop.

Also spot-checked: malformed program text returns
`{"status": "static_error", "diagnostics": ["PARSE_ERROR ..."]}` (HTTP
200, not a crash), and an unknown `task_id` returns
`{"status": "server_error", "error": {"code": "UNKNOWN_TASK", ...}}`.

## `POST /plan` and `GET /plan/status` — server-side inference

Added 2026-09-02 (plan `s2-consolidated-program` §A7). Runs the same GGUF
and `agent_core.gbnf` the browser path uses, through llama-cpp-python on
this machine's CPU. The client builds the prompt exactly as it does for
the browser path and posts it, so both paths send byte-identical text.

- `GET /plan/status` → `{available, model, loaded, reason?}`
- `POST /plan {"warm": true}` → loads the model (first call), returns status
- `POST /plan {"prompt": "<full chat-formatted prompt>", "max_tokens"?: 250, "stop"?: ["<|im_end|>"]}`
  → `{text, finish_reason, tokens_out, tokens_in, gen_ms, model}`

`--model PATH` picks the checkpoint (default `qwen3.5-0.8b-s1-q8.gguf`);
`--ctx N` sets the context size. Generation is serialized with a lock
(llama.cpp contexts are not thread-safe).

## `POST /write` — the writer tool's backend

Called by the sandbox, not by the client: when `/validate` runs a program
that calls an `EXTERNAL` tool (`write_text` in the kanban world), the
sandbox POSTs `{kind, params}` here and uses the returned `{text}`. With
no server (the eval harness), the sandbox returns a deterministic stub
`[write_text: <brief>]` instead, so evals score routing, not prose.
Backed by `--writer-model` (default: the untuned `Qwen3.5-0.8B-Q8_0.gguf`,
a second llama.cpp instance). The merged S1 planner checkpoint was tried
first and just echoes the data list back; the base weights write a proper
short message. Falls back to the planner weights if the file is missing.
