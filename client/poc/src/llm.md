# wllama API notes (v3.6.1, installed 2026-08-31)

Findings are from `node_modules/@wllama/wllama`'s own `README.md`,
`esm/*.d.ts`, `esm/index.js` (the bundled runtime, read directly — it's
short enough to grep), and `guides/intro-v3.md` / `guides/intro-v3.1.md`.
Not from memory of older wllama releases, which had a materially different
API (old `create_completion`/single-vs-multi-thread WASM builds — both
gone in v3.x).

## Package facts

- `@wllama/wllama@3.6.1`, published by `ngxson`, MIT.
- v3.x rewrote wllama's core to reuse `server-context.cpp` from
  `llama-server` itself, so the completion API is now OAI-compatible
  (`createChatCompletion`, `createCompletion`, `createEmbedding`) instead
  of the older bespoke API.
- `esm/index.js` is a **fully self-contained bundle — zero `import`
  statements**. The entire emscripten glue (the compiled llama.cpp runtime)
  is inlined as a JS string constant and turned into a `Worker` via
  `URL.createObjectURL(new Blob([code], {type: 'text/javascript'}))` at
  runtime (`esm/index.js:1273` in this build). There is **no separate
  worker `.js` file to vendor** — that surprised me going in, since older
  wllama docs/discussions reference a worker script path.

## Constructing a Wllama instance

```js
import { Wllama } from '@wllama/wllama'; // or a relative path with no bundler, see below

const wllama = new Wllama({ default: '/path/to/wllama.wasm' });
```

`AssetsPathConfig` (`esm/wllama.d.ts`):
```ts
interface AssetsPathConfig {
  default: string;
  'single-thread/wllama.wasm'?: string; // legacy, pre-v3.1 — not needed now
  'multi-thread/wllama.wasm'?: string;  // legacy, pre-v3.1 — not needed now
}
```
As of v3.1 (`guides/intro-v3.1.md`) wllama ships **one WASM build** that
handles single-thread, multi-thread, and WebGPU at runtime — the old
single/multi-thread split is gone. Only `default` needs a value; the two
legacy keys exist for advanced overrides. This build only has one
`.wasm` file (`esm/wasm/wllama.wasm`, ~8.06 MB), confirmed by listing the
installed package — no second variant on disk.

`default` is resolved with `new URL(relativePath, document.baseURI).href`
inside wllama (`absoluteUrl()`, `esm/index.js:1084`), i.e. **relative to
the page**, not to wherever `@wllama/wllama`'s own JS happens to live. To
stay correct regardless of where the HTML page that imports `llm.js` sits,
`llm.js` resolves the wasm URL relative to **its own module URL**
(`import.meta.url`) into an absolute URL before handing it to `Wllama`, so
`document.baseURI` never enters into it.

## Loading a GGUF from a URL

```js
await wllama.loadModelFromUrl(modelUrl, {
  n_ctx: 4096,
  n_threads: 4, // optional; omitted lets wllama pick hardwareConcurrency/2
});
```
`loadModelFromUrl(modelSourceOrURL: ModelSource | string, params?)` — a
plain string URL works directly (`ModelSource` is only needed for the
multi-shard / mmproj case). Confirmed in `esm/wllama.d.ts`. This is a real
fetch of the URL (with the browser's own IndexedDB-backed cache layer in
front of it, via `CacheManager`/`ModelManager`) — not a local-file-only
API. `loadModelFromHF` is the HF Hub sibling; not used here since the GGUF
is served from our own static server.

`LoadModelParams` (`esm/types/types.d.ts`) has ~50 fields (rope, YARN,
LoRA adapters, speculative decoding, etc.) — `n_ctx` and `n_threads` are
the only ones this POC sets, matching `baselines/qwen/run_a.py`'s
`--ctx 4096` default.

## Grammar-constrained generation — CONFIRMED working in this version

```js
const res = await wllama.createCompletion({
  prompt: promptText,
  grammar: grammarText,   // raw GBNF text, as a string
  temperature: 0,
  max_tokens: 250,
  stop: ['<|im_end|>'],
});
res.choices[0].text          // generated text, stop string NOT included
res.usage.completion_tokens  // token count
```

`grammar?: string` lives on `SamplingParams` (`esm/types/types.d.ts:94`),
which both `RawCompletionParams` (used by `createCompletion`) and
`ChatCompletionParams` (used by `createChatCompletion`) intersect with. It
is a **raw GBNF grammar string**, not a JSON-schema wrapper and not a file
path — exactly the same shape `baselines/qwen/run_a.py:96-98` passes to
`llama-cpp-python`'s `grammar=`, and exactly what `agent_core.gbnf` already
is. No conversion needed; `llm.js` just `fetch()`s the `.gbnf` file as text
and passes it straight through.

I did not find any JS-level "translate this GBNF into something else"
step — `grammar` is a pure pass-through into the compiled WASM binary
(`esm/wasm/wllama.wasm`), which is llama.cpp's own `server-context.cpp`,
the same grammar-sampler engine `run_a.py` already exercises on the native
side. This is the strongest evidence available without a browser: the
field exists, is typed as a string, and is wired straight into the native
completion call with no JS-side reinterpretation.

**I could not run an actual generation to observe grammar-constrained
output** — no browser in this sandbox, and `Wllama`'s `CacheManager`
requires a browser storage backend (IndexedDB/OPFS) and throws
`No supported storage backend found` under plain Node (confirmed by
constructing a `Wllama` instance in Node — see below). This is expected
per the task scope, not a defect in `llm.js`. Treat grammar support as
**API-confirmed, not runtime-confirmed** until this runs in an actual
browser against `agent_core.gbnf`.

Stop strings: `stop?: string | string[]` on `RawCompletionParams` — array
form used here (`['<|im_end|>']`), matching `run_a.py`. Behavior (stop text
excluded from the returned string) matches llama.cpp/llama-server's
`/completion` endpoint semantics, which `server-context.cpp` also
implements natively for wllama.

Temperature 0 / greedy: `temperature?: number` on `RawCompletionParams`
(there's also a `temp?: number` from the intersected `SamplingParams` —
`temperature` is the OAI-compatible top-level alias; `llm.js` uses
`temperature: 0`, matching `run_a.py`'s `temperature=0.0`).

Max tokens: `max_tokens?: number`, used directly.

## WebGPU: real, but immature and unconfirmed here

- `wllama.isSupportWebGPU()` is a pure capability check —
  `!!navigator.gpu` (`esm/index.js:1239-1240`) — not a confirmation that a
  given completion call actually ran on GPU.
- WebGPU landed in v3.1 (`guides/intro-v3.1.md`, PR #215). The guide is
  explicit: **"Currently only supports Chrome (for Firefox, a flag must be
  enabled manually)."** Safari needs `wllama.setCompat('default',
  'firefox_safari')` and the guide warns performance is "significantly
  degraded" in that compat path.
- When supported, GPU offload is **on by default** — v3.1 auto-offloads
  all layers unless you pass `n_gpu_layers: 0` (which forces WASM-only) or
  a smaller number. `llm.js` leaves `n_gpu_layers` unset, i.e. it attempts
  GPU automatically when available, and reports `backend: 'webgpu'` in
  that case.
- **Important gap**: nothing in the public API (`Wllama` class members,
  `LoadedContextInfo`, completion responses) reports *which backend a
  completion call actually dispatched to*. `llm.js`'s `backend` field is
  therefore `isSupportWebGPU()`'s capability check, evaluated once at
  `createPlanner()` time — i.e. "WebGPU was available and offload was
  attempted," not "this generation call ran on GPU." If a browser
  environment reports `navigator.gpu` present but the actual llama.cpp
  WebGPU backend still falls back to CPU internally for some layers/ops,
  `llm.js` cannot currently detect that distinction. Treat `backend:
  'webgpu'` in the returned object as "requested," and corroborate with
  wall-clock `loadMs`/`genMs` (WASM CPU inference on a 0.8B model should be
  markedly slower) when this actually runs.
- Multi-threading (the CPU path's own scaling knob, orthogonal to WebGPU)
  requires `crossOriginIsolated` — i.e. `Cross-Origin-Embedder-Policy` and
  `Cross-Origin-Opener-Policy` response headers from whatever static server
  serves this POC — confirmed via `isSupportMultiThread()`
  (`esm/index.js:1118`), which probes `SharedArrayBuffer` support. Without
  those headers, wllama silently falls back to `n_threads: 1` regardless of
  what's requested; `wllama.isMultithread()` / `wllama.getNumThreads()`
  (available after `loadModel`) can confirm at runtime what was actually
  used.

## Contract conformance

The `createPlanner` contract in the task (backend/loadMs/generate returning
text/tokensOut/genMs, grammar+temp0+stop+maxTokens) **is fully supported
by the real v3.6.1 API** — no gaps required papering over. The one caveat
worth flagging again: `backend` is a capability-check-at-load-time value,
not a verified per-call dispatch trace (see WebGPU section above) — that's
a real limitation of what wllama exposes, not a shortcut taken in `llm.js`.

## No-bundler / vanilla usage

README's "Simple usage with ES6 module" section is exactly this POC's
shape:
```js
import { Wllama } from './esm/index.js';
const CONFIG_PATHS = { default: './esm/wasm/wllama.wasm' };
const wllama = new Wllama(CONFIG_PATHS);
```
`llm.js` follows this pattern, importing the vendored bundle by relative
path (`../vendor/wllama/index.js`) and resolving the wasm path via
`import.meta.url` (see above) instead of a path relative to the page.

README also documents a CDN option (`@wllama/wllama/esm/wasm-from-cdn.js`)
— explicitly **not used here**, since the task requires same-origin static
serving with no network dependency beyond the local dev server.

## Vendored static assets (`client/poc/vendor/wllama/`)

Copied straight out of the installed npm package (`esm/` subtree) with no
modification:

| File | Source in `@wllama/wllama` package | Size | Required MIME type |
|---|---|---|---|
| `vendor/wllama/index.js` | `esm/index.js` | ~367 KB | `text/javascript` (or `application/javascript`) — served as an ES module, `<script type="module">`/`import` |
| `vendor/wllama/wasm/wllama.wasm` | `esm/wasm/wllama.wasm` | ~8.06 MB | `application/wasm` — required for `WebAssembly.instantiateStreaming`; if a static server serves it with the wrong type, wllama falls back to a slower `ArrayBuffer` compile path (not a hard failure, just slower) |

Nothing else from the package needs to be served — no separate worker
script (see "self-contained bundle" above), no `.d.ts` files (types only,
not fetched at runtime), no CDN fallback file (`esm/wasm-from-cdn.js`) is
used.

Not vendored, served from elsewhere by whatever wires up `index.html`
(explicitly out of scope for this file per the task boundaries):
- `baselines/qwen/models/qwen3.5-0.8b-condB-q8.gguf` (~812 MB;
  `.gguf` → `application/octet-stream` is fine, browsers don't care as
  long as `Content-Length`/range requests work for progress tracking)
- `baselines/qwen/agent_core.gbnf` (plaintext; `text/plain` is fine —
  `llm.js` just does `fetch(grammarUrl).then(r => r.text())`)

If multi-threaded WASM is wanted later, the static server must also add
`Cross-Origin-Embedder-Policy: require-corp` and
`Cross-Origin-Opener-Policy: same-origin` response headers (see WebGPU /
multi-threading section above) — that's a server config concern for
whichever step owns `index.html` + the dev server, not something `llm.js`
can set itself.

## Verification performed without a browser

- `npm install @wllama/wllama` → resolved `3.6.1`, confirmed via
  `node_modules/@wllama/wllama/package.json`.
- Read `README.md`, all `esm/*.d.ts` (`wllama.d.ts`, `types/types.d.ts`,
  `types/oai-compat.d.ts`, `model-manager.d.ts`, `cache-manager.d.ts`,
  `worker.d.ts`, `wasm-from-cdn.d.ts`), and `guides/intro-v3.md` /
  `guides/intro-v3.1.md` directly from the installed package (not fetched
  from GitHub — the installed copy already had everything needed and is
  guaranteed to match the installed version).
- Grepped the bundled `esm/index.js` runtime directly to confirm behavior
  the `.d.ts` files don't fully pin down: worker creation via Blob URL
  (no separate worker file), `absoluteUrl()`'s resolution base, the
  `useMultiThread` / `isSupportMultiThread()` gate, and
  `isSupportWebGPU() === !!navigator.gpu`.
- `node --check src/llm.js` and `node --check vendor/wllama/index.js` —
  both pass (no syntax errors).
- Dynamic `import('./src/llm.js')` in Node — resolves cleanly, confirms
  the relative import path into `vendor/wllama/index.js` is correct and
  `createPlanner` is exported as a function.
- Constructing `new Wllama({ default: <url> })` in Node throws
  `Error: No supported storage backend found` from `CacheManager` — this
  is **expected**: wllama's cache layer requires a browser storage backend
  (IndexedDB/OPFS), which Node doesn't provide. This confirms the module
  loads and executes real wllama code (not a stub), failing for an
  environmental reason outside `llm.js`'s control, exactly as anticipated
  by the task's "no browser available" scoping.
- Confirmed `baselines/qwen/models/qwen3.5-0.8b-condB-q8.gguf` exists on
  disk (~812 MB) and `baselines/qwen/agent_core.gbnf` exists — did not
  download/read the full GGUF (only `ls -la` for size).
- Did not run an actual grammar-constrained generation end-to-end — no
  browser in this sandbox. That remains the one open item for whoever
  wires up `index.html`/the dev server next.
