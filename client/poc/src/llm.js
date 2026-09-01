// Browser inference stand-in — dev-only POC.
// See ../../.claude/plans/browser-inference-standin.md and ./llm.md for
// background and API research notes.
//
// Plain ES module, no bundler. Loads @wllama/wllama (llama.cpp compiled to
// WASM) from ../vendor/wllama/, points it at the WASM binary vendored
// alongside it, loads a GGUF model from a same-origin URL, and runs
// grammar-constrained (GBNF), temperature-0 completion.
//
// wllama v3.6.1 confirmed API used here (see llm.md for how this was
// verified against node_modules/@wllama/wllama's .d.ts / README, not memory
// of older wllama releases):
//   - new Wllama(pathConfig, wllamaConfig?)
//   - wllama.isSupportWebGPU(): boolean         (capability check only)
//   - wllama.loadModelFromUrl(url, params): Promise<void>
//   - wllama.createCompletion({ prompt, grammar, temperature, max_tokens,
//       stop, ... }): Promise<RawCompletionResponse>
//     `grammar` is a raw GBNF string, part of SamplingParams, forwarded
//     straight to llama.cpp's native grammar sampler (server-context.cpp,
//     the same engine as llama-server). Confirmed present in
//     esm/types/types.d.ts; there is no separate "grammar mode" flag to
//     enable it.

import { Wllama } from '../vendor/wllama/index.js';

// Resolve vendored asset URLs relative to THIS module's own location, not
// the page's location — robust regardless of where the HTML page that
// imports llm.js lives, as long as client/poc/'s internal directory layout
// (src/ next to vendor/) is preserved.
const WLLAMA_WASM_URL = new URL(
  '../vendor/wllama/wasm/wllama.wasm',
  import.meta.url
).href;

// Matches baselines/qwen/run_a.py's default --ctx 4096.
const DEFAULT_N_CTX = 4096;

/**
 * @param {object} opts
 * @param {string} opts.modelUrl same-origin URL to the .gguf file
 * @param {string} opts.grammarUrl same-origin URL to a plaintext GBNF grammar file
 * @param {number} [opts.nThreads] thread count override; omit to let wllama
 *   pick (defaults to floor(navigator.hardwareConcurrency / 2), and only
 *   actually uses multiple threads if the page is cross-origin-isolated —
 *   see llm.md).
 * @returns {Promise<{backend: 'wasm'|'webgpu', loadMs: number, generate: Function}>}
 */
export async function createPlanner({ modelUrl, grammarUrl, nThreads } = {}) {
  if (!modelUrl) {
    throw new Error('createPlanner: modelUrl is required');
  }
  if (!grammarUrl) {
    throw new Error('createPlanner: grammarUrl is required');
  }

  // Fetch the grammar text ourselves (contract requirement — llm.js owns
  // this fetch, callers just pass a URL).
  const grammarRes = await fetch(grammarUrl);
  if (!grammarRes.ok) {
    throw new Error(
      `createPlanner: failed to fetch grammar from ${grammarUrl}: ` +
        `${grammarRes.status} ${grammarRes.statusText}`
    );
  }
  const grammarText = await grammarRes.text();

  const wllama = new Wllama({ default: WLLAMA_WASM_URL });

  // Capability check only — v3.6.1's public API does not expose which
  // backend a given inference call actually dispatched to (see llm.md).
  // This reports whether WebGPU offload was available/attempted, not a
  // confirmed post-hoc trace of what ran.
  const webgpuSupported = wllama.isSupportWebGPU();

  const loadStart = performance.now();
  await wllama.loadModelFromUrl(modelUrl, {
    n_ctx: DEFAULT_N_CTX,
    ...(nThreads != null ? { n_threads: nThreads } : {}),
    // n_gpu_layers intentionally left unset: per the wllama v3.6.1 README,
    // WebGPU offload (all layers) is attempted automatically when
    // navigator.gpu is available. Pass n_gpu_layers: 0 to force WASM-only.
  });
  const loadMs = performance.now() - loadStart;

  const backend = webgpuSupported ? 'webgpu' : 'wasm';

  return {
    backend,
    loadMs,

    /**
     * @param {string} promptText FULL prompt including chat markup —
     *   generate() does not add a system prompt or any other markup.
     * @param {object} [options]
     * @param {number} [options.maxTokens]
     * @param {string[]} [options.stop] stop strings; matched stop text is
     *   NOT included in the returned text (llama.cpp/llama-server behavior).
     */
    async generate(promptText, { maxTokens = 250, stop = [] } = {}) {
      const genStart = performance.now();
      const res = await wllama.createCompletion({
        prompt: promptText,
        grammar: grammarText,
        temperature: 0,
        max_tokens: maxTokens,
        stop,
      });
      const genMs = performance.now() - genStart;

      const choice = res.choices[0];
      return {
        text: choice ? choice.text : '',
        tokensOut: res.usage ? res.usage.completion_tokens : 0,
        genMs,
      };
    },
  };
}
