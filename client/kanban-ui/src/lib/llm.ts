// Ported from client/poc/src/llm.js — same @wllama/wllama contract, converted
// to TS. See client/poc/src/llm.md for the API research notes behind this
// file; nothing about the wllama usage itself changed in the port.
//
// The vendored bundle lives in src/vendor/wllama/index.js (NOT public/) and
// is imported normally so Vite bundles it — Vite refuses to import anything
// under public/ from application code ("this file is in /public ... should
// not be imported from source code"), and since the bundle is a
// self-contained ESM with zero internal imports (see llm.md), bundling it
// is harmless. Only the wasm binary stays a public/ static asset,
// referenced purely as a URL string (never imported).
// @ts-expect-error — no type declarations shipped alongside the vendored JS
import { Wllama } from '../vendor/wllama/index.js';

const WLLAMA_WASM_URL = '/vendor/wllama/wasm/wllama.wasm';

// Matches baselines/qwen/run_a.py's default --ctx 4096.
const DEFAULT_N_CTX = 4096;

export interface Planner {
  backend: 'wasm' | 'webgpu';
  loadMs: number;
  generate(
    promptText: string,
    options?: { maxTokens?: number; stop?: string[] }
  ): Promise<{ text: string; tokensOut: number; genMs: number }>;
  /** Frees the loaded model/WASM runtime. Call before creating a new
   * Planner against the same model (e.g. a CPU-only toggle reload) —
   * wllama's OPFS-backed model cache can't have two concurrent access
   * handles open on the same file. */
  unload(): Promise<void>;
}

export type LoadStage = 'grammar' | 'model';

export interface CreatePlannerOptions {
  modelUrl: string;
  grammarUrl: string;
  /** thread count override; omit to let wllama pick. */
  nThreads?: number;
  /** 0 forces WASM-only decoding (skips WebGPU offload entirely) — the
   * "CPU only" toggle. Omit to let wllama pick (WebGPU when available). */
  nGpuLayers?: number;
  /** called right before each stage starts — wllama exposes no real byte-level
   * progress (see llm.md), so this is staging only, not a percentage. */
  onStage?: (stage: LoadStage) => void;
}

export async function createPlanner({
  modelUrl,
  grammarUrl,
  nThreads,
  nGpuLayers,
  onStage,
}: CreatePlannerOptions): Promise<Planner> {
  if (!modelUrl) throw new Error('createPlanner: modelUrl is required');
  if (!grammarUrl) throw new Error('createPlanner: grammarUrl is required');

  onStage?.('grammar');
  const grammarRes = await fetch(grammarUrl);
  if (!grammarRes.ok) {
    throw new Error(
      `createPlanner: failed to fetch grammar from ${grammarUrl}: ` +
        `${grammarRes.status} ${grammarRes.statusText}`
    );
  }
  const grammarText = await grammarRes.text();

  const wllama = new Wllama({ default: WLLAMA_WASM_URL });

  // Capability check only — see llm.md: wllama does not expose which
  // backend a given completion call actually dispatched to.
  const webgpuSupported: boolean = wllama.isSupportWebGPU();

  onStage?.('model');
  const loadStart = performance.now();
  await wllama.loadModelFromUrl(modelUrl, {
    n_ctx: DEFAULT_N_CTX,
    ...(nThreads != null ? { n_threads: nThreads } : {}),
    ...(nGpuLayers != null ? { n_gpu_layers: nGpuLayers } : {}),
  });
  const loadMs = performance.now() - loadStart;

  // nGpuLayers === 0 forced WASM regardless of WebGPU support (see
  // n_gpu_layers doc comment above) — report what actually happened, not
  // just what the hardware could have done.
  const backend: 'wasm' | 'webgpu' = nGpuLayers === 0 ? 'wasm' : webgpuSupported ? 'webgpu' : 'wasm';

  return {
    backend,
    loadMs,
    async unload() {
      await wllama.exit();
    },
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

      const choice = res.choices?.[0];
      return {
        text: choice ? choice.text : '',
        tokensOut: res.usage ? res.usage.completion_tokens : 0,
        genMs,
      };
    },
  };
}
