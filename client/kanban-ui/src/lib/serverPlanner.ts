// Server-side inference (plan s2-consolidated-program §A7): the same Planner
// contract as the in-browser wllama planner (./llm.ts), backed by
// POST /plan on server/dev_server.py, which runs the same GGUF + grammar
// through llama-cpp-python on the server's CPU. No model download, no
// WASM/WebGPU — the "no hosted GPU" deployment shape. The prompt is built
// client-side exactly as for the browser path, so both paths send the model
// byte-identical text.
import type { Planner } from './llm';

export interface ServerPlanStatus {
  available: boolean;
  model: string | null;
  loaded: boolean;
  reason?: string;
}

export async function fetchServerPlanStatus(): Promise<ServerPlanStatus> {
  const res = await fetch('/plan/status');
  if (!res.ok) return { available: false, model: null, loaded: false, reason: `${res.status} ${res.statusText}` };
  return (await res.json()) as ServerPlanStatus;
}

export async function createServerPlanner(): Promise<Planner> {
  const t0 = performance.now();
  const status = await fetchServerPlanStatus();
  if (!status.available) {
    throw new Error(`server inference unavailable: ${status.reason ?? 'no model configured on the server'}`);
  }
  // First call loads the model server-side; do it now so the loading view
  // covers it instead of the first request.
  const warm = await fetch('/plan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ warm: true }),
  });
  if (!warm.ok) throw new Error(`server model load failed: ${warm.status} ${warm.statusText}`);
  const loadMs = performance.now() - t0;

  return {
    backend: 'server',
    loadMs,
    async unload() {
      /* nothing held client-side */
    },
    async generate(promptText, { maxTokens = 250, stop = [] } = {}) {
      const genStart = performance.now();
      const res = await fetch('/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText, max_tokens: maxTokens, stop }),
      });
      const body = await res.json().catch(() => null);
      if (!res.ok || !body || body.error) {
        throw new Error(`POST /plan failed: ${body?.error?.message ?? `${res.status} ${res.statusText}`}`);
      }
      return {
        text: body.text as string,
        tokensOut: body.tokens_out as number,
        // server-measured generation time; the round trip is what the person
        // waited for, so report that instead
        genMs: performance.now() - genStart,
      };
    },
  };
}
