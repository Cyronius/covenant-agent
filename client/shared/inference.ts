// Where generation runs, and on which checkpoint. Shared by every demo app
// (client/kanban-ui, client/rpg-ui) so the two behave identically.
//
// Three modes, one preference. `Planner.backend` was already
// 'wasm' | 'webgpu' | 'server' (./llm.ts); what the apps had was a two-way
// browser/server switch plus a separate "CPU only" checkbox, which is the
// same three states wearing a disguise:
//
//   server  POST /plan — llama.cpp on the dev server's CPU. No download.
//   wasm    in-browser, n_gpu_layers: 0 — multi-threaded WASM only.
//   webgpu  in-browser, WebGPU offload when the browser supports it.
//
// Neither browser mode is reliably faster than the other: WebGPU dispatch
// for a small quantized model can lose to plain WASM on a given laptop, so
// this is "try both", not "the fast one".
import { createPlanner, type LoadStage, type Planner } from './llm';
import { createServerPlanner } from './serverPlanner';

export type InferenceMode = 'server' | 'wasm' | 'webgpu';

export const INFERENCE_MODES: { mode: InferenceMode; label: string; hint: string }[] = [
  { mode: 'server', label: 'Server', hint: 'llama.cpp on the dev server CPU — no model download' },
  { mode: 'wasm', label: 'WASM', hint: 'in-browser, CPU only (n_gpu_layers 0)' },
  { mode: 'webgpu', label: 'WebGPU', hint: 'in-browser, GPU offload where supported' },
];

export interface ModelInfo {
  name: string;
  /** 'qwen': the client builds the prompt (our tuned checkpoints were SFT'd
   *  on that exact markup). 'chat': the server applies the GGUF's own
   *  template — any other instruct model. */
  template: 'qwen' | 'chat';
  tuned: boolean;
  size_mb: number;
}

export interface ModelList {
  default: string | null;
  loaded: string | null;
  models: ModelInfo[];
}

/** GET /models — what checkpoints this dev server has on disk. */
export async function fetchModels(): Promise<ModelList> {
  const res = await fetch('/models');
  if (!res.ok) throw new Error(`GET /models failed: ${res.status} ${res.statusText}`);
  return (await res.json()) as ModelList;
}

/** Per-app persisted preference. Browser storage can throw outright (private
 *  mode, blocked site data), so every read and write is guarded and the app
 *  still works with nothing stored. */
export function loadPreference(appKey: string, fallback: InferenceMode = 'server'): InferenceMode {
  try {
    const raw = localStorage.getItem(`${appKey}:inference`);
    if (raw === 'server' || raw === 'wasm' || raw === 'webgpu') return raw;
  } catch {
    /* unavailable — use the fallback for this session */
  }
  return fallback;
}

export function savePreference(appKey: string, mode: InferenceMode): void {
  try {
    localStorage.setItem(`${appKey}:inference`, mode);
  } catch {
    /* unavailable — the choice still applies for this session */
  }
}

export function loadModelPreference(appKey: string): string | null {
  try {
    return localStorage.getItem(`${appKey}:model`);
  } catch {
    return null;
  }
}

export function saveModelPreference(appKey: string, model: string): void {
  try {
    localStorage.setItem(`${appKey}:model`, model);
  } catch {
    /* unavailable */
  }
}

export interface CreateForOptions {
  mode: InferenceMode;
  /** GGUF filename as GET /models lists it. */
  model: string;
  grammarUrl?: string;
  onStage?: (stage: LoadStage) => void;
}

/** Build a planner for a mode + checkpoint. `n_gpu_layers` is a load-time
 *  setting, so switching either one means loading again — there is no
 *  in-place swap. */
export async function createPlannerFor({
  mode,
  model,
  grammarUrl = '/agent_core.gbnf',
  onStage,
}: CreateForOptions): Promise<Planner> {
  if (mode === 'server') return createServerPlanner(model);
  return createPlanner({
    modelUrl: `/models/${model}`,
    grammarUrl,
    nGpuLayers: mode === 'wasm' ? 0 : undefined,
    onStage,
  });
}
