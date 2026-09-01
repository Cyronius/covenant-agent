// Thin client for POST /validate — see client/poc/server/README.md for the
// full contract. Proxied through Vite's dev server to the running
// `python client/poc/server/dev_server.py` (see ../../vite.config.ts).
import type { KanbanState } from '../data/board';
import type { Registers } from './prompt';
import type { KanbanContext } from './kanbanPrompt';

export interface CallLogEntry {
  tool: string; // Tn symbol
  name: string; // tool name, e.g. "delete_card"
  args: unknown[];
  ok: boolean;
  error: { code: string; message?: string } | null;
}

export interface ValidateRequest {
  // Either task_id (one of the three fixed curriculum tasks) OR
  // context+world (as returned by POST /kanban_prompt, for a freely-typed
  // request) — see client/poc/server/README.md's `/validate` section.
  task_id?: string;
  context?: KanbanContext;
  world?: string;
  now?: number;
  text: string;
  state: KanbanState | null;
  registers: Registers;
  pause_types: Record<string, string> | null;
  /**
   * Optional override of the default approval token. `false` deliberately
   * runs unapproved so the real runtime effect gate (runtime/sandbox.js)
   * blocks the first DELETE/SEND/PAY call with a genuine EFFECT_BLOCKED;
   * omit to use the default (the fixed tasks' own stored value, or `false`
   * with an inline context, which has none).
   */
  approval?: boolean;
}

export interface ValidateResponse {
  status:
    | 'ok'
    | 'paused'
    | 'static_error'
    | 'effect_blocked'
    | 'error'
    | 'server_error'
    | string;
  diagnostics: string[];
  final_state: KanbanState | null;
  registers: Registers;
  calls: CallLogEntry[];
  return_value: unknown;
  pause_envs: Record<string, string>[] | null;
  error: { code: string; message?: string; tool?: string; effect?: string } | null;
}

export async function validate(req: ValidateRequest): Promise<ValidateResponse> {
  const res = await fetch('/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  // dev_server.py returns a JSON body even on its one failure path (HTTP 500
  // "server_error") — surface that instead of a bare network error where we
  // can.
  const body = await res.json().catch(() => null);
  if (!body) {
    throw new Error(`POST /validate failed: ${res.status} ${res.statusText}`);
  }
  return body as ValidateResponse;
}
