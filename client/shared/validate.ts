// Thin client for POST /validate — see server/README.md for the
// full contract. Proxied through Vite's dev server to the running
// `python server/dev_server.py` (see ../../vite.config.ts).
import type { Registers } from './prompt';

/** The JSON form of core.ir.TaskContext, as POST /kanban_prompt and
 *  POST /rpg_prompt return it. Shared by every world. */
export interface ToolParamDecl {
  sym: string;
  type: string;
  required: boolean;
  desc: string;
}

export interface ToolDecl {
  sym: string;
  name: string;
  desc: string;
  params: ToolParamDecl[];
  returns: string | null;
  effects: string[];
}

export interface FieldDecl {
  sym: string;
  entity: string | null;
  name: string;
  type: string;
  desc: string;
}

export interface ConstDecl {
  /** C<n> (spec 0.3.x) or S/N/B/D/I<n> (0.4.0 typed letters). */
  sym: string;
  type: string;
  value: unknown;
  desc: string;
  /** 0.4.0 §2.2 string kind: "name" | "text" | "enum:<entity>.<field>". */
  kind?: string;
  /** position in the task's constants list; absent for schema enum values. */
  index?: number | null;
}

export interface TaskContextJson {
  tools: ToolDecl[];
  fields: FieldDecl[];
  constants: ConstDecl[];
  initial_registers?: Record<string, string>;
}

export interface CallLogEntry {
  tool: string; // Tn symbol
  name: string; // tool name, e.g. "delete_card"
  args: unknown[];
  ok: boolean;
  error: { code: string; message?: string } | null;
}

export interface ValidateRequest<TState = unknown> {
  // Either task_id (one of the three fixed curriculum tasks) OR
  // context+world (as returned by POST /kanban_prompt, for a freely-typed
  // request) — see server/README.md's `/validate` section.
  task_id?: string;
  context?: TaskContextJson;
  world?: string;
  now?: number;
  text: string;
  state: TState | null;
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

export interface ValidateResponse<TState = unknown> {
  status:
    | 'ok'
    | 'paused'
    | 'aborted'
    | 'static_error'
    | 'effect_blocked'
    | 'error'
    | 'server_error'
    | string;
  diagnostics: string[];
  final_state: TState | null;
  registers: Registers;
  calls: CallLogEntry[];
  return_value: unknown;
  pause_envs: Record<string, string>[] | null;
  /** On effect_blocked, `args` are the blocked call's actual (narrowed) params —
   * e.g. the drafted message text — so the approval gate can preview them.
   * Code `BULK_WRITE` is the other shape: the whole unapproved run finished
   * and wrote to more records than the run's `bulk_write_limit`, so `calls`
   * carries every write it would make and there is no single `tool`. */
  error: {
    code: string;
    message?: string;
    tool?: string;
    effect?: string;
    args?: unknown[];
    count?: number;
    calls?: { tool: string; name: string; args: unknown[] }[];
  } | null;
  /** ABORT reason (spec §4) when status === 'aborted'. */
  reason?: 'NOT_FOUND' | 'AMBIGUOUS' | 'UNSUPPORTED' | 'NEEDS_INFO' | string | null;
  /** ABORT referents (spec §4 0.3.0): the T/F/C symbols the reason is about,
   * so the UI can ask for the thing by its description instead of showing
   * the enum. Empty for a bare abort. */
  refs?: string[];
}

export async function validate<TState = unknown>(
  req: ValidateRequest<TState>
): Promise<ValidateResponse<TState>> {
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
  return body as ValidateResponse<TState>;
}
