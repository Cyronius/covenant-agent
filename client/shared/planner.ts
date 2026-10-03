// The demo's one planner: POST /plan on server/dev_server.py. The server
// decides which checkpoint that is (DEMO_PLANNER, the newest tiny planner),
// so there is no picker and nothing to choose here — every world always runs
// the same planner (.claude/plans/rpg-exits-perception.md part 1).
//
// A tiny planner reads the task (request + context); a GGUF planner, which
// the server can still be started with for an experiment, reads the prompt
// text. generate() sends both and the server uses whichever it needs.

/** The task behind a prompt. `registers` / `pauseTypes` are what /validate
 *  returned at the last PAUSE (its `registers` and `pause_envs[0]`). */
export interface PlanTask {
  request: string;
  context: unknown;
  registers?: unknown;
  pauseTypes?: Record<string, string> | null;
}

export interface PlanResult {
  text: string;
  tokensOut: number;
  genMs: number;
  /** tiny planner only: how long the request was, and whether it went past
   *  the planner's request budget (the tail is then never read). */
  requestTokens?: number;
  truncated?: boolean;
}

export interface Planner {
  /** the checkpoint the server runs, e.g. `tiny:clt_RD` */
  model: string;
  loadMs: number;
  generate(
    promptText: string,
    options: { maxTokens?: number; stop?: string[]; grammar?: string; task: PlanTask }
  ): Promise<PlanResult>;
}

interface PlanStatus {
  available: boolean;
  model: string | null;
  reason?: string;
}

/** Loads the server's planner (the first /plan call would otherwise pay for
 *  it) and returns a handle to it. */
export async function createPlanner(): Promise<Planner> {
  const t0 = performance.now();
  const res = await fetch('/plan/status');
  const status: PlanStatus = res.ok
    ? await res.json()
    : { available: false, model: null, reason: `${res.status} ${res.statusText}` };
  if (!status.available || !status.model) {
    throw new Error(`no planner on the server: ${status.reason ?? 'none configured'}`);
  }
  const warm = await fetch('/plan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ warm: true }),
  });
  if (!warm.ok) throw new Error(`planner load failed: ${warm.status} ${warm.statusText}`);
  const model = status.model;

  return {
    model,
    loadMs: performance.now() - t0,
    async generate(promptText, { maxTokens = 250, stop = [], grammar, task }) {
      const genStart = performance.now();
      const res = await fetch('/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // `grammar` is this request's own, built from its symbol table
        body: JSON.stringify({
          prompt: promptText,
          max_tokens: maxTokens,
          stop,
          grammar,
          request: task.request,
          context: task.context,
          registers: task.registers ?? null,
          pause_types: task.pauseTypes ?? null,
        }),
      });
      const body = await res.json().catch(() => null);
      if (!res.ok || !body || body.error) {
        throw new Error(`POST /plan failed: ${body?.error?.message ?? `${res.status} ${res.statusText}`}`);
      }
      return {
        text: body.text as string,
        tokensOut: body.tokens_out as number,
        // the round trip is what the person waited for, not the server's
        // generation time alone
        genMs: performance.now() - genStart,
        requestTokens: body.request_tokens,
        truncated: body.truncated,
      };
    },
  };
}
