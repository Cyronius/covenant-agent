// Client for POST /kanban_prompt — see client/poc/server/README.md. Builds
// a fresh grammar-matched TOOLS/FIELDS/CONSTANTS context for the `kanban`
// world from whatever a person actually typed, against the current fake
// board, instead of picking one of a fixed set of pre-written requests.
import type { KanbanState } from '../data/board';

export interface KanbanToolParam {
  sym: string;
  type: string;
  required: boolean;
  desc: string;
}

export interface KanbanToolDecl {
  sym: string;
  name: string;
  desc: string;
  params: KanbanToolParam[];
  returns: string | null;
  effects: string[];
}

export interface KanbanFieldDecl {
  sym: string;
  entity: string | null;
  name: string;
  type: string;
  desc: string;
}

export interface KanbanConstDecl {
  sym: string;
  type: string;
  value: unknown;
  desc: string;
}

export interface KanbanContext {
  tools: KanbanToolDecl[];
  fields: KanbanFieldDecl[];
  constants: KanbanConstDecl[];
  initial_registers?: Record<string, string>;
}

export interface KanbanPromptResponse {
  input_text: string;
  context: KanbanContext;
  world: string;
  now: number;
}

export async function fetchKanbanPrompt(
  request: string,
  state: KanbanState
): Promise<KanbanPromptResponse> {
  const res = await fetch('/kanban_prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request, state }),
  });
  const body = await res.json().catch(() => null);
  if (!body || body.error) {
    throw new Error(body?.error?.message ?? `POST /kanban_prompt failed: ${res.status}`);
  }
  return body as KanbanPromptResponse;
}

/** Strips the "message: " desc prefix constants_from_board()
 * (client/poc/server/dev_server.py) adds for readability in the raw prompt.
 * Card descs are "card 3 — Title" and user descs are a plain name — both
 * already read fine as-is for a UI display. */
export function describeConstant(c: KanbanConstDecl): string {
  return c.desc.replace(/^message: /, '');
}

/** Best-effort: pulls the operand tokens off the first `CALL <sym> ...` line
 * in a generated program and resolves any Cn tokens against the context's
 * constants, for building a human-readable approval-gate preview. Reg/
 * regfield/NOW operands are shown as-is (nothing to resolve client-side
 * without executing). */
export function describeCallSite(programText: string, sym: string, context: KanbanContext): string[] {
  const m = programText.match(new RegExp(`CALL ${sym}\\b([^\\n]*)`));
  if (!m) return [];
  const operandPart = m[1].split(' -> ')[0].trim();
  if (!operandPart) return [];
  return operandPart.split(/\s+/).map((token) => {
    const cm = token.match(/^C\d+$/);
    if (!cm) return token;
    const c = context.constants.find((x) => x.sym === token);
    return c ? describeConstant(c) : token;
  });
}
