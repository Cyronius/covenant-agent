// Client for POST /db_prompt — the Database Analyst's analogue of
// client/app/src/worlds/kanban/lib/kanbanPrompt.ts. See server/README.md.
import type { CrmState } from '../data/crm';
import type { TaskContextJson } from '../../../../../shared/validate';

export type DbContext = TaskContextJson;

export interface DbPromptResponse {
  input_text: string;
  context: DbContext;
  world: string;
  now: number;
  grammar: string;
  system: string;
}

export async function fetchDbPrompt(request: string, state: CrmState): Promise<DbPromptResponse> {
  const res = await fetch('/db_prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request, state }),
  });
  const body = await res.json().catch(() => null);
  if (!body || body.error) {
    throw new Error(body?.error?.message ?? `POST /db_prompt failed: ${res.status}`);
  }
  return body as DbPromptResponse;
}

/** Same idea as kanbanPrompt.ts's describeCallSite: pulls the operand tokens
 * off the first `CALL <sym> ...` line and resolves any constant tokens
 * against the context, for the approval-gate preview. */
export function describeCallSite(programText: string, sym: string, context: DbContext): string[] {
  const m = programText.match(new RegExp(`CALL ${sym}\\b([^\\n]*)`));
  if (!m) return [];
  const operandPart = m[1].split(' -> ')[0].trim();
  if (!operandPart) return [];
  return operandPart.split(/\s+/).map((token) => {
    const cm = token.match(/^[A-Z]\d+$/);
    if (!cm) return token;
    const c = context.constants.find((x) => x.sym === token);
    return c ? String(c.value) : token;
  });
}
