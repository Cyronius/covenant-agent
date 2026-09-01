// Turns a real call-log entry (runtime/sandbox.js's {tool, name, args, ok,
// error}) into a short human-readable detail string, resolving card_N /
// user_N ids against whichever board snapshot has them. No fabricated
// content — every word here traces back to the sandbox's own return value.
import type { KanbanState } from '../data/board';
import { cardById, userById } from '../data/board';
import type { CallLogEntry } from './validate';

function describeArg(v: unknown, before: KanbanState, after: KanbanState): string {
  if (typeof v === 'string') {
    if (v.startsWith('card_')) {
      const c = cardById(after, v) ?? cardById(before, v);
      return c ? `#${v.replace('card_', '')} "${c.title}"` : v;
    }
    if (v.startsWith('user_')) {
      const u = userById(after, v) ?? userById(before, v);
      return u ? u.name : v;
    }
    return `"${v}"`;
  }
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v);
}

export function describeCall(
  entry: CallLogEntry,
  before: KanbanState,
  after: KanbanState
): string {
  if (!entry.ok) {
    return entry.error ? `blocked — ${entry.error.code}` : 'failed';
  }
  switch (entry.name) {
    case 'list_cards': {
      const n = after.entities.card.filter((c) => !c.archived).length;
      return `${n} card${n === 1 ? '' : 's'} on the board`;
    }
    case 'list_users':
      return `${after.entities.user.length} team members`;
    default:
      return entry.args.map((a) => describeArg(a, before, after)).join(' · ') || '—';
  }
}
