// Turns a real call-log entry into a short human-readable detail string,
// resolving customer_N/ticket_N/invoice_N/user_N ids against whichever
// state snapshot has them. Mirrors client/app/src/worlds/kanban/lib/describe.ts.
import type { CrmState } from '../data/crm';
import { customerById, ticketById, invoiceById, userById } from '../data/crm';
import type { CallLogEntry } from '../../../../../shared/validate';

const PREFIXES: [string, (s: CrmState, id: string) => { name?: string; id: string } | undefined][] = [
  ['customer_', customerById],
  ['ticket_', ticketById],
  ['invoice_', invoiceById],
  ['user_', userById],
];

export function describeArg(v: unknown, before: CrmState, after: CrmState): string {
  if (typeof v === 'string') {
    for (const [prefix, lookup] of PREFIXES) {
      if (v.startsWith(prefix)) {
        const rec = lookup(after, v) ?? lookup(before, v);
        const num = v.slice(prefix.length);
        if (!rec) return v;
        const label = `${prefix.slice(0, -1)} ${num}`;
        return 'name' in rec && rec.name ? `${label} "${rec.name}"` : label;
      }
    }
    return `"${v}"`;
  }
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v);
}

export function describeCall(entry: CallLogEntry, before: CrmState, after: CrmState): string {
  if (!entry.ok) {
    return entry.error ? `blocked — ${entry.error.code}` : 'failed';
  }
  switch (entry.name) {
    case 'list_customers':
      return `${before.entities.customer.length} customers`;
    case 'list_tickets':
      return `${before.entities.ticket.length} tickets`;
    case 'list_invoices':
      return `${before.entities.invoice.length} invoices`;
    case 'list_staff':
      return `${before.entities.user.length} staff`;
    default:
      return entry.args.map((a) => describeArg(a, before, after)).join(' · ') || '—';
  }
}
