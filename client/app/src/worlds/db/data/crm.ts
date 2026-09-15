// Types for runtime/worlds/crm.py's WORLD["entities"]. Unlike kanban's board
// (client-authored, because kanban.py has no default_state) this world's
// fake data lives entirely in Python — POST /db_new hands back a copy of
// crm.py's own default_state, so there's nothing to mirror here.
export interface CrmUser {
  id: string;
  name: string;
  email: string;
}

export interface CrmCustomer {
  id: string;
  name: string;
  email: string;
  manager: string;
  delinquent: boolean;
  plan: string;
  signup: number;
}

export interface CrmTicket {
  id: string;
  customer: string;
  status: string;
  priority: number;
  opened: number;
}

export interface CrmInvoice {
  id: string;
  customer: string;
  amount: number;
  paid: boolean;
  due: number;
}

export interface CrmState {
  entities: {
    user: CrmUser[];
    customer: CrmCustomer[];
    ticket: CrmTicket[];
    invoice: CrmInvoice[];
  };
  outbox: unknown[];
  payments: unknown[];
}

export type Effect = 'READ' | 'WRITE' | 'DELETE' | 'SEND' | 'PAY';

// Tool name -> effect, mirroring runtime/worlds/crm.py's WORLD["tools"] 1:1 —
// same reasoning as board.ts's TOOL_EFFECTS: fixed infrastructure the demo
// shares, cheaper to hand-write once than fetch and re-derive per request.
export const TOOL_EFFECTS: Record<string, Effect> = {
  list_customers: 'READ',
  get_customer: 'READ',
  list_tickets: 'READ',
  list_invoices: 'READ',
  list_staff: 'READ',
  get_staff: 'READ',
  close_ticket: 'WRITE',
  set_priority: 'WRITE',
  set_delinquent: 'WRITE',
  delete_ticket: 'DELETE',
  send_email: 'SEND',
  send_invoice: 'SEND',
  charge_customer: 'PAY',
};

export async function fetchDbState(): Promise<CrmState> {
  const res = await fetch('/db_new', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  const body = await res.json().catch(() => null);
  if (!body || body.error) {
    throw new Error(body?.error?.message ?? `POST /db_new failed: ${res.status}`);
  }
  return body.state as CrmState;
}

export function customerById(state: CrmState, id: string): CrmCustomer | undefined {
  return state.entities.customer.find((c) => c.id === id);
}

export function ticketById(state: CrmState, id: string): CrmTicket | undefined {
  return state.entities.ticket.find((t) => t.id === id);
}

export function invoiceById(state: CrmState, id: string): CrmInvoice | undefined {
  return state.entities.invoice.find((i) => i.id === id);
}

export function userById(state: CrmState, id: string): CrmUser | undefined {
  return state.entities.user.find((u) => u.id === id);
}
