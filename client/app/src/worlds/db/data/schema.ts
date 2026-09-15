// A hand-authored mirror of runtime/worlds/crm.py's WORLD["entities"], the
// same convention client/app/src/worlds/kanban/data/tools.ts already uses
// for ToolsPanel.tsx — fixed infrastructure, cheap to keep in sync by hand
// rather than fetch and re-derive per request. Keep in sync with crm.py.
export interface SchemaField {
  name: string;
  type: string;
}

export interface SchemaEntity {
  entity: string;
  fields: SchemaField[];
}

export const SCHEMA: SchemaEntity[] = [
  {
    entity: 'customer',
    fields: [
      { name: 'id', type: 'ID:customer' },
      { name: 'name', type: 'STR' },
      { name: 'email', type: 'STR' },
      { name: 'manager', type: 'ID:user' },
      { name: 'delinquent', type: 'BOOL' },
      { name: 'plan', type: 'STR (basic | pro | enterprise)' },
      { name: 'signup', type: 'TIME' },
    ],
  },
  {
    entity: 'ticket',
    fields: [
      { name: 'id', type: 'ID:ticket' },
      { name: 'customer', type: 'ID:customer' },
      { name: 'status', type: 'STR (open | closed)' },
      { name: 'priority', type: 'INT (1 low .. 3 high)' },
      { name: 'opened', type: 'TIME' },
    ],
  },
  {
    entity: 'invoice',
    fields: [
      { name: 'id', type: 'ID:invoice' },
      { name: 'customer', type: 'ID:customer' },
      { name: 'amount', type: 'INT (cents)' },
      { name: 'paid', type: 'BOOL' },
      { name: 'due', type: 'TIME' },
    ],
  },
  {
    entity: 'user',
    fields: [
      { name: 'id', type: 'ID:user' },
      { name: 'name', type: 'STR' },
      { name: 'email', type: 'STR' },
    ],
  },
];
