// Our fake board — authored data, in the exact shape
// runtime/worlds/kanban.py's WORLD["entities"] expects (see that file's
// default_state for the reference shape this mirrors), anchored to the same
// `now` all three of our tasks carry (data/curriculum_tasks.jsonl).
//
// Two ids are load-bearing, not arbitrary: "user_1" must be Bob (the
// L3/L7 tasks' constant C0 is the literal id "user_1", resolved at compile
// time from data/curriculum_tasks.jsonl — see TASKS in ./tasks.ts), and a
// card with id "card_4" must exist (L0's constant C0 is literally "card_4").
// Every other id/title/date is ours to invent.

export const NOW = 1_760_000_000; // matches task["now"] for L0/L3/L7
const DAY = 86400;

export interface KanbanUser {
  id: string;
  name: string;
  email: string;
}

export interface KanbanCard {
  id: string;
  title: string;
  status: 'todo' | 'doing' | 'done';
  assignee: string;
  due: number; // epoch seconds
  urgent: boolean;
  archived: boolean;
  created: number; // epoch seconds
}

export interface KanbanState {
  entities: {
    user: KanbanUser[];
    card: KanbanCard[];
  };
  outbox: unknown[];
  payments: unknown[];
}

export const USERS: KanbanUser[] = [
  { id: 'user_1', name: 'Bob Alvarez', email: 'bob@understory.test' },
  { id: 'user_2', name: 'Priya Nandan', email: 'priya@understory.test' },
  { id: 'user_3', name: 'Theo Marsh', email: 'theo@understory.test' },
  { id: 'user_4', name: 'Kade Whitfield', email: 'kade@understory.test' },
  { id: 'user_5', name: 'Luz Ferreira', email: 'luz@understory.test' },
];

export const INITIAL_CARDS: KanbanCard[] = [
  { id: 'card_1', title: 'Fix OAuth redirect loop on staging', status: 'doing', assignee: 'user_1', due: NOW - 3 * DAY, urgent: true, archived: false, created: NOW - 30 * DAY },
  { id: 'card_2', title: 'Write release notes for v4.2', status: 'todo', assignee: 'user_2', due: NOW + 6 * DAY, urgent: false, archived: false, created: NOW - 12 * DAY },
  { id: 'card_3', title: 'Audit tool-call pause UX for DELETE effects', status: 'doing', assignee: 'user_3', due: NOW + 2 * DAY, urgent: true, archived: false, created: NOW - 18 * DAY },
  { id: 'card_4', title: 'Retire legacy webhook handler', status: 'todo', assignee: 'user_1', due: NOW - 8 * DAY, urgent: false, archived: false, created: NOW - 40 * DAY },
  { id: 'card_5', title: 'Design empty-state illustration for board', status: 'todo', assignee: 'user_2', due: NOW + 13 * DAY, urgent: false, archived: false, created: NOW - 9 * DAY },
  { id: 'card_6', title: 'Reproduce race condition in segment resume', status: 'doing', assignee: 'user_4', due: NOW + 1 * DAY, urgent: true, archived: false, created: NOW - 15 * DAY },
  { id: 'card_7', title: 'Onboard Luz to on-call rotation', status: 'todo', assignee: 'user_5', due: NOW + 8 * DAY, urgent: false, archived: false, created: NOW - 6 * DAY },
  { id: 'card_8', title: 'Ship grammar-constrained decode benchmarks', status: 'done', assignee: 'user_4', due: NOW - 6 * DAY, urgent: false, archived: false, created: NOW - 35 * DAY },
  { id: 'card_9', title: 'Bob: quarterly perf self-review', status: 'todo', assignee: 'user_1', due: NOW - 4 * DAY, urgent: false, archived: false, created: NOW - 20 * DAY },
  { id: 'card_10', title: 'Patch prompt-injection regression in E-crowded suite', status: 'doing', assignee: 'user_3', due: NOW - 1 * DAY, urgent: true, archived: false, created: NOW - 11 * DAY },
  { id: 'card_11', title: 'Bob: renew staging TLS cert', status: 'todo', assignee: 'user_1', due: NOW - 9 * DAY, urgent: false, archived: false, created: NOW - 45 * DAY },
  { id: 'card_12', title: 'Migrate fixtures to 142-domain worldgen', status: 'done', assignee: 'user_2', due: NOW - 13 * DAY, urgent: false, archived: false, created: NOW - 50 * DAY },
  { id: 'card_13', title: 'Draft coursebuilder domain mirror spec', status: 'todo', assignee: 'user_5', due: NOW + 11 * DAY, urgent: false, archived: false, created: NOW - 4 * DAY },
  { id: 'card_14', title: 'Old kanban prototype cleanup', status: 'done', assignee: 'user_4', due: NOW - 58 * DAY, urgent: false, archived: true, created: NOW - 90 * DAY },
];

export function initialState(): KanbanState {
  return {
    entities: {
      user: USERS.map((u) => ({ ...u })),
      card: INITIAL_CARDS.map((c) => ({ ...c })),
    },
    outbox: [],
    payments: [],
  };
}

export type Effect = 'READ' | 'WRITE' | 'DELETE' | 'SEND';

// Tool name -> effect, mirroring runtime/worlds/kanban.py's WORLD["tools"]
// 1:1 (kept small and static rather than fetched, since it's fixed
// infrastructure all three tasks share).
export const TOOL_EFFECTS: Record<string, Effect> = {
  list_cards: 'READ',
  get_card: 'READ',
  create_card: 'WRITE',
  set_status: 'WRITE',
  archive_card: 'WRITE',
  assign_card: 'WRITE',
  delete_card: 'DELETE',
  list_users: 'READ',
  get_user: 'READ',
  send_message: 'SEND',
};

export function userById(state: KanbanState, id: string): KanbanUser | undefined {
  return state.entities.user.find((u) => u.id === id);
}

export function cardById(state: KanbanState, id: string): KanbanCard | undefined {
  return state.entities.card.find((c) => c.id === id);
}

export function isOverdue(card: KanbanCard, now: number): boolean {
  return card.due < now && card.status !== 'done' && !card.archived;
}
