// Static mirror of runtime/worlds/kanban.py's WORLD["tools"] — same reasoning
// as TOOL_EFFECTS in ./board.ts: this is fixed infrastructure the demo's one
// world shares, not something worth a round-trip to fetch. Drives the "What
// can it do?" panel so a person opening the demo can see the agent's actual
// tool surface (name, params, effect) without reading the grammar or server
// code.
import type { Effect } from './board';

export interface ToolParam {
  name: string;
  type: string;
  desc: string;
}

export interface ToolDecl {
  name: string;
  desc: string;
  params: ToolParam[];
  returns: string | null;
  effect: Effect;
}

export const TOOLS: ToolDecl[] = [
  {
    name: 'list_cards',
    desc: 'List every card on the board, archived ones included.',
    params: [],
    returns: 'LIST OBJ:card',
    effect: 'READ',
  },
  {
    name: 'get_card',
    desc: 'Fetch a single card by its id.',
    params: [{ name: 'card', type: 'ID:card', desc: 'the card to fetch' }],
    returns: 'OBJ:card',
    effect: 'READ',
  },
  {
    name: 'create_card',
    desc: 'Create a new card with a title, a due date, and an assignee.',
    params: [
      { name: 'title', type: 'STR', desc: 'card title' },
      { name: 'due', type: 'TIME', desc: 'due date' },
      { name: 'assignee', type: 'ID:user', desc: 'user the card is assigned to' },
    ],
    returns: 'OBJ:card',
    effect: 'WRITE',
  },
  {
    name: 'set_status',
    desc: "Set a card's workflow status (todo, doing, or done).",
    params: [
      { name: 'card', type: 'ID:card', desc: 'the card' },
      { name: 'status', type: 'STR', desc: 'new status' },
    ],
    returns: 'OBJ:card',
    effect: 'WRITE',
  },
  {
    name: 'archive_card',
    desc: 'Archive a card so it no longer shows on the board.',
    params: [{ name: 'card', type: 'ID:card', desc: 'card to archive' }],
    returns: 'OBJ:card',
    effect: 'WRITE',
  },
  {
    name: 'assign_card',
    desc: 'Assign a card to a user.',
    params: [
      { name: 'card', type: 'ID:card', desc: 'the card' },
      { name: 'user', type: 'ID:user', desc: 'new assignee' },
    ],
    returns: 'OBJ:card',
    effect: 'WRITE',
  },
  {
    name: 'delete_card',
    desc: 'Permanently delete a card. This cannot be undone.',
    params: [{ name: 'card', type: 'ID:card', desc: 'card to delete' }],
    returns: null,
    effect: 'DELETE',
  },
  {
    name: 'list_users',
    desc: 'List every user on the board.',
    params: [],
    returns: 'LIST OBJ:user',
    effect: 'READ',
  },
  {
    name: 'get_user',
    desc: 'Fetch a single user by id.',
    params: [{ name: 'user', type: 'ID:user', desc: 'the user' }],
    returns: 'OBJ:user',
    effect: 'READ',
  },
  {
    name: 'write_text',
    desc: "Write a short message from a brief (the request, in the requester's words) and the cards it should mention. Returns the text — drafted by the model, never composed by the planner.",
    params: [
      { name: 'brief', type: 'STR', desc: 'what to write' },
      { name: 'data', type: 'LIST OBJ:card', desc: 'cards the message is about' },
    ],
    returns: 'STR',
    effect: 'EXTERNAL',
  },
  {
    name: 'send_message',
    desc: 'Send a direct message to a user.',
    params: [
      { name: 'user', type: 'ID:user', desc: 'recipient' },
      { name: 'text', type: 'STR', desc: 'message text' },
    ],
    returns: null,
    effect: 'SEND',
  },
];
