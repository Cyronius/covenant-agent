// Client for POST /rpg_new and POST /rpg_prompt (server/README.md).
//
// The server owns both the dungeon and the perception rule: it deals the
// starting state, and each turn it renders that state into the request text
// and the constants the model may name. This app never computes what the
// model can see — it draws the same window the server reported, so the fog
// on screen is the fog in the prompt.
import type { TaskContextJson } from '../../../shared/validate';

export interface Player {
  id: string;
  x: number;
  y: number;
  hp: number;
  max_hp: number;
  attack: number;
}

export interface Enemy {
  id: string;
  kind: string;
  x: number;
  y: number;
  hp: number;
  attack: number;
}

export interface Item {
  id: string;
  kind: string;
  x: number;
  y: number;
  held: boolean;
}

export interface Door {
  id: string;
  x: number;
  y: number;
  locked: boolean;
  open: boolean;
}

export interface RpgState {
  entities: {
    player: Player[];
    enemy: Enemy[];
    item: Item[];
    door: Door[];
  };
  outbox: unknown[];
  payments: unknown[];
  map: { width: number; height: number; rows: string[] };
  scenario: string;
  quest: string;
  vision: number;
  max_turns: number;
  turn: number;
  status: 'playing' | 'won' | 'dead';
  log: string[];
  memory: string[];
  turn_budget: number;
  actions_this_turn: number;
}

export interface NearbyThing {
  id: string;
  type: 'enemy' | 'item' | 'door';
  kind?: string;
  dx: number;
  dy: number;
  adjacent?: boolean;
  here?: boolean;
  hp?: number;
  locked?: boolean;
  open?: boolean;
}

export interface Outcome {
  status: string;
  won: boolean;
  dead: boolean;
  turn: number;
  hp: number;
  key_taken: boolean;
  door_opened: boolean;
  enemies_left: number;
}

export interface RpgPromptResponse {
  input_text: string;
  context: TaskContextJson;
  world: 'rpg';
  now: number;
  /** GBNF for this turn's symbol table - typed CALL slots, so an argument of
   * the wrong type cannot be decoded. */
  grammar: string;
  /** SYSTEM text matching the surface the context was serialized in. */
  system: string;
  /** the same state, with `memory` advanced by this observation — thread
   *  this copy forward, not the one you sent */
  state: RpgState;
  observation: {
    request: string;
    window: string[];
    nearby: NearbyThing[];
    outcome: Outcome;
  };
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const parsed = await res.json().catch(() => null);
  if (!parsed || parsed.error) {
    throw new Error(parsed?.error?.message ?? `POST ${path} failed: ${res.status}`);
  }
  return parsed as T;
}

export async function newGame(scenario?: string): Promise<RpgState> {
  const body = await post<{ state: RpgState }>('/rpg_new', scenario ? { scenario } : {});
  return body.state;
}

export async function fetchRpgPrompt(state: RpgState): Promise<RpgPromptResponse> {
  return post<RpgPromptResponse>('/rpg_prompt', { state });
}

export function player(state: RpgState): Player {
  return state.entities.player[0];
}

/** Inventory, for the HUD: items carried are off-map (x/y = -1). */
export function carried(state: RpgState): Item[] {
  return state.entities.item.filter((i) => i.held);
}
