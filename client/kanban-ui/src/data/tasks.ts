// Copied verbatim from client/poc/fixtures/tasks.json (itself exported from
// data/curriculum_tasks.jsonl by client/poc/tools/export_fixtures.py) — these
// three real kanban-world eval tasks supply the exact TOOLS/FIELDS/CONSTANTS
// catalog the grammar-constrained model was tuned against, and the exact
// REQUEST text each of our three chat turns sends. Do not hand-edit; if these
// need to change, re-export from client/poc/fixtures/tasks.json.

export interface KanbanTask {
  id: string;
  request: string;
  inputText: string;
}

export const TASKS: Record<string, KanbanTask> = {
  "L0_kanban_delete": {
    id: "L0_kanban_delete",
    request: "Delete card 4.",
    inputText: "REQUEST: Delete card 4.\nTOOLS:\nT0 (F1:ID:card F10:ID:user) -> OBJ:card [WRITE] :: Assign a card to a user.\nT1 (F1:ID:card) -> OBJ:card [WRITE] :: Archive a card so it no longer shows on the board.\nT2 () -> LIST OBJ:user [READ] :: List every user on the board.\nT3 (F5:ID:user) -> OBJ:user [READ] :: Fetch a single user by id.\nT4 (F1:ID:card F11:STR) -> OBJ:card [WRITE] :: Set a card's workflow status (todo, doing, or done).\nT5 () -> LIST OBJ:card [READ] :: List every card on the board, archived ones included.\nT6 (F6:STR F0:TIME F10:ID:user) -> OBJ:card [WRITE] :: Create a new card with a title, a due date, and an assignee.\nT7 (F5:ID:user F4:STR) -> - [SEND] :: Send a direct message to a user.\nT8 (F1:ID:card) -> OBJ:card [READ] :: Fetch a single card by its id.\nT9 (F1:ID:card) -> - [DELETE] :: Permanently delete a card. This cannot be undone.\nFIELDS:\nF0 card TIME :: card.due\nF1 card ID:card :: card.id\nF2 user STR :: user.name\nF3 card BOOL :: card.archived\nF4 - STR :: message text\nF5 user ID:user :: user.id\nF6 card STR :: card.title\nF7 card TIME :: card.created\nF8 card BOOL :: card.urgent\nF9 user STR :: user.email\nF10 card ID:user :: card.assignee\nF11 card STR :: card.status\nCONSTANTS:\nC0 ID:card :: card 4\n",
  },
  "L3_kanban_bob_overdue": {
    id: "L3_kanban_bob_overdue",
    request: "Archive the overdue cards assigned to Bob, except the urgent ones.",
    inputText: "REQUEST: Archive the overdue cards assigned to Bob, except the urgent ones.\nTOOLS:\nT0 () -> LIST OBJ:card [READ] :: List every card on the board, archived ones included.\nT1 () -> LIST OBJ:user [READ] :: List every user on the board.\nT2 (F9:ID:card F7:ID:user) -> OBJ:card [WRITE] :: Assign a card to a user.\nT3 (F9:ID:card) -> - [DELETE] :: Permanently delete a card. This cannot be undone.\nT4 (F0:ID:user) -> OBJ:user [READ] :: Fetch a single user by id.\nT5 (F0:ID:user F3:STR) -> - [SEND] :: Send a direct message to a user.\nT6 (F9:ID:card) -> OBJ:card [READ] :: Fetch a single card by its id.\nT7 (F9:ID:card F6:STR) -> OBJ:card [WRITE] :: Set a card's workflow status (todo, doing, or done).\nT8 (F8:STR F2:TIME F7:ID:user) -> OBJ:card [WRITE] :: Create a new card with a title, a due date, and an assignee.\nT9 (F9:ID:card) -> OBJ:card [WRITE] :: Archive a card so it no longer shows on the board.\nFIELDS:\nF0 user ID:user :: user.id\nF1 card BOOL :: card.urgent\nF2 card TIME :: card.due\nF3 - STR :: message text\nF4 card TIME :: card.created\nF5 user STR :: user.email\nF6 card STR :: card.status\nF7 card ID:user :: card.assignee\nF8 card STR :: card.title\nF9 card ID:card :: card.id\nF10 user STR :: user.name\nF11 card BOOL :: card.archived\nCONSTANTS:\nC0 ID:user :: Bob\nC1 BOOL :: true\n",
  },
  "L7_kanban_parallel_check": {
    id: "L7_kanban_parallel_check",
    request: "Fetch the board and Bob's profile together; if Bob has any overdue cards, send him a heads-up.",
    inputText: "REQUEST: Fetch the board and Bob's profile together; if Bob has any overdue cards, send him a heads-up.\nTOOLS:\nT0 () -> LIST OBJ:user [READ] :: List every user on the board.\nT1 (F7:ID:card) -> OBJ:card [READ] :: Fetch a single card by its id.\nT2 (F7:ID:card F10:STR) -> OBJ:card [WRITE] :: Set a card's workflow status (todo, doing, or done).\nT3 (F8:ID:user F1:STR) -> - [SEND] :: Send a direct message to a user.\nT4 () -> LIST OBJ:card [READ] :: List every card on the board, archived ones included.\nT5 (F8:ID:user) -> OBJ:user [READ] :: Fetch a single user by id.\nT6 (F7:ID:card) -> OBJ:card [WRITE] :: Archive a card so it no longer shows on the board.\nT7 (F7:ID:card F3:ID:user) -> OBJ:card [WRITE] :: Assign a card to a user.\nT8 (F11:STR F9:TIME F3:ID:user) -> OBJ:card [WRITE] :: Create a new card with a title, a due date, and an assignee.\nT9 (F7:ID:card) -> - [DELETE] :: Permanently delete a card. This cannot be undone.\nFIELDS:\nF0 user STR :: user.email\nF1 - STR :: message text\nF2 card TIME :: card.created\nF3 card ID:user :: card.assignee\nF4 user STR :: user.name\nF5 card BOOL :: card.urgent\nF6 card BOOL :: card.archived\nF7 card ID:card :: card.id\nF8 user ID:user :: user.id\nF9 card TIME :: card.due\nF10 card STR :: card.status\nF11 card STR :: card.title\nCONSTANTS:\nC0 ID:user :: Bob\nC1 STR :: heads-up text\n",
  },
};

