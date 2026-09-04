"use strict";
/*
 * RPG world rules (plan .claude/plans/rpg-demo-app.md).
 *
 * Loaded by runtime/sandbox.js's `engine` impl op. Every tool the model can
 * call lands in one of the exported functions; `end_turn` is the enemy phase,
 * run once per turn by the sandbox's post_hook after the program finishes.
 *
 * Contract with the sandbox:
 *   fn(state, params, rt) -> result
 *   rt = { now, ToolError, clone }
 * Functions mutate `state` in place and push a line onto `state.log`.
 *
 * VALIDATE THEN MUTATE. The sandbox returns the (mutated) state even when a
 * tool throws (runtime/sandbox.js catch), so a function that mutates before
 * discovering the action is illegal would leave a half-applied action in a
 * state the harness keeps. Every check happens before the first write.
 *
 * Deterministic: no RNG anywhere. Same program + same state => same result,
 * which is what makes paired cross-model episodes comparable.
 */

const DIRS = {
  north: [0, -1],
  south: [0, 1],
  east: [1, 0],
  west: [-1, 0],
};

const WALL = "#";
const EXIT = "E";
const AGGRO = 3;          // Manhattan distance at which an enemy engages
const POTION_HEAL = 5;
const TURN_BUDGET_DEFAULT = 3;

// ---------------------------------------------------------------- helpers

function player(state) {
  const list = (state.entities && state.entities.player) || [];
  return list[0];
}

function enemies(state) {
  return (state.entities && state.entities.enemy) || [];
}

function items(state) {
  return (state.entities && state.entities.item) || [];
}

function doors(state) {
  return (state.entities && state.entities.door) || [];
}

function tileAt(state, x, y) {
  const map = state.map || { width: 0, height: 0, rows: [] };
  if (x < 0 || y < 0 || x >= map.width || y >= map.height) return WALL;
  const row = map.rows[y] || "";
  return row[x] || WALL;
}

function enemyAt(state, x, y) {
  return enemies(state).find((e) => e.x === x && e.y === y && e.hp > 0);
}

function doorAt(state, x, y) {
  return doors(state).find((d) => d.x === x && d.y === y);
}

function adjacent(a, b) {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) === 1;
}

function log(state, text) {
  if (!Array.isArray(state.log)) state.log = [];
  state.log.push(text);
}

/** Every action goes through here first: the game must be running, the
 *  player alive, and the turn's action budget not yet spent.
 *
 *  It also clears last turn's events. `log` is what the next observation
 *  renders as "Last turn:", so it must hold exactly one turn: the player's
 *  actions plus the enemy phase that answered them. Clearing here rather
 *  than in end_turn is what keeps the enemy's moves attached to the turn
 *  they belong to — clearing at the end would drop them before they were
 *  ever shown. (Without this the log grew for the whole episode and the
 *  prompt filled up with history: caught 2026-09-04 by reading turn 7 of an
 *  oracle run.) */
function beginAction(state, rt) {
  const p = player(state);
  if (!p) throw new rt.ToolError("NOT_FOUND", "no player in this world");
  if (!(state.actions_this_turn > 0)) state.log = [];
  if (state.status === "won") {
    throw new rt.ToolError("INVALID_ARGUMENT", "the dungeon is already cleared");
  }
  if (state.status === "dead" || p.hp <= 0) {
    throw new rt.ToolError("INVALID_ARGUMENT", "you are dead");
  }
  const budget = state.turn_budget === undefined
    ? TURN_BUDGET_DEFAULT : state.turn_budget;
  const spent = state.actions_this_turn || 0;
  if (spent >= budget) {
    throw new rt.ToolError("RATE_LIMITED",
      `turn budget spent (${budget} actions per turn)`);
  }
  return p;
}

/** Charged for every action that reaches the mutation stage AND for every
 *  action rejected as illegal — otherwise a TRY-wrapped program could probe
 *  the map for free. Callers invoke it after validation, before mutation;
 *  the throw paths call it explicitly. */
function spend(state) {
  state.actions_this_turn = (state.actions_this_turn || 0) + 1;
}

function fail(state, rt, code, message) {
  spend(state);
  log(state, `failed: ${message}`);
  return new rt.ToolError(code, message);
}

/** Walkable for the player: floor/exit tile, no living enemy, no closed door. */
function blockedFor(state, x, y) {
  const t = tileAt(state, x, y);
  if (t === WALL) return "a wall";
  const door = doorAt(state, x, y);
  if (door && !door.open) return door.locked ? "a locked door" : "a closed door";
  if (enemyAt(state, x, y)) return "an enemy";
  return null;
}

// ------------------------------------------------------------------ tools

function move(state, params, rt) {
  const p = beginAction(state, rt);
  const dir = String(params[0] || "").toLowerCase();
  if (!(dir in DIRS)) {
    throw fail(state, rt, "INVALID_ARGUMENT",
      `${params[0]} is not a direction (north, south, east, west)`);
  }
  const [dx, dy] = DIRS[dir];
  const nx = p.x + dx;
  const ny = p.y + dy;
  const blocker = blockedFor(state, nx, ny);
  if (blocker) {
    throw fail(state, rt, "INVALID_ARGUMENT", `cannot move ${dir}: ${blocker}`);
  }
  spend(state);
  p.x = nx;
  p.y = ny;
  log(state, `moved ${dir}`);
  if (tileAt(state, nx, ny) === EXIT) {
    state.status = "won";
    log(state, "reached the stairs down — the dungeon is cleared");
  }
  return rt.clone(p);
}

function attack(state, params, rt) {
  const p = beginAction(state, rt);
  const target = enemies(state).find((e) => e.id === params[0]);
  if (!target || target.hp <= 0) {
    throw fail(state, rt, "NOT_FOUND", `no enemy ${params[0]} here`);
  }
  if (!adjacent(p, target)) {
    throw fail(state, rt, "INVALID_ARGUMENT",
      `${target.kind} is not adjacent — step next to it first`);
  }
  spend(state);
  target.hp -= p.attack;
  if (target.hp <= 0) {
    target.hp = 0;
    state.entities.enemy = enemies(state).filter((e) => e.id !== target.id);
    log(state, `killed the ${target.kind}`);
  } else {
    log(state, `hit the ${target.kind} for ${p.attack} (${target.hp} HP left)`);
  }
  return rt.clone(target);
}

function pick_up(state, params, rt) {
  const p = beginAction(state, rt);
  const item = items(state).find((i) => i.id === params[0]);
  if (!item || item.held) {
    throw fail(state, rt, "NOT_FOUND", `no item ${params[0]} on the floor`);
  }
  if (item.x !== p.x || item.y !== p.y) {
    throw fail(state, rt, "INVALID_ARGUMENT",
      `the ${item.kind} is not on your tile — walk onto it first`);
  }
  spend(state);
  item.held = true;
  item.x = -1;
  item.y = -1;
  log(state, `picked up the ${item.kind}`);
  return rt.clone(item);
}

function use_item(state, params, rt) {
  const p = beginAction(state, rt);
  const item = items(state).find((i) => i.id === params[0]);
  if (!item) throw fail(state, rt, "NOT_FOUND", `no item ${params[0]}`);
  if (!item.held) {
    throw fail(state, rt, "INVALID_ARGUMENT",
      `you are not carrying the ${item.kind} — pick it up first`);
  }
  if (item.kind === "key") {
    throw fail(state, rt, "INVALID_ARGUMENT",
      "a key is not used on its own — interact with the locked door");
  }
  if (item.kind !== "potion") {
    throw fail(state, rt, "INVALID_ARGUMENT", `cannot use a ${item.kind}`);
  }
  spend(state);
  const healed = Math.min(POTION_HEAL, p.max_hp - p.hp);
  p.hp += healed;
  state.entities.item = items(state).filter((i) => i.id !== item.id);
  log(state, `drank the potion, healed ${healed} (${p.hp}/${p.max_hp} HP)`);
  return rt.clone(p);
}

function interact(state, params, rt) {
  const p = beginAction(state, rt);
  const door = doors(state).find((d) => d.id === params[0]);
  if (!door) throw fail(state, rt, "NOT_FOUND", `no door ${params[0]}`);
  if (!adjacent(p, door)) {
    throw fail(state, rt, "INVALID_ARGUMENT",
      "the door is not adjacent — step next to it first");
  }
  if (door.open) {
    throw fail(state, rt, "INVALID_ARGUMENT", "that door is already open");
  }
  if (door.locked) {
    const key = items(state).find((i) => i.kind === "key" && i.held);
    if (!key) {
      throw fail(state, rt, "INVALID_ARGUMENT",
        "the door is locked and you have no key");
    }
    spend(state);
    door.locked = false;
    door.open = true;
    state.entities.item = items(state).filter((i) => i.id !== key.id);
    log(state, "unlocked the door with the key and opened it");
    return rt.clone(door);
  }
  spend(state);
  door.open = true;
  log(state, "opened the door");
  return rt.clone(door);
}

// -------------------------------------------------------------- turn end

/** The enemy phase, run once per turn by the sandbox post_hook after the
 *  program ends (see the plan: acting after every call would invalidate the
 *  snapshot the model planned against). Also rolls the turn counter and
 *  clears the per-turn action budget. */
function end_turn(state) {
  const p = player(state);
  if (!p) return state;
  if (state.status === "playing" && p.hp > 0) {
    for (const e of enemies(state)) {
      if (e.hp <= 0) continue;
      const dist = Math.abs(e.x - p.x) + Math.abs(e.y - p.y);
      if (dist > AGGRO) continue;
      if (dist === 1) {
        p.hp -= e.attack;
        log(state, `the ${e.kind} hits you for ${e.attack}`);
        if (p.hp <= 0) {
          p.hp = 0;
          state.status = "dead";
          log(state, "you died");
          break;
        }
        continue;
      }
      // step one tile toward the player: larger axis first, then the other
      const dx = p.x - e.x;
      const dy = p.y - e.y;
      const tries = Math.abs(dx) >= Math.abs(dy)
        ? [[Math.sign(dx), 0], [0, Math.sign(dy)]]
        : [[0, Math.sign(dy)], [Math.sign(dx), 0]];
      for (const [sx, sy] of tries) {
        if (sx === 0 && sy === 0) continue;
        const nx = e.x + sx;
        const ny = e.y + sy;
        if (tileAt(state, nx, ny) === WALL) continue;
        const door = doorAt(state, nx, ny);
        if (door && !door.open) continue;
        if (enemyAt(state, nx, ny)) continue;
        if (nx === p.x && ny === p.y) continue;
        e.x = nx;
        e.y = ny;
        log(state, `the ${e.kind} moves closer`);
        break;
      }
    }
  }
  state.turn = (state.turn || 0) + 1;
  state.actions_this_turn = 0;
  // A turn where nothing ran never reaches beginAction's clear, so cap the
  // log rather than let a run of failed turns accumulate one.
  if (state.log && state.log.length > 12) {
    state.log = state.log.slice(-12);
  }
  return state;
}

module.exports = { move, attack, pick_up, use_item, interact, end_turn };
