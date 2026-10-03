"use strict";
/*
 * Grid rooms rules (plan .claude/plans/borrowed-worlds.md, "3. Grid rooms").
 *
 * BabyAI's world in compass steps: rooms joined by coloured doors, a locked
 * door opens only while you carry the key of its colour (the key stays in
 * hand, as in Minigrid's Door.toggle), and you carry one object at a time.
 * The mission is a list of steps in BabyAI's grammar, checked after every
 * action that succeeds; BabyAI's non-strict order applies - a step done early
 * does not fail the mission, it counts once the steps before it are done.
 * Provenance: data/borrowed/babyai/README.md.
 *
 * Every refusal says what to do instead ("the blue door is locked - you need
 * the blue key"), because the next turn's observation carries it verbatim.
 *
 * VALIDATE THEN MUTATE. Deterministic, no RNG.
 */

const T = require("./turnbase.js");

const DIRS = { north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0] };
const DIR_ORDER = ["north", "south", "east", "west"];
const WALL = "#";
const PICKABLE = new Set(["ball", "box", "key"]);

function agent(state) {
  return T.one(state, "agent");
}

function objects(state) {
  return T.records(state, "object");
}

function carried(state) {
  return objects(state).find((o) => o.held);
}

function tileAt(state, x, y) {
  const map = state.map || { width: 0, height: 0, rows: [] };
  if (x < 0 || y < 0 || x >= map.width || y >= map.height) return WALL;
  const row = map.rows[y] || "";
  return row[x] || WALL;
}

function objectAt(state, x, y) {
  return objects(state).find((o) => !o.held && o.x === x && o.y === y);
}

function inside(room, x, y) {
  return room.x0 <= x && x <= room.x1 && room.y0 <= y && y <= room.y1;
}

/** The room a tile is in; a doorway belongs to both rooms it joins. */
function roomsOf(state, x, y) {
  const rooms = state.rooms || [];
  const direct = rooms.filter((r) => inside(r, x, y)).map((r) => r.id);
  if (direct.length) return direct;
  return rooms.filter((r) => DIR_ORDER.some((d) =>
    inside(r, x + DIRS[d][0], y + DIRS[d][1]))).map((r) => r.id);
}

function roomName(state, id) {
  const r = (state.rooms || []).find((x) => x.id === id);
  return r ? r.name : id;
}

function name(o) {
  return `${o.color} ${o.kind}`;
}

function dist(a, o) {
  return Math.abs(a.x - o.x) + Math.abs(a.y - o.y);
}

function droppable(state, x, y) {
  return tileAt(state, x, y) !== WALL && !objectAt(state, x, y)
    && (state.rooms || []).some((r) => inside(r, x, y));
}

function direction(state, rt, raw) {
  const d = String(raw || "").toLowerCase();
  if (!(d in DIRS)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `${raw} is not a direction - use north, south, east or west`);
  }
  return d;
}

function findObject(state, rt, id) {
  const o = objects(state).find((x) => x.id === id);
  if (!o) {
    throw T.fail(state, rt, "NOT_FOUND",
      `there is no ${id} - name something from this turn's list`);
  }
  return o;
}

function lockedWords(state, door) {
  const held = carried(state);
  if (held && held.kind === "key" && held.color === door.color) {
    return `the ${name(door)} is locked - open it, you carry the ${door.color} key`;
  }
  const extra = held && held.kind === "key" ? `, not the ${name(held)}` : "";
  return `the ${name(door)} is locked - you need the ${door.color} key${extra}`;
}

function tooFar(o, a) {
  const n = dist(a, o);
  return `the ${name(o)} is ${n} steps away - walk next to it first`;
}

// ------------------------------------------------------------ the mission

function matches(o, desc) {
  return o.kind === desc.kind && (!desc.color || o.color === desc.color);
}

function stepDone(state, step) {
  const a = agent(state);
  const objs = objects(state);
  if (step.verb === "goto") {
    return objs.some((o) => !o.held && matches(o, step.obj) && dist(a, o) === 1);
  }
  if (step.verb === "pickup") {
    const held = carried(state);
    return Boolean(held && matches(held, step.obj));
  }
  if (step.verb === "open") {
    return objs.some((o) => o.kind === "door" && o.open && matches(o, step.obj));
  }
  if (step.verb === "putnext") {
    return objs.some((m) => !m.held && matches(m, step.obj)
      && objs.some((f) => f !== m && !f.held && matches(f, step.fixed)
        && dist(m, f) === 1));
  }
  return false;
}

function stepWords(step) {
  const d = (x) => (x.color ? `${x.color} ` : "") + x.kind;
  if (step.verb === "goto") return `go to the ${d(step.obj)}`;
  if (step.verb === "pickup") return `pick up the ${d(step.obj)}`;
  if (step.verb === "open") return `open the ${d(step.obj)}`;
  return `put the ${d(step.obj)} next to the ${d(step.fixed)}`;
}

/** After every successful action: in order for "then"/"after" (a step
 *  counts once the ones before it are done, and a later one already true
 *  counts at once), any order for "and". */
function checkMission(state) {
  const m = state.mission;
  if (!m || !Array.isArray(m.steps)) return;
  if (!Array.isArray(m.done)) m.done = m.steps.map(() => false);
  const ordered = m.connective !== "and";
  for (let i = 0; i < m.steps.length; i += 1) {
    if (m.done[i]) continue;
    if (stepDone(state, m.steps[i])) {
      m.done[i] = true;
      if (m.steps.length > 1) T.log(state, `done: ${stepWords(m.steps[i])}`);
    } else if (ordered) {
      break;
    }
  }
  if (m.done.every(Boolean)) {
    state.status = "won";
    T.log(state, "mission complete");
  }
}

function visit(state) {
  const a = agent(state);
  if (!Array.isArray(state.visited)) state.visited = [];
  const fresh = roomsOf(state, a.x, a.y).filter((r) => !state.visited.includes(r));
  for (const r of fresh) state.visited.push(r);
  return fresh;
}

// ------------------------------------------------------------------ tools

function move(state, params, rt) {
  T.beginAction(state, rt);
  const d = direction(state, rt, params[0]);
  const a = agent(state);
  const x = a.x + DIRS[d][0];
  const y = a.y + DIRS[d][1];
  if (tileAt(state, x, y) === WALL) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `cannot move ${d}: a wall is in the way`);
  }
  const o = objectAt(state, x, y);
  if (o && o.kind === "door" && !o.open) {
    throw T.fail(state, rt, "INVALID_ARGUMENT", o.locked
      ? `cannot move ${d}: ${lockedWords(state, o)}`
      : `cannot move ${d}: the ${name(o)} is shut - open it first`);
  }
  if (o && o.kind !== "door") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `cannot move ${d}: the ${name(o)} is in the way - pick it up or go around`);
  }
  T.spend(state);
  a.x = x;
  a.y = y;
  a.facing = d;
  const fresh = visit(state);
  T.log(state, fresh.length
    ? `moved ${d}, now seeing the ${fresh.map((r) => roomName(state, r)).join(" and the ")}`
    : `moved ${d}`);
  checkMission(state);
  return rt.clone(a);
}

function drop(state, params, rt) {
  T.beginAction(state, rt);
  const d = direction(state, rt, params[0]);
  const held = carried(state);
  if (!held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "you are not carrying anything to drop");
  }
  const a = agent(state);
  const x = a.x + DIRS[d][0];
  const y = a.y + DIRS[d][1];
  if (!droppable(state, x, y)) {
    const o = objectAt(state, x, y);
    const what = tileAt(state, x, y) === WALL ? "a wall"
      : o ? `the ${name(o)}` : "a doorway";
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `cannot drop the ${name(held)} ${d}: ${what} is there - drop it on empty floor`);
  }
  T.spend(state);
  held.held = false;
  held.x = x;
  held.y = y;
  a.facing = d;
  T.log(state, `dropped the ${name(held)} to the ${d}`);
  checkMission(state);
  return rt.clone(a);
}

function pick_up(state, params, rt) {
  T.beginAction(state, rt);
  const o = findObject(state, rt, params[0]);
  if (o.held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `you are already carrying the ${name(o)}`);
  }
  if (o.kind === "door") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${name(o)} cannot be picked up - open it instead`);
  }
  const a = agent(state);
  if (dist(a, o) !== 1) {
    throw T.fail(state, rt, "INVALID_ARGUMENT", tooFar(o, a));
  }
  const held = carried(state);
  if (held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `your hands are full - drop the ${name(held)} first`);
  }
  T.spend(state);
  o.held = true;
  T.log(state, `picked up the ${name(o)}`);
  checkMission(state);
  return rt.clone(o);
}

function open(state, params, rt) {
  T.beginAction(state, rt);
  const o = findObject(state, rt, params[0]);
  if (o.kind !== "door") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${name(o)} is not a door - pick it up instead`);
  }
  if (o.open) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${name(o)} is already open - walk through it`);
  }
  const a = agent(state);
  if (dist(a, o) !== 1) {
    throw T.fail(state, rt, "INVALID_ARGUMENT", tooFar(o, a));
  }
  const held = carried(state);
  if (o.locked && !(held && held.kind === "key" && held.color === o.color)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT", lockedWords(state, o));
  }
  T.spend(state);
  const wasLocked = o.locked;
  o.locked = false;
  o.open = true;
  T.log(state, wasLocked
    ? `unlocked the ${name(o)} with the ${name(held)}`
    : `opened the ${name(o)}`);
  checkMission(state);
  return rt.clone(o);
}

/** The carried object goes on a free floor tile beside the target and at
 *  right angles to the agent's side of it (so it touches both), first free
 *  one in north/south/east/west order. */
function put_next_to(state, params, rt) {
  T.beginAction(state, rt);
  const o = findObject(state, rt, params[0]);
  const held = carried(state);
  if (!held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "you are not carrying anything - pick something up first");
  }
  if (o.held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `you are carrying the ${name(o)} - put it next to something else`);
  }
  const a = agent(state);
  if (dist(a, o) !== 1) {
    throw T.fail(state, rt, "INVALID_ARGUMENT", tooFar(o, a));
  }
  const dx = a.x - o.x;
  const dy = a.y - o.y;
  let spot = null;
  for (const d of DIR_ORDER) {
    const [px, py] = DIRS[d];
    if (px * dx + py * dy !== 0) continue;
    if (droppable(state, o.x + px, o.y + py)) {
      spot = [o.x + px, o.y + py];
      break;
    }
  }
  if (!spot) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `no free tile beside the ${name(o)} that is also beside you - stand on another side of it`);
  }
  T.spend(state);
  held.held = false;
  held.x = spot[0];
  held.y = spot[1];
  T.log(state, `put the ${name(held)} next to the ${name(o)}`);
  checkMission(state);
  return rt.clone(held);
}

// -------------------------------------------------------------- turn end

function end_turn(state) {
  if (state.status === "playing" && state.max_turns
      && (state.turn || 0) + 1 >= state.max_turns) {
    state.status = "out_of_time";
    T.log(state, "out of time");
  }
  return T.endTurn(state);
}

module.exports = { move, drop, pick_up, open, put_next_to, end_turn };
