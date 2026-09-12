"use strict";
/*
 * Warehouse robot rules
 * (plan .claude/plans/archive/task-families.md, family A).
 *
 * A grid like the dungeon's, but the robot knows the floor plan and reads it
 * as prose - what varies from the RPG on purpose is the rendering, not the
 * skill. The battery is the second thing that varies: the shortest route to
 * the tote is often not the route you can finish, so some turns the right
 * move is a detour to the charging pad.
 *
 * Deterministic, no RNG. Same program + same state => same result.
 */

const T = require("./turnbase.js");

const DIRS = { north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0] };
const RACK = "#";
const DRIVE_COST = 1;

function robot(state) {
  return T.one(state, "robot");
}

function totes(state) {
  return T.records(state, "tote");
}

function tileAt(state, x, y) {
  const map = state.map || { width: 0, height: 0, rows: [] };
  if (x < 0 || y < 0 || x >= map.width || y >= map.height) return RACK;
  const row = map.rows[y] || "";
  return row[x] || RACK;
}

function at(rec, x, y) {
  return rec && rec.x === x && rec.y === y;
}

function carried(state) {
  return totes(state).find((t) => t.held);
}

/** Won when every tote the job names is sitting on the dock, unheld. */
function checkDone(state) {
  const dock = T.one(state, "dock");
  if (!dock) return;
  const wanted = state.job || [];
  const done = wanted.every((id) => {
    const t = totes(state).find((x) => x.id === id);
    return t && !t.held && at(t, dock.x, dock.y);
  });
  if (done && wanted.length) {
    state.status = "done";
    T.log(state, "every tote on the job is on the outbound dock");
  }
}

// ------------------------------------------------------------------ tools

function drive(state, params, rt) {
  T.beginAction(state, rt);
  const r = robot(state);
  const dir = String(params[0] || "").toLowerCase();
  if (!(dir in DIRS)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `${params[0]} is not a direction (north, south, east, west)`);
  }
  if (r.battery <= 0) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "the battery is flat - you cannot drive");
  }
  const [dx, dy] = DIRS[dir];
  const nx = r.x + dx;
  const ny = r.y + dy;
  if (tileAt(state, nx, ny) === RACK) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `cannot drive ${dir}: a rack is in the way`);
  }
  T.spend(state);
  r.x = nx;
  r.y = ny;
  r.battery -= DRIVE_COST;
  const held = carried(state);
  if (held) {
    held.x = nx;
    held.y = ny;
  }
  T.log(state, `drove ${dir} to aisle ${nx}, bay ${ny} `
    + `(${r.battery} battery left)`);
  checkDone(state);
  return rt.clone(r);
}

function lift(state, params, rt) {
  T.beginAction(state, rt);
  const r = robot(state);
  const tote = totes(state).find((t) => t.id === params[0]);
  if (!tote) throw T.fail(state, rt, "NOT_FOUND", `no tote ${params[0]}`);
  if (tote.held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `you are already carrying tote ${tote.label}`);
  }
  if (carried(state)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "the deck is full - set the tote you are carrying down first");
  }
  if (!at(tote, r.x, r.y)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `tote ${tote.label} is not on your bay - drive onto it first`);
  }
  T.spend(state);
  tote.held = true;
  T.log(state, `lifted tote ${tote.label}`);
  return rt.clone(tote);
}

function set_down(state, params, rt) {
  T.beginAction(state, rt);
  const r = robot(state);
  const tote = totes(state).find((t) => t.id === params[0]);
  if (!tote) throw T.fail(state, rt, "NOT_FOUND", `no tote ${params[0]}`);
  if (!tote.held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `you are not carrying tote ${tote.label}`);
  }
  T.spend(state);
  tote.held = false;
  tote.x = r.x;
  tote.y = r.y;
  T.log(state, `set tote ${tote.label} down at aisle ${r.x}, bay ${r.y}`);
  checkDone(state);
  return rt.clone(tote);
}

function charge(state, params, rt) {
  T.beginAction(state, rt);
  const r = robot(state);
  const pad = T.one(state, "charger");
  if (!pad || !at(pad, r.x, r.y)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "there is no charging pad on this bay");
  }
  if (r.battery >= r.max_battery) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "the battery is already full");
  }
  T.spend(state);
  const gained = r.max_battery - r.battery;
  r.battery = r.max_battery;
  T.log(state, `charged +${gained} (${r.battery}/${r.max_battery})`);
  return rt.clone(r);
}

// -------------------------------------------------------------- turn end

/** No adversary here - the clock is the battery. A robot that runs the pack
 *  down away from the pad is stranded, which is this world's `dead`. */
function end_turn(state) {
  const r = robot(state);
  if (r && state.status === "playing" && r.battery <= 0) {
    const pad = T.one(state, "charger");
    if (!pad || !at(pad, r.x, r.y)) {
      state.status = "stranded";
      T.log(state, "the battery is flat and the pad is out of reach");
    }
  }
  return T.endTurn(state);
}

module.exports = { drive, lift, set_down, charge, end_turn };
