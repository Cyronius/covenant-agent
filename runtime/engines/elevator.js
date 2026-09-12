"use strict";
/*
 * Elevator dispatcher rules
 * (plan .claude/plans/archive/task-families.md, family A).
 *
 * The world in this family where doing nothing is a legal move and sometimes
 * the right one: a rider is announced a few ticks before they press the
 * button, so a car already standing on their floor should hold rather than
 * set off and come back. No tuned checkpoint has ever emitted a do-nothing
 * turn, and this is the world that asks for one.
 *
 * The other thing it varies is the argument: every tool here takes a number
 * off a table (a floor) or nothing at all - no entity ids.
 *
 * Deterministic, no RNG.
 */

const T = require("./turnbase.js");

function car(state) {
  return T.one(state, "car");
}

function riders(state) {
  return T.records(state, "rider");
}

/** A rider is waiting once their tick has come round and they are neither
 *  aboard nor dropped off. */
function waiting(state, rider) {
  return rider.appears <= (state.tick || 0) && !rider.aboard
    && !rider.delivered;
}

function checkDone(state) {
  if (riders(state).every((r) => r.delivered)) {
    state.status = "done";
    T.log(state, "every rider has been dropped off");
  }
}

function tick(state, n) {
  state.tick = (state.tick || 0) + n;
}

// ------------------------------------------------------------------ tools

function go_to(state, params, rt) {
  T.beginAction(state, rt);
  const c = car(state);
  const floor = Number(params[0]);
  if (!Number.isInteger(floor) || floor < 1 || floor > state.floors) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `floor ${params[0]} is not in this building (1..${state.floors})`);
  }
  if (floor === c.floor) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the car is already on floor ${floor}`);
  }
  if (c.doors_open) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "close the doors before moving - open them again to let people out");
  }
  const travel = Math.abs(floor - c.floor);
  T.spend(state);
  c.floor = floor;
  tick(state, travel);
  T.log(state, `moved to floor ${floor} (${travel} ticks)`);
  return rt.clone(c);
}

/** One exchange: everyone waiting on this floor gets on, everyone whose stop
 *  this is gets off, and the doors shut again. Doing it with nobody there is
 *  legal and wastes the tick - the log says so, which is the feedback the
 *  next turn reads. */
function open_doors(state, params, rt) {
  T.beginAction(state, rt);
  const c = car(state);
  T.spend(state);
  tick(state, 1);
  let off = 0;
  let on = 0;
  for (const r of riders(state)) {
    if (r.aboard && r.dest === c.floor) {
      r.aboard = false;
      r.delivered = true;
      off += 1;
    }
  }
  const capacity = state.capacity === undefined ? 4 : state.capacity;
  for (const r of riders(state)) {
    if (!waiting(state, r) || r.origin !== c.floor) continue;
    if (riders(state).filter((x) => x.aboard).length >= capacity) {
      T.log(state, `the car is full - ${r.name} is still waiting`);
      continue;
    }
    r.aboard = true;
    on += 1;
  }
  if (off || on) {
    T.log(state, `doors on floor ${c.floor}: ${off} off, ${on} on`);
  } else {
    T.log(state, `doors on floor ${c.floor}: nobody was there`);
  }
  checkDone(state);
  return rt.clone(c);
}

/** Stand still for a tick. The only tool in any of these worlds that is
 *  allowed to change nothing. */
function hold(state, params, rt) {
  T.beginAction(state, rt);
  const c = car(state);
  T.spend(state);
  tick(state, 1);
  T.log(state, `held on floor ${c.floor}`);
  return rt.clone(c);
}

// -------------------------------------------------------------- turn end

/** Riders give up if they have waited too long; a episode that loses one is
 *  over. The give-up clock is what makes holding a decision rather than a
 *  free move. */
function end_turn(state) {
  if (state.status === "playing") {
    const patience = state.patience === undefined ? 25 : state.patience;
    for (const r of riders(state)) {
      if (r.delivered || r.aboard) continue;
      if (r.appears > (state.tick || 0)) continue;
      if ((state.tick || 0) - r.appears > patience) {
        state.status = "walked_out";
        T.log(state, `${r.name} gave up waiting on floor ${r.origin}`);
        break;
      }
    }
  }
  return T.endTurn(state);
}

module.exports = { go_to, open_doors, hold, end_turn };
