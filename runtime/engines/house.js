"use strict";
/*
 * Text-adventure house rules
 * (plan .claude/plans/archive/task-families.md, family A).
 *
 * Held out, like the dungeon: rooms joined by named ways out, no grid
 * anywhere, everything in prose. Scoring the family on this and on the
 * dungeon together is what separates "learned the family" from "learned to
 * read a grid".
 *
 * Deterministic, no RNG.
 */

const T = require("./turnbase.js");

function explorer(state) {
  return T.one(state, "explorer");
}

function ways(state) {
  return T.records(state, "way");
}

function things(state) {
  return T.records(state, "thing");
}

function roomOf(state) {
  return explorer(state).room;
}

function held(state) {
  return things(state).filter((t) => t.held);
}

function lightHere(state, roomId) {
  const room = T.records(state, "room").find((r) => r.id === roomId);
  if (!room || !room.dark) return true;
  return held(state).some((t) => t.kind === "lamp" && t.lit);
}

// ------------------------------------------------------------------ tools

function go(state, params, rt) {
  T.beginAction(state, rt);
  const p = explorer(state);
  const way = ways(state).find((w) => w.id === params[0]);
  if (!way) throw T.fail(state, rt, "NOT_FOUND", `no way out called ${params[0]}`);
  if (way.room !== p.room) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${way.heading} way is not in this room`);
  }
  if (way.shut) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      way.locked ? `the ${way.heading} way is locked`
                 : `the ${way.heading} way is shut - open it first`);
  }
  T.spend(state);
  if (!lightHere(state, way.to)) {
    // you get as far as the doorway and back out; the turn is gone
    T.log(state, `it is pitch dark through the ${way.heading} way - `
      + "you back out");
    return rt.clone(T.records(state, "room").find((r) => r.id === p.room));
  }
  p.room = way.to;
  const room = T.records(state, "room").find((r) => r.id === way.to);
  T.log(state, `went ${way.heading} into the ${room.name}`);
  if (way.to === state.goal_room) {
    state.status = "done";
    T.log(state, `you have reached the ${room.name}`);
  }
  return rt.clone(room);
}

function take(state, params, rt) {
  T.beginAction(state, rt);
  const p = explorer(state);
  const thing = things(state).find((t) => t.id === params[0]);
  if (!thing) throw T.fail(state, rt, "NOT_FOUND", `no ${params[0]} here`);
  if (thing.held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `you are already carrying the ${thing.name}`);
  }
  if (thing.room !== p.room) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${thing.name} is not in this room`);
  }
  T.spend(state);
  thing.held = true;
  thing.room = "";
  T.log(state, `took the ${thing.name}`);
  return rt.clone(thing);
}

function use(state, params, rt) {
  T.beginAction(state, rt);
  const thing = things(state).find((t) => t.id === params[0]);
  if (!thing) throw T.fail(state, rt, "NOT_FOUND", `no ${params[0]}`);
  if (!thing.held) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `you are not carrying the ${thing.name} - take it first`);
  }
  if (thing.kind !== "lamp") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${thing.name} does nothing on its own - `
      + "a key opens a locked way, so open the way with it");
  }
  T.spend(state);
  thing.lit = !thing.lit;
  T.log(state, `${thing.lit ? "lit" : "put out"} the ${thing.name}`);
  return rt.clone(thing);
}

function open(state, params, rt) {
  T.beginAction(state, rt);
  const p = explorer(state);
  const way = ways(state).find((w) => w.id === params[0]);
  if (!way) throw T.fail(state, rt, "NOT_FOUND", `no way out called ${params[0]}`);
  if (way.room !== p.room) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${way.heading} way is not in this room`);
  }
  if (!way.shut) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${way.heading} way is already open`);
  }
  if (way.locked) {
    const key = held(state).find((t) => t.kind === "key" && t.opens === way.id);
    if (!key) {
      throw T.fail(state, rt, "INVALID_ARGUMENT",
        `the ${way.heading} way is locked and you have no key for it`);
    }
    T.spend(state);
    way.locked = false;
    way.shut = false;
    T.log(state, `unlocked the ${way.heading} way with the ${key.name}`);
    return rt.clone(way);
  }
  T.spend(state);
  way.shut = false;
  T.log(state, `opened the ${way.heading} way`);
  return rt.clone(way);
}

function end_turn(state) {
  return T.endTurn(state);
}

module.exports = { go, take, use, open, end_turn };
