"use strict";
/*
 * Screen rules for the page worlds
 * (plan .claude/plans/archive/task-families.md, family C: UI navigation
 * and form filling).
 *
 * The situation is a screen - what is on it, what it currently says, and
 * where you can go from here - and the tools are the five things a person
 * does to a screen. It reuses family A's machinery wholesale; what is new is
 * that the goal is a form's end state rather than a tile.
 *
 * `click` and `submit` are deliberately close: clicking a form's confirm
 * button fails and says to submit it instead. That is family B's pressure
 * (choose by description, not by verb habit) inside family C's world.
 *
 * Deterministic, no RNG.
 */

const T = require("./turnbase.js");

function app(state) {
  return T.one(state, "app");
}

function screens(state) {
  return T.records(state, "screen");
}

function elements(state) {
  return T.records(state, "element");
}

function screenById(state, id) {
  return screens(state).find((s) => s.id === id);
}

function onScreen(state, el) {
  return el.screen === app(state).screen;
}

/** The goal is a form's end state: a screen to be on, values to have set,
 *  and forms to have submitted. */
function checkDone(state) {
  const goal = state.goal || {};
  if (goal.screen && app(state).screen !== goal.screen) return;
  for (const [id, want] of Object.entries(goal.values || {})) {
    const el = elements(state).find((e) => e.id === id);
    if (!el || String(el.value) !== String(want)) return;
  }
  for (const id of goal.submitted || []) {
    const el = elements(state).find((e) => e.id === id);
    if (!el || !el.done) return;
  }
  state.status = "done";
  T.log(state, "the task is finished");
}

// ------------------------------------------------------------------ tools

function open(state, params, rt) {
  T.beginAction(state, rt);
  const a = app(state);
  const target = screenById(state, params[0]);
  if (!target) throw T.fail(state, rt, "NOT_FOUND", `no screen ${params[0]}`);
  if (target.id === a.screen) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `${target.name} is already open`);
  }
  const nav = (state.nav || {})[a.screen] || [];
  if (!nav.includes(target.id)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `you cannot get to ${target.name} from ${screenById(state, a.screen).name}`);
  }
  T.spend(state);
  a.screen = target.id;
  T.log(state, `opened ${target.name}`);
  checkDone(state);
  return rt.clone(target);
}

function click(state, params, rt) {
  T.beginAction(state, rt);
  const a = app(state);
  const el = elements(state).find((e) => e.id === params[0]);
  if (!el) throw T.fail(state, rt, "NOT_FOUND", `no control ${params[0]}`);
  if (!onScreen(state, el)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" is not on the screen you have open`);
  }
  if (el.kind === "submit") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" saves the form - submit it instead of clicking it`);
  }
  if (el.kind === "field" || el.kind === "select") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" takes a value - fill it in instead of clicking it`);
  }
  if (el.kind === "text") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" is a label, not a control`);
  }
  T.spend(state);
  if (el.kind === "toggle") {
    el.value = el.value === "on" ? "off" : "on";
    T.log(state, `turned "${el.label}" ${el.value}`);
  } else if (el.kind === "link") {
    const target = screenById(state, el.target);
    if (target) {
      a.screen = target.id;
      T.log(state, `followed "${el.label}" to ${target.name}`);
    } else {
      T.log(state, `clicked "${el.label}"`);
    }
  } else {
    el.done = true;
    T.log(state, `clicked "${el.label}"`);
  }
  checkDone(state);
  return rt.clone(el);
}

function fill(state, params, rt) {
  T.beginAction(state, rt);
  const el = elements(state).find((e) => e.id === params[0]);
  const value = params[1] === undefined || params[1] === null
    ? "" : String(params[1]);
  if (!el) throw T.fail(state, rt, "NOT_FOUND", `no field ${params[0]}`);
  if (!onScreen(state, el)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" is not on the screen you have open`);
  }
  if (el.kind !== "field" && el.kind !== "select") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" is a ${el.kind}, not something you type into`);
  }
  if (el.kind === "select") {
    const options = String(el.options || "").split("|").filter(Boolean);
    if (!options.includes(value)) {
      throw T.fail(state, rt, "INVALID_ARGUMENT",
        `"${value}" is not one of the choices for "${el.label}" `
        + `(${options.join(", ")})`);
    }
  }
  T.spend(state);
  el.value = value;
  T.log(state, `set "${el.label}" to "${value}"`);
  checkDone(state);
  return rt.clone(el);
}

function submit(state, params, rt) {
  T.beginAction(state, rt);
  const a = app(state);
  const el = elements(state).find((e) => e.id === params[0]);
  if (!el) throw T.fail(state, rt, "NOT_FOUND", `no button ${params[0]}`);
  if (!onScreen(state, el)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" is not on the screen you have open`);
  }
  if (el.kind !== "submit") {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" does not save anything - click it instead`);
  }
  const blank = elements(state).find(
    (e) => e.screen === el.screen && e.required && !String(e.value || ""));
  if (blank) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${blank.label}" is still empty and the form will not save`);
  }
  T.spend(state);
  el.done = true;
  T.log(state, `saved with "${el.label}"`);
  const target = screenById(state, el.target);
  if (target) {
    a.screen = target.id;
    T.log(state, `${target.name} opened`);
  }
  checkDone(state);
  return rt.clone(el);
}

function read(state, params, rt) {
  T.beginAction(state, rt);
  const el = elements(state).find((e) => e.id === params[0]);
  if (!el) throw T.fail(state, rt, "NOT_FOUND", `no element ${params[0]}`);
  if (!onScreen(state, el)) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `"${el.label}" is not on the screen you have open`);
  }
  T.spend(state);
  T.log(state, `read "${el.label}": ${el.value || "(empty)"}`);
  return String(el.value || "");
}

function end_turn(state) {
  return T.endTurn(state);
}

module.exports = { open, click, fill, submit, read, end_turn };
