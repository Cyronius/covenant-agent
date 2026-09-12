"use strict";
/*
 * Shared turn machinery for the decision worlds (warehouse, house, elevator,
 * cards, page). runtime/engines/rpg.js grew all of this first and keeps its
 * own copy - it is the held-out exam and nothing here should be able to
 * change it by accident.
 *
 * Same contract as every engine: fn(state, params, rt) -> result, mutating
 * `state` in place and pushing one line onto `state.log`. VALIDATE THEN
 * MUTATE - the sandbox hands the harness back the mutated state even when a
 * tool throws.
 */

const TURN_BUDGET_DEFAULT = 3;

function log(state, text) {
  if (!Array.isArray(state.log)) state.log = [];
  state.log.push(text);
}

/** Every action starts here: the episode must be live and the turn's budget
 *  unspent. Also clears last turn's events on the first action of a turn, so
 *  "Last turn:" renders exactly one turn (rpg.js's beginAction comment has
 *  the history this fixes). */
function beginAction(state, rt) {
  if (!(state.actions_this_turn > 0)) state.log = [];
  if (state.status && state.status !== "playing") {
    throw new rt.ToolError("INVALID_ARGUMENT",
      `the episode is over (${state.status})`);
  }
  const budget = state.turn_budget === undefined
    ? TURN_BUDGET_DEFAULT : state.turn_budget;
  if ((state.actions_this_turn || 0) >= budget) {
    throw new rt.ToolError("RATE_LIMITED",
      `turn budget spent (${budget} actions per turn)`);
  }
}

/** Charged for actions that mutate AND for actions rejected as illegal -
 *  otherwise a TRY-wrapped program probes the world for free. */
function spend(state) {
  state.actions_this_turn = (state.actions_this_turn || 0) + 1;
}

function fail(state, rt, code, message) {
  spend(state);
  log(state, `failed: ${message}`);
  return new rt.ToolError(code, message);
}

/** The tail every post_hook shares: roll the turn, clear the budget, and keep
 *  the log to one turn even when nothing executed (a program that did not
 *  compile never reaches beginAction's clear). */
function endTurn(state) {
  if (!(state.actions_this_turn > 0)) state.log = [];
  state.turn = (state.turn || 0) + 1;
  state.actions_this_turn = 0;
  if (state.log && state.log.length > 12) state.log = state.log.slice(-12);
  return state;
}

function records(state, entity) {
  return (state.entities && state.entities[entity]) || [];
}

function one(state, entity) {
  return records(state, entity)[0];
}

module.exports = { beginAction, spend, fail, log, endTurn, records, one,
                   TURN_BUDGET_DEFAULT };
