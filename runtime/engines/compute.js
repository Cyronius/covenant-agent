"use strict";
/*
 * Compute tools (spec 0.7.0 §6; plan step 1g): sum/avg/max/min over a list
 * of numbers, the standard tool block rendered into every context so a
 * program can fold a MAP-projected field without a new opcode ("can this be
 * a tool?", spec §12). Loaded by runtime/sandbox.js's `engine` impl op.
 *
 * Contract: fn(state, params, rt) -> result. `state` is unused — these are
 * pure functions of their single LIST argument, never gated (no state
 * change, no consequence).
 *
 * Empty list: `sum` is 0 (the identity element); `avg`/`max`/`min` are
 * `null`, matching `FIRST`'s "empty list -> NULL" precedent (spec §4)
 * rather than picking an arbitrary in-range number.
 */

function sum(state, params) {
  return params[0].reduce((a, b) => a + b, 0);
}

function avg(state, params) {
  const list = params[0];
  return list.length === 0 ? null : sum(state, params) / list.length;
}

function max(state, params) {
  const list = params[0];
  return list.length === 0 ? null : Math.max(...list);
}

function min(state, params) {
  const list = params[0];
  return list.length === 0 ? null : Math.min(...list);
}

module.exports = { sum, avg, max, min };
