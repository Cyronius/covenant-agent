"use strict";
/*
 * Workshop crafting rules (.claude/plans/borrowed-worlds.md, "2. Crafting").
 *
 * TextCraft's mechanic and Crafter's tech tree with our own items: raw
 * materials are gathered, everything else is crafted from counts of
 * ingredients, and some recipes need a station standing in the workshop.
 * Stations are crafted like anything else and never used up.
 *
 * Every refusal says what to do instead ("gather it", "craft the kiln
 * first"): the episode generator plays wrong moves on purpose and labels the
 * next turn from what the refusal produced, so the reason is the lesson.
 *
 * VALIDATE THEN MUTATE. Deterministic, no RNG.
 */

const T = require("./turnbase.js");

function items(state) {
  return T.records(state, "item");
}

function byId(state, id) {
  return items(state).find((i) => i.id === id);
}

function recipeOf(state, item) {
  return (state.recipes || {})[item.id];
}

function lookup(state, rt, id) {
  const item = byId(state, id);
  if (!item) throw T.fail(state, rt, "NOT_FOUND", `no item called ${id}`);
  return item;
}

/** "gather sand" or "craft glass pane": how to come by an ingredient. */
function howToGet(item) {
  return item.raw ? `gather ${item.name}` : `craft ${item.name}`;
}

function checkDone(state) {
  const goal = state.goal || {};
  const item = byId(state, goal.item);
  if (item && item.have >= (goal.count || 1)) {
    state.status = "done";
    T.log(state, `the job is done: ${item.have} ${item.name} made`);
  }
}

// ------------------------------------------------------------------ tools

function gather(state, params, rt) {
  T.beginAction(state, rt);
  const item = lookup(state, rt, params[0]);
  if (!item.raw) {
    const recipe = recipeOf(state, item);
    const what = recipe
      ? recipe.needs.map(([id, n]) => `${n} ${byId(state, id).name}`)
        .join(", ")
      : "its ingredients";
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `${item.name} does not grow wild - craft it from ${what}`);
  }
  T.spend(state);
  item.have += item.gives;
  T.log(state, `gathered ${item.gives} ${item.name} (now ${item.have})`);
  return rt.clone(item);
}

function craft(state, params, rt) {
  T.beginAction(state, rt);
  const item = lookup(state, rt, params[0]);
  if (item.raw) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `${item.name} is a raw material with no recipe - gather it instead`);
  }
  const recipe = recipeOf(state, item);
  if (!recipe) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `nobody here knows how to make ${item.name}`);
  }
  if (item.station && item.have > 0) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      `the ${item.name} already stands in the workshop - `
      + "stations are never used up");
  }
  if (recipe.at) {
    const station = byId(state, recipe.at);
    if (!station || station.have < 1) {
      throw T.fail(state, rt, "INVALID_ARGUMENT",
        `${item.name} needs a ${station ? station.name : recipe.at} `
        + `standing here - craft the ${station ? station.name : "station"} `
        + "first");
    }
  }
  for (const [id, n] of recipe.needs) {
    const ing = byId(state, id);
    if (ing.have < n) {
      throw T.fail(state, rt, "INVALID_ARGUMENT",
        `short ${n - ing.have} ${ing.name} for ${item.name} - `
        + `${howToGet(ing)} first`);
    }
  }
  T.spend(state);
  for (const [id, n] of recipe.needs) byId(state, id).have -= n;
  item.have += recipe.makes;
  if (!state.crafted) state.crafted = {};
  state.crafted[item.id] = (state.crafted[item.id] || 0) + recipe.makes;
  T.log(state, item.station
    ? `built the ${item.name}`
    : `crafted ${recipe.makes} ${item.name} (now ${item.have})`);
  checkDone(state);
  return rt.clone(item);
}

/** Read-only in effect, but it takes one of the turn's actions like any
 *  other, so it is not a free probe. */
function inspect(state, params, rt) {
  T.beginAction(state, rt);
  const item = lookup(state, rt, params[0]);
  T.spend(state);
  let text;
  if (item.raw) {
    text = `${item.name}: raw material, gather brings ${item.gives}, `
      + `you hold ${item.have}`;
  } else {
    const recipe = recipeOf(state, item) || { needs: [], makes: 0, at: "" };
    const needs = recipe.needs
      .map(([id, n]) => `${n} ${byId(state, id).name}`).join(", ");
    const at = recipe.at ? ` at a ${byId(state, recipe.at).name}` : "";
    text = `${item.name}: ${needs}${at} make ${recipe.makes}, `
      + `you hold ${item.have}`;
  }
  T.log(state, `inspected ${text}`);
  return text;
}

// -------------------------------------------------------------- turn end

function end_turn(state) {
  return T.endTurn(state);
}

module.exports = { gather, craft, inspect, end_turn };
