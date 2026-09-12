"use strict";
/*
 * Card table rules
 * (plan .claude/plans/archive/task-families.md, family A).
 *
 * The smallest instance of "read the situation": no map, no ids, three
 * actions, and the whole state is four numbers. If a model cannot choose
 * here it is not the grid reading that beat it.
 *
 * Blackjack with the fiddly parts removed - no splits, no insurance, dealer
 * stands on all 17. The shoe is dealt from `state.deck` in order, so the
 * world stays deterministic: same program + same state => same cards.
 */

const T = require("./turnbase.js");

const DEALER_STANDS = 17;

function table(state) {
  return T.one(state, "table");
}

function draw(state) {
  const card = state.deck[state.deck_pos];
  state.deck_pos += 1;
  return card;
}

/** Blackjack totals: an ace is 11 while that fits under 21, else 1. */
function score(cards) {
  let total = 0;
  let aces = 0;
  for (const c of cards) {
    total += c === 1 ? 11 : c;
    if (c === 1) aces += 1;
  }
  let soft = aces > 0;
  while (total > 21 && aces > 0) {
    total -= 10;
    aces -= 1;
    soft = aces > 0;
  }
  return { total, soft };
}

function cardWord(c) {
  if (c === 1) return "an ace";
  if (c === 10) return "a ten";
  return `a ${c}`;
}

function sync(state) {
  const t = table(state);
  const p = score(state.player_cards);
  t.player_total = p.total;
  t.player_soft = p.soft;
  t.cards_taken = state.player_cards.length - 2;
  t.dealer_up = state.dealer_cards[0];
  t.can_double = state.player_cards.length === 2
    && t.bankroll >= t.bet;
}

function deal(state) {
  const t = table(state);
  t.hand_no += 1;
  if (t.hand_no > t.hands) {
    state.status = "done";
    sync(state);
    return;
  }
  t.bet = state.base_bet;
  state.player_cards = [draw(state), draw(state)];
  state.dealer_cards = [draw(state), draw(state)];
  sync(state);
  const p = score(state.player_cards);
  const d = score(state.dealer_cards);
  if (p.total === 21 || d.total === 21) {
    // naturals settle themselves; nobody gets a decision on them
    if (p.total === 21 && d.total !== 21) {
      t.bankroll += Math.floor(t.bet * 3 / 2);
      T.log(state, `hand ${t.hand_no}: blackjack, +${Math.floor(t.bet * 3 / 2)}`);
    } else if (d.total === 21 && p.total !== 21) {
      t.bankroll -= t.bet;
      T.log(state, `hand ${t.hand_no}: dealer blackjack, -${t.bet}`);
    } else {
      T.log(state, `hand ${t.hand_no}: both blackjack, push`);
    }
    deal(state);
    return;
  }
  T.log(state, `hand ${t.hand_no} dealt: you have `
    + `${p.total}${p.soft ? " soft" : ""}, the dealer shows `
    + `${cardWord(t.dealer_up)}`);
}

/** Dealer plays out and the hand settles; the next hand is dealt straight
 *  away so the next observation is always a live decision. */
function settle(state, playerTotal) {
  const t = table(state);
  let d = score(state.dealer_cards);
  while (d.total < DEALER_STANDS) {
    state.dealer_cards.push(draw(state));
    d = score(state.dealer_cards);
  }
  let delta;
  if (playerTotal > 21) delta = -t.bet;
  else if (d.total > 21 || playerTotal > d.total) delta = t.bet;
  else if (playerTotal === d.total) delta = 0;
  else delta = -t.bet;
  t.bankroll += delta;
  const verb = delta > 0 ? `won ${delta}` : delta < 0 ? `lost ${-delta}` : "pushed";
  T.log(state, `hand ${t.hand_no}: you had ${playerTotal > 21 ? "bust" : playerTotal}`
    + `, the dealer made ${d.total > 21 ? "bust" : d.total} - you ${verb}`);
  if (t.bankroll <= 0) {
    state.status = "broke";
    T.log(state, "the bankroll is gone");
    return;
  }
  deal(state);
}

// ------------------------------------------------------------------ tools

function hit(state, params, rt) {
  T.beginAction(state, rt);
  const t = table(state);
  T.spend(state);
  const card = draw(state);
  state.player_cards.push(card);
  sync(state);
  const p = score(state.player_cards);
  T.log(state, `took ${cardWord(card)} - you have ${p.total}`
    + `${p.soft ? " soft" : ""}`);
  if (p.total > 21) settle(state, p.total);
  return rt.clone(table(state));
}

function stand(state, params, rt) {
  T.beginAction(state, rt);
  T.spend(state);
  const p = score(state.player_cards);
  T.log(state, `stood on ${p.total}`);
  settle(state, p.total);
  return rt.clone(table(state));
}

function double_down(state, params, rt) {
  T.beginAction(state, rt);
  const t = table(state);
  if (state.player_cards.length !== 2) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "you can only double on your first two cards");
  }
  if (t.bankroll < t.bet) {
    throw T.fail(state, rt, "INVALID_ARGUMENT",
      "not enough chips left to double the bet");
  }
  T.spend(state);
  t.bet *= 2;
  const card = draw(state);
  state.player_cards.push(card);
  sync(state);
  const p = score(state.player_cards);
  T.log(state, `doubled to ${t.bet} and took ${cardWord(card)} - `
    + `you have ${p.total > 21 ? "bust" : p.total}`);
  settle(state, p.total);
  return rt.clone(table(state));
}

// -------------------------------------------------------------- turn end

/** Nothing acts against you between turns; the shoe running out is the only
 *  thing that can end an episode the player did not end. */
function end_turn(state) {
  if (state.status === "playing"
      && state.deck_pos > state.deck.length - 8) {
    state.status = "done";
    T.log(state, "the shoe is finished");
  }
  return T.endTurn(state);
}

module.exports = { hit, stand, double_down, end_turn, score };
