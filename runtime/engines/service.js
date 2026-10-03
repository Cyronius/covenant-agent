"use strict";
/*
 * Customer service under a written policy (runtime/worlds/service.py):
 * the rules of tau2-bench's retail, airline and telecom policies
 * (github.com/sierra-research/tau2-bench, MIT), enforced where tau2 leaves
 * them to the agent. A request the policy forbids throws PERMISSION_DENIED
 * with a message that starts "policy:" and names the rule; a malformed call
 * throws INVALID_ARGUMENT; a missing record NOT_FOUND.
 *
 * Contract as every engine: fn(state, params, rt) -> value, mutating `state`
 * in place. VALIDATE THEN MUTATE: the sandbox hands back the state even when
 * a tool throws. A refusal also leaves one `failed: <message>` line on
 * state.log (outside `entities`, so it never affects scoring) for a planner
 * that reads the last turn.
 */

function recs(state, entity) {
  if (!state.entities[entity]) state.entities[entity] = [];
  return state.entities[entity];
}

function find(state, entity, id, rt) {
  const r = recs(state, entity).find((x) => x.id === id);
  if (!r) throw new rt.ToolError("NOT_FOUND", `${entity} ${id} not found`);
  return r;
}

function deny(state, rt, message) {
  if (!Array.isArray(state.log)) state.log = [];
  state.log.push(`failed: policy: ${message}`);
  return new rt.ToolError("PERMISSION_DENIED", `policy: ${message}`);
}

function bad(rt, message) {
  return new rt.ToolError("INVALID_ARGUMENT", message);
}

function dollars(cents) {
  return `$${(cents / 100).toFixed(2)}`;
}

// ------------------------------------------------------------------ retail

const RETAIL_REASONS = ["no longer needed", "ordered by mistake"];

function ownedMethod(state, rt, userId, pmId) {
  const pm = find(state, "payment_method", pmId, rt);
  if (pm.user !== userId) {
    throw deny(state, rt,
      `payment method ${pmId} is not in this customer's profile`);
  }
  return pm;
}

function retail_find_user_id_by_email(state, params, rt) {
  const email = String(params[0] || "").trim().toLowerCase();
  const u = recs(state, "user").find(
    (x) => String(x.email).toLowerCase() === email);
  if (!u) throw new rt.ToolError("NOT_FOUND", "user not found");
  return u.id;
}

function retail_find_user_id_by_name_zip(state, params, rt) {
  const [first, last, zip] = params.map((v) => String(v || "").trim().toLowerCase());
  const u = recs(state, "user").find(
    (x) => String(x.first_name).toLowerCase() === first &&
      String(x.last_name).toLowerCase() === last && String(x.zip) === zip);
  if (!u) throw new rt.ToolError("NOT_FOUND", "user not found");
  return u.id;
}

function retail_cancel_pending_order(state, params, rt) {
  const o = find(state, "order", params[0], rt);
  const reason = String(params[1] || "").trim().toLowerCase();
  if (o.status !== "pending") {
    throw deny(state, rt,
      `only a pending order can be cancelled (order ${o.id} is ${o.status})`);
  }
  if (!RETAIL_REASONS.includes(reason)) {
    throw deny(state, rt, "the cancellation reason must be 'no longer " +
      "needed' or 'ordered by mistake'");
  }
  const pm = recs(state, "payment_method").find((x) => x.id === o.payment_method);
  if (pm && pm.kind === "gift_card") pm.balance += o.total;
  o.status = "cancelled";
  o.cancel_reason = reason;
  return rt.clone(o);
}

function pendingForChange(state, rt, o, what) {
  if (!String(o.status).startsWith("pending")) {
    throw deny(state, rt,
      `only a pending order's ${what} can be modified (order ${o.id} is ${o.status})`);
  }
}

function retail_modify_pending_order_address(state, params, rt) {
  const o = find(state, "order", params[0], rt);
  pendingForChange(state, rt, o, "address");
  ["address1", "address2", "city", "state", "zip"].forEach((f, i) => {
    o[f] = params[i + 1] === null || params[i + 1] === undefined ? "" : params[i + 1];
  });
  return rt.clone(o);
}

function retail_modify_pending_order_payment(state, params, rt) {
  const o = find(state, "order", params[0], rt);
  pendingForChange(state, rt, o, "payment");
  const pm = ownedMethod(state, rt, o.user, params[1]);
  if (pm.id === o.payment_method) {
    throw deny(state, rt, "the new payment method must differ from the current one");
  }
  if (pm.kind === "gift_card" && pm.balance < o.total) {
    throw deny(state, rt,
      `gift card ${pm.id} (${dollars(pm.balance)}) does not cover the order total of ${dollars(o.total)}`);
  }
  const old = recs(state, "payment_method").find((x) => x.id === o.payment_method);
  if (pm.kind === "gift_card") pm.balance -= o.total;
  if (old && old.kind === "gift_card") old.balance += o.total;
  o.payment_method = pm.id;
  return rt.clone(o);
}

/** Shared checks for changing an item to another variant. Returns
 *  {item, order, variant, pm, diff}. */
function itemSwap(state, params, rt) {
  const it = find(state, "order_item", params[0], rt);
  const o = find(state, "order", it.order, rt);
  const v = find(state, "variant", params[1], rt);
  return { it, o, v };
}

function checkSwap(state, rt, it, o, v, pmId) {
  if (it.status !== "ordered") {
    throw bad(rt, `item ${it.id} is already part of a request (${it.status})`);
  }
  if (v.product !== it.product) {
    throw deny(state, rt,
      `an item can only be changed to another option of the same product (${it.name})`);
  }
  if (v.id === it.variant) {
    throw bad(rt, "the new item must differ from the current one");
  }
  if (!v.available) {
    throw deny(state, rt, `variant ${v.id} of ${it.name} is not available`);
  }
  const pm = ownedMethod(state, rt, o.user, pmId);
  const diff = v.price - it.price;
  if (pm.kind === "gift_card" && pm.balance < diff) {
    throw deny(state, rt,
      `gift card ${pm.id} (${dollars(pm.balance)}) does not cover the price difference of ${dollars(diff)}`);
  }
  return { pm, diff };
}

function retail_modify_pending_order_items(state, params, rt) {
  const { it, o, v } = itemSwap(state, params, rt);
  if (o.locked) {
    throw deny(state, rt,
      `the items of order ${o.id} were already modified once and cannot be modified again`);
  }
  if (o.status !== "pending" && o.status !== "pending (item modified)") {
    throw deny(state, rt,
      `only a pending order's items can be modified (order ${o.id} is ${o.status})`);
  }
  const { pm, diff } = checkSwap(state, rt, it, o, v, params[2]);
  if (pm.kind === "gift_card") pm.balance -= diff;
  it.variant = v.id;
  it.price = v.price;
  it.status = "modified";
  it.payment_method = pm.id;
  o.total += diff;
  o.status = "pending (item modified)";
  return rt.clone(o);
}

function deliveredFor(state, rt, o, kind) {
  // several items of one order in one program are one request (the order's
  // status already says so); an order locked by an earlier request, or with
  // a request of the other kind, refuses
  const same = kind === "exchange" ? "exchange requested" : "return requested";
  const other = kind === "exchange" ? "return requested" : "exchange requested";
  if (o.locked || o.status === other) {
    throw deny(state, rt,
      `a delivered order can be returned or exchanged only once (order ${o.id} is ${o.status})`);
  }
  if (o.status !== "delivered" && o.status !== same) {
    throw deny(state, rt,
      `only a delivered order can be ${kind === "exchange" ? "exchanged" : "returned"} (order ${o.id} is ${o.status})`);
  }
  return same;
}

function retail_exchange_delivered_order_items(state, params, rt) {
  const { it, o, v } = itemSwap(state, params, rt);
  const next = deliveredFor(state, rt, o, "exchange");
  const { pm } = checkSwap(state, rt, it, o, v, params[2]);
  it.status = "exchange requested";
  it.new_variant = v.id;
  it.payment_method = pm.id;
  o.status = next;
  return rt.clone(o);
}

function retail_return_delivered_order_items(state, params, rt) {
  const it = find(state, "order_item", params[0], rt);
  const o = find(state, "order", it.order, rt);
  const next = deliveredFor(state, rt, o, "return");
  if (it.status !== "ordered") {
    throw bad(rt, `item ${it.id} is already part of a request (${it.status})`);
  }
  const pm = ownedMethod(state, rt, o.user, params[1]);
  if (pm.id !== o.payment_method && pm.kind !== "gift_card") {
    throw deny(state, rt,
      "the refund must go to the original payment method or to a gift card");
  }
  it.status = "return requested";
  it.payment_method = pm.id;
  o.status = next;
  return rt.clone(o);
}

function retail_modify_user_address(state, params, rt) {
  const u = find(state, "user", params[0], rt);
  ["address1", "address2", "city", "state", "zip"].forEach((f, i) => {
    u[f] = params[i + 1] === null || params[i + 1] === undefined ? "" : params[i + 1];
  });
  return rt.clone(u);
}

// ----------------------------------------------------------------- airline

const CABINS = ["basic_economy", "economy", "business"];
const AIRLINE_REASONS = ["change of plan", "airline cancelled flight",
                         "health", "weather", "other"];
const FREE_BAGS = {
  regular: { basic_economy: 0, economy: 1, business: 2 },
  silver: { basic_economy: 1, economy: 2, business: 3 },
  gold: { basic_economy: 2, economy: 3, business: 4 },
};
const DAY = 86400;

function segmentsOf(state, resId) {
  return recs(state, "segment").filter((s) => s.reservation === resId);
}

function segFlight(state, seg) {
  return recs(state, "flight").find((f) => f.id === seg.flight) || null;
}

function segFlown(state, seg) {
  const f = segFlight(state, seg);
  return !!f && (f.status === "landed" || f.status === "flying");
}

function flownAny(state, resId) {
  return segmentsOf(state, resId).some((s) => segFlown(state, s));
}

function payForChange(state, rt, r, pmId, amount) {
  const pm = ownedMethod(state, rt, r.user, pmId);
  if (pm.kind === "certificate") {
    throw deny(state, rt, "a travel certificate cannot pay for a change; use a gift card or credit card");
  }
  if (pm.kind === "gift_card" && amount > 0 && pm.balance < amount) {
    throw deny(state, rt,
      `gift card ${pm.id} (${dollars(pm.balance)}) does not cover ${dollars(amount)}`);
  }
  return pm;
}

function settle(state, pm, amount) {
  if (amount === 0) return;
  if (pm.kind === "gift_card") pm.balance -= amount;
  state.payments.push({ payment_method: pm.id, amount });
}

function airline_search_direct_flight(state, params, rt) {
  const [o, d, date] = params;
  return rt.clone(recs(state, "flight").filter(
    (f) => f.origin === o && f.destination === d && f.date === date));
}

function airline_cancel_reservation(state, params, rt) {
  const r = find(state, "reservation", params[0], rt);
  const reason = String(params[1] || "").trim().toLowerCase();
  if (r.status === "cancelled") throw bad(rt, `reservation ${r.id} is already cancelled`);
  if (!AIRLINE_REASONS.includes(reason)) {
    throw bad(rt, `unknown cancellation reason '${params[1]}'`);
  }
  if (flownAny(state, r.id)) {
    throw deny(state, rt,
      `part of reservation ${r.id} has already been flown; it cannot be cancelled`);
  }
  const recent = rt.now - r.created <= DAY;
  const airlineCancelled = segmentsOf(state, r.id).some((s) => {
    const f = segFlight(state, s);
    return f && f.status === "cancelled";
  });
  const insured = r.insurance && (reason === "health" || reason === "weather");
  if (!(recent || airlineCancelled || r.cabin === "business" || insured)) {
    throw deny(state, rt,
      `reservation ${r.id} can be cancelled only within 24 hours of booking, ` +
      "for an airline-cancelled flight, in business class, or with insurance " +
      "and a health or weather reason");
  }
  r.status = "cancelled";
  r.cancel_reason = reason;
  return rt.clone(r);
}

function priceFor(f, cabin) {
  return f[`price_${cabin}`];
}

function airline_update_reservation_cabin(state, params, rt) {
  const r = find(state, "reservation", params[0], rt);
  const cabin = params[1];
  if (!CABINS.includes(cabin)) throw bad(rt, `unknown cabin '${cabin}'`);
  if (r.status === "cancelled") throw bad(rt, `reservation ${r.id} is cancelled`);
  if (cabin === r.cabin) throw bad(rt, `reservation ${r.id} is already ${cabin}`);
  if (flownAny(state, r.id)) {
    throw deny(state, rt,
      `the cabin cannot be changed once a flight of reservation ${r.id} has been flown`);
  }
  const segs = segmentsOf(state, r.id);
  let diff = 0;
  const prices = [];
  for (const s of segs) {
    const f = segFlight(state, s);
    if (!f || f.status !== "available") {
      throw deny(state, rt,
        `flight ${s.flight} is not open for booking, so its cabin cannot change`);
    }
    prices.push(priceFor(f, cabin));
    diff += (priceFor(f, cabin) - s.price) * r.passenger_count;
  }
  const pm = payForChange(state, rt, r, params[2], diff);
  segs.forEach((s, i) => { s.price = prices[i]; });
  settle(state, pm, diff);
  r.cabin = cabin;
  r.changed = true;
  return rt.clone(r);
}

function airline_update_reservation_flights(state, params, rt) {
  const s = find(state, "segment", params[0], rt);
  const r = find(state, "reservation", s.reservation, rt);
  const f = find(state, "flight", params[1], rt);
  if (r.status === "cancelled") throw bad(rt, `reservation ${r.id} is cancelled`);
  if (r.cabin === "basic_economy") {
    throw deny(state, rt, `basic economy flights cannot be modified (reservation ${r.id})`);
  }
  if (segFlown(state, s)) {
    throw deny(state, rt, `flight ${s.flight} has already departed and cannot be changed`);
  }
  if (f.origin !== s.origin || f.destination !== s.destination) {
    throw deny(state, rt,
      "a change cannot alter the origin, destination or trip type of a reservation");
  }
  if (f.id === s.flight) throw bad(rt, "the new flight must differ from the current one");
  if (f.status !== "available") {
    throw deny(state, rt, `flight ${f.id} is not available for booking (${f.status})`);
  }
  const price = priceFor(f, r.cabin);
  const diff = (price - s.price) * r.passenger_count;
  const pm = payForChange(state, rt, r, params[2], diff);
  s.flight = f.id;
  s.date = f.date;
  s.price = price;
  settle(state, pm, diff);
  r.changed = true;
  return rt.clone(r);
}

function airline_update_reservation_baggages(state, params, rt) {
  const r = find(state, "reservation", params[0], rt);
  const total = params[1];
  if (!Number.isInteger(total) || total < 0) throw bad(rt, "total_baggages must be a count");
  if (r.status === "cancelled") throw bad(rt, `reservation ${r.id} is cancelled`);
  if (total < r.total_baggages) {
    throw deny(state, rt,
      `checked bags can be added but not removed (reservation ${r.id} has ${r.total_baggages})`);
  }
  const u = find(state, "user", r.user, rt);
  const free = ((FREE_BAGS[u.membership] || FREE_BAGS.regular)[r.cabin] || 0) * r.passenger_count;
  const nonfree = Math.max(0, total - free);
  const cost = Math.max(0, nonfree - r.nonfree_baggages) * 5000;
  const pm = payForChange(state, rt, r, params[2], cost);
  settle(state, pm, cost);
  r.total_baggages = total;
  r.nonfree_baggages = Math.max(nonfree, r.nonfree_baggages);
  return rt.clone(r);
}

function airline_update_reservation_passengers(state, params, rt) {
  const p = find(state, "passenger", params[0], rt);
  const r = find(state, "reservation", p.reservation, rt);
  if (r.status === "cancelled") throw bad(rt, `reservation ${r.id} is cancelled`);
  ["first_name", "last_name", "dob"].forEach((f, i) => {
    if (params[i + 1] !== null && params[i + 1] !== undefined) p[f] = params[i + 1];
  });
  return rt.clone(r);
}

function airline_send_certificate(state, params, rt) {
  const r = find(state, "reservation", params[0], rt);
  const amount = params[1];
  const u = find(state, "user", r.user, rt);
  if (!(u.membership === "silver" || u.membership === "gold" || r.insurance ||
        r.cabin === "business")) {
    throw deny(state, rt,
      "compensation is only for silver or gold members, travellers with insurance, or business class");
  }
  const statuses = segmentsOf(state, r.id).map((s) => (segFlight(state, s) || {}).status);
  let expected;
  if (statuses.includes("cancelled")) {
    expected = 10000 * r.passenger_count;
  } else if (statuses.includes("delayed")) {
    if (!(r.status === "cancelled" || r.changed)) {
      throw deny(state, rt,
        "a delayed flight is compensated only after the reservation is changed or cancelled");
    }
    expected = 5000 * r.passenger_count;
  } else {
    throw deny(state, rt,
      `compensation is only for a cancelled or delayed flight, and reservation ${r.id} has neither`);
  }
  if (amount !== expected) {
    throw deny(state, rt,
      `the certificate must be ${dollars(expected)} ($100 per passenger for a cancellation, $50 for a delay)`);
  }
  const n = recs(state, "payment_method").filter((x) => x.kind === "certificate").length + 1;
  const cert = { id: `certificate_new_${n}`, user: u.id, kind: "certificate",
                 balance: amount, last_four: "" };
  recs(state, "payment_method").push(cert);
  return rt.clone(cert);
}

// ----------------------------------------------------------------- telecom

const NETWORK_MODES = ["4g_5g_preferred", "4g_only", "3g_only", "2g_only"];

function telecom_get_customer_by_phone(state, params, rt) {
  const num = String(params[0] || "").trim();
  const c = recs(state, "customer").find((x) => x.phone_number === num);
  if (c) return rt.clone(c);
  const line = recs(state, "line").find((x) => x.phone_number === num);
  if (line) return rt.clone(find(state, "customer", line.customer, rt));
  throw new rt.ToolError("NOT_FOUND", "customer not found");
}

function telecom_send_payment_request(state, params, rt) {
  const b = find(state, "bill", params[0], rt);
  if (b.status !== "Overdue") {
    throw deny(state, rt, `only an overdue bill can be sent for payment (bill ${b.id} is ${b.status})`);
  }
  if (recs(state, "bill").some((x) => x.customer === b.customer && x.status === "Awaiting Payment")) {
    throw deny(state, rt, "a customer can have only one bill awaiting payment at a time");
  }
  b.status = "Awaiting Payment";
  return rt.clone(b);
}

function telecom_make_payment(state, params, rt) {
  const b = find(state, "bill", params[0], rt);
  if (b.status !== "Awaiting Payment") {
    throw bad(rt, `bill ${b.id} has no payment request to accept (${b.status})`);
  }
  b.status = "Paid";
  return rt.clone(b);
}

function telecom_resume_line(state, params, rt) {
  const l = find(state, "line", params[0], rt);
  if (l.status !== "Suspended") throw bad(rt, `line ${l.id} is not suspended`);
  if (String(l.contract_end) < String(rt.today || "")) {
    throw deny(state, rt,
      `line ${l.id}'s contract ended on ${l.contract_end}; its suspension cannot be lifted`);
  }
  if (recs(state, "bill").some((b) => b.customer === l.customer &&
      (b.status === "Overdue" || b.status === "Awaiting Payment"))) {
    throw deny(state, rt, "every overdue bill must be paid before the suspension is lifted");
  }
  l.status = "Active";
  for (const p of recs(state, "phone")) if (p.line === l.id) p.needs_reboot = true;
  return rt.clone(l);
}

function telecom_enable_roaming(state, params, rt) {
  const l = find(state, "line", params[0], rt);
  l.roaming_enabled = true;
  return rt.clone(l);
}

function telecom_disable_roaming(state, params, rt) {
  const l = find(state, "line", params[0], rt);
  l.roaming_enabled = false;
  return rt.clone(l);
}

function telecom_refuel_data(state, params, rt) {
  const l = find(state, "line", params[0], rt);
  const gb = params[1];
  if (!Number.isInteger(gb) || gb <= 0) throw bad(rt, "gb_amount must be a positive number of GB");
  if (gb > 2) throw deny(state, rt, `at most 2 GB can be refuelled (asked for ${gb} GB)`);
  if (l.status !== "Active") throw deny(state, rt, `line ${l.id} is ${l.status}; data cannot be refuelled`);
  l.data_refuel_gb += gb;
  return rt.clone(l);
}

function phoneOp(fn) {
  return (state, params, rt) => {
    const p = find(state, "phone", params[0], rt);
    fn(p, params, state, rt);
    return rt.clone(p);
  };
}

const telecom_toggle_airplane_mode = phoneOp((p) => { p.airplane_mode = !p.airplane_mode; });
const telecom_toggle_wifi_calling = phoneOp((p) => { p.wifi_calling = !p.wifi_calling; });
const telecom_toggle_data = phoneOp((p) => { p.mobile_data = !p.mobile_data; });
const telecom_toggle_roaming = phoneOp((p) => { p.roaming = !p.roaming; });
const telecom_toggle_data_saver_mode = phoneOp((p) => { p.data_saver = !p.data_saver; });
const telecom_disconnect_vpn = phoneOp((p) => { p.vpn = false; });
const telecom_reset_apn_settings = phoneOp((p) => { p.apn = "reset (reboot to apply)"; });
const telecom_reboot_device = phoneOp((p) => {
  if (String(p.apn).startsWith("reset")) p.apn = "default";
  p.needs_reboot = false;
});
const telecom_reseat_sim_card = phoneOp((p, params, state, rt) => {
  if (p.sim === "pin_locked") {
    throw deny(state, rt,
      "the SIM is PIN-locked; unlocking it needs a human agent, reseating will not help");
  }
  p.sim = "seated";
});
const telecom_set_network_mode_preference = phoneOp((p, params, state, rt) => {
  const mode = String(params[1] || "");
  if (!NETWORK_MODES.includes(mode)) throw bad(rt, `unknown network mode '${mode}'`);
  p.network_mode = mode;
});
const telecom_grant_app_permission = phoneOp((p, params, state, rt) => {
  const app = String(params[1] || "").trim().toLowerCase();
  const perm = String(params[2] || "").trim().toLowerCase();
  if (app !== "messaging") throw bad(rt, `unknown app '${params[1]}'`);
  if (perm !== "sms" && perm !== "storage") throw bad(rt, `unknown permission '${params[2]}'`);
  p[`${perm}_permission`] = true;
});

// the telecom clock: tau2's policy date, as the date string the contract end
// dates are compared against (rt.now is the epoch the sandbox passes)
function withToday(fn) {
  return (state, params, rt) => fn(state, params, Object.assign({}, rt, {
    today: new Date(rt.now * 1000).toISOString().slice(0, 10),
  }));
}

module.exports = {
  retail_find_user_id_by_email, retail_find_user_id_by_name_zip,
  retail_cancel_pending_order, retail_modify_pending_order_address,
  retail_modify_pending_order_payment, retail_modify_pending_order_items,
  retail_exchange_delivered_order_items, retail_return_delivered_order_items,
  retail_modify_user_address,
  airline_search_direct_flight, airline_cancel_reservation,
  airline_update_reservation_cabin, airline_update_reservation_flights,
  airline_update_reservation_baggages, airline_update_reservation_passengers,
  airline_send_certificate,
  telecom_get_customer_by_phone, telecom_send_payment_request,
  telecom_make_payment, telecom_resume_line: withToday(telecom_resume_line),
  telecom_enable_roaming, telecom_disable_roaming, telecom_refuel_data,
  telecom_toggle_airplane_mode, telecom_toggle_wifi_calling, telecom_toggle_data,
  telecom_toggle_roaming, telecom_toggle_data_saver_mode, telecom_disconnect_vpn,
  telecom_reset_apn_settings, telecom_reboot_device, telecom_reseat_sim_card,
  telecom_set_network_mode_preference, telecom_grant_app_permission,
};
