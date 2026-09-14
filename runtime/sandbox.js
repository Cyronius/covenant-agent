#!/usr/bin/env node
"use strict";
/*
 * Agent Core sandbox (F3).
 *
 * Reads one JSON job from stdin:
 *   { js, state, tools, fields, constants, now, approval,
 *     error_injection, initial_registers, max_ops, post_hook }
 * Executes the compiled program in a bare `vm` context (no require, no fs,
 * no network — the only host object handed in is `rt`), against an in-memory
 * world state via a generic, data-driven tool engine.
 *
 * Two extension points beyond the generic CRUD ops, for worlds whose rules
 * are not CRUD (the `rpg` grid world):
 *   impl {op:"engine", module, fn}  — dispatch the call to a whitelisted
 *     module under runtime/engines/. Loaded in the host, never in the vm
 *     context, so the program still sees no require/fs/network.
 *   post_hook {module, fn}          — run once after the program finishes or
 *     halts on a tool error, before the result is written (the RPG's enemy
 *     phase). Skipped when nothing executed (static errors never reach here;
 *     an effect-blocked run is skipped explicitly).
 *
 * Writes one JSON result to stdout:
 *   { status: ok|paused|aborted|error|effect_blocked,
 *     return_value, registers, state, calls, error, ops }
 *
 * Effect gate: any DELETE/SEND/PAY call without `approval` halts the run
 * with EFFECT_BLOCKED (not catchable by TRY).
 *
 * Preview runs (opt-in, `preview`): halting at the first destructive call
 * previews one call but hands back a token good for the whole program - one
 * click on "delete_card #2" deleted seven cards (2026-09-14). With `preview`
 * an unapproved run doesn't halt: it runs to the end against the state clone
 * it already works on (which the caller discards and re-sends once a person
 * clicks), and the result becomes effect_blocked carrying every call that
 * needed the approval - DESTRUCTIVE if any DELETE/SEND/PAY is among them,
 * else BULK_WRITE when the writes exceed `bulk_write_limit`.
 */

const vm = require("node:vm");
const path = require("node:path");

const DESTRUCTIVE = new Set(["DELETE", "SEND", "PAY"]);
const WATCHDOG_MS = 5000;
// Rule modules loadable by the `engine` op / post_hook. A hardcoded list, not
// the world registry: harness/run.py honors a task's own `sandbox` payload,
// impl descriptors included, so the whitelist is what actually guards this.
const ENGINES = new Set(["rpg", "warehouse", "elevator", "cards", "house",
                         "page"]);

function loadEngine(name) {
  if (!ENGINES.has(name)) {
    throw new ToolError("INVALID_ARGUMENT", `unknown engine ${name}`);
  }
  // __dirname, not cwd: harness/run.py spawns node without a cwd.
  return require(path.join(__dirname, "engines", `${name}.js`));
}

class ToolError extends Error {
  constructor(code, message) {
    super(message || code);
    this.code = code;
  }
}

class EffectBlocked extends Error {
  constructor(tool, effect, args) {
    super(`EFFECT_BLOCKED ${effect} ${tool}`);
    this.tool = tool;
    this.effect = effect;
    this.args = args; // the call's narrowed params, for approval previews
  }
}

function clone(v) {
  return v === undefined ? undefined : JSON.parse(JSON.stringify(v));
}

function narrowId(v) {
  if (v && typeof v === "object" && !Array.isArray(v) && "id" in v) return v.id;
  return v;
}

function main(input) {
  const state = clone(input.state);
  state.entities = state.entities || {};
  state.outbox = state.outbox || [];
  state.payments = state.payments || [];

  const toolsBySym = new Map();
  for (const t of input.tools || []) toolsBySym.set(t.sym, t);
  const fieldsBySym = new Map();
  for (const f of input.fields || []) fieldsBySym.set(f.sym, f);
  const constants = input.constants || {};
  const now = input.now || 0;
  const approval = !!input.approval;
  const preview = !!input.preview;
  const maxOps = input.max_ops || 100000;
  const injections = (input.error_injection || []).map((e) => ({
    tool: e.tool || null, name: e.name || null, code: e.code,
    times: e.times === undefined ? 1 : e.times,
  }));

  const calls = [];
  let ops = 0;
  let line = 0;  // source line of the instruction now executing (rt.at)
  const idCounters = new Map();

  function budget() {
    ops += 1;
    if (ops > maxOps) throw new ToolError("RATE_LIMITED", "op budget exceeded");
  }

  function nextId(entity) {
    if (!idCounters.has(entity)) {
      let max = 0;
      for (const rec of state.entities[entity] || []) {
        const m = /^.*_(\d+)$/.exec(String(rec.id));
        if (m) max = Math.max(max, parseInt(m[1], 10));
      }
      idCounters.set(entity, max);
    }
    const n = idCounters.get(entity) + 1;
    idCounters.set(entity, n);
    return `${entity}_${n}`;
  }

  function findRecord(entity, id) {
    const list = state.entities[entity] || [];
    return list.find((r) => r.id === id);
  }

  function checkInjection(tool) {
    for (const inj of injections) {
      const match = (inj.tool && inj.tool === tool.sym) ||
                    (inj.name && inj.name === tool.name);
      if (match && inj.times !== 0) {
        if (inj.times > 0) inj.times -= 1;
        return inj.code;
      }
    }
    return null;
  }

  function runImpl(tool, params) {
    const impl = tool.impl;
    switch (impl.op) {
      case "list":
        return clone(state.entities[impl.entity] || []);
      case "list_by": {
        // records of `entity` whose `field` equals the id param (a
        // container's items: a lesson's elements, an item's notes)
        const key = params[impl.id_param];
        return clone((state.entities[impl.entity] || []).filter((r) => r[impl.field] === key));
      }
      case "get": {
        const rec = findRecord(impl.entity, params[impl.id_param]);
        if (!rec) throw new ToolError("NOT_FOUND", `${impl.entity} not found`);
        return clone(rec);
      }
      case "create": {
        const rec = { id: nextId(impl.entity) };
        for (const [k, v] of Object.entries(impl.defaults || {})) {
          rec[k] = v === "$now" ? now : v;
        }
        (impl.param_fields || []).forEach((fname, i) => {
          // null slot = a param with no field (e.g. a position hint); an
          // omitted optional param keeps the default
          if (fname && params[i] !== undefined && params[i] !== null) rec[fname] = params[i];
        });
        if (!state.entities[impl.entity]) state.entities[impl.entity] = [];
        state.entities[impl.entity].push(rec);
        return clone(rec);
      }
      case "update": {
        const rec = findRecord(impl.entity, params[impl.id_param]);
        if (!rec) throw new ToolError("NOT_FOUND", `${impl.entity} not found`);
        for (const [k, v] of Object.entries(impl.set_const || {})) rec[k] = v;
        for (const [k, i] of Object.entries(impl.set_from_params || {})) {
          // an omitted optional param leaves the field alone (coursebuilder's
          // flattened props are all optional)
          if (params[i] !== undefined && params[i] !== null) rec[k] = params[i];
        }
        return clone(rec);
      }
      case "delete": {
        const list = state.entities[impl.entity] || [];
        const i = list.findIndex((r) => r.id === params[impl.id_param]);
        if (i < 0) throw new ToolError("NOT_FOUND", `${impl.entity} not found`);
        list.splice(i, 1);
        return null;
      }
      case "send": {
        const entry = { channel: impl.channel };
        (impl.param_map || []).forEach((key, i) => { entry[key] = params[i]; });
        state.outbox.push(entry);
        return null;
      }
      case "external": {
        // EXTERNAL tools (spec §7) run outside the sandbox. With
        // input.external_url set (the dev server), POST {kind, params} and
        // use the text that comes back; otherwise return a deterministic
        // stub so eval scores routing, not prose.
        if (input.external_url) {
          const { spawnSync } = require("node:child_process");
          const script = "const [u,b]=[process.argv[1],process.argv[2]];" +
            "fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:b})" +
            ".then(r=>r.text()).then(t=>process.stdout.write(t))" +
            ".catch(e=>{process.stderr.write(String(e));process.exit(1)});";
          const body = JSON.stringify({ kind: impl.kind, params });
          const r = spawnSync(process.execPath, ["-e", script, input.external_url, body],
                              { encoding: "utf8", timeout: 120000 });
          if (r.status !== 0) {
            throw new ToolError("EXTERNAL_FAILED", `${impl.kind}: ${(r.stderr || "").slice(0, 200)}`);
          }
          const out = JSON.parse(r.stdout);
          if (out.error) throw new ToolError("EXTERNAL_FAILED", String(out.error.message || out.error));
          return out.text;
        }
        return `[${impl.kind}: ${String(params[0])}]`;
      }
      case "pay": {
        const entry = {};
        (impl.param_map || []).forEach((key, i) => { entry[key] = params[i]; });
        state.payments.push(entry);
        return null;
      }
      case "noop": {
        // Decoy tools (harness/decoys.py): they take the reference tool's
        // signature and a neighbouring description, and they change nothing.
        // A model that picks one on description alone therefore fails the
        // task's state check rather than quietly corrupting the world.
        if (impl.entity !== undefined && impl.id_param !== undefined) {
          const rec = findRecord(impl.entity, params[impl.id_param]);
          if (!rec) throw new ToolError("NOT_FOUND", `${impl.entity} not found`);
          return clone(rec);
        }
        return null;
      }
      case "engine": {
        // Non-CRUD world rules (runtime/engines/<module>.js). The module
        // mutates `state` in place and returns the tool's value; it must
        // validate before mutating, since a thrown ToolError still leaves
        // this state as the run's result.
        const mod = loadEngine(impl.module);
        const fn = mod[impl.fn];
        if (typeof fn !== "function") {
          throw new ToolError("INVALID_ARGUMENT",
                              `engine ${impl.module} has no ${impl.fn}`);
        }
        return fn(state, params, { now, ToolError, clone });
      }
      default:
        throw new ToolError("INVALID_ARGUMENT", `bad impl op ${impl.op}`);
    }
  }

  // registers bound so far, for the error report: the compiled program
  // installs rt.snapshot over its own locals (core/compile.py emit)
  const boundRegisters = () => {
    const out = {};
    if (typeof rt.snapshot !== "function") return out;
    for (const [k, v] of Object.entries(rt.snapshot())) {
      if (v !== undefined) out[k] = v;
    }
    return out;
  };

  const rt = {
    ToolError,
    isToolError: (e) => e instanceof ToolError,
    snapshot: null,
    at(n) { line = n; },
    async call(sym, args) {
      budget();
      const tool = toolsBySym.get(sym);
      if (!tool) throw new ToolError("NOT_FOUND", `unknown tool ${sym}`);
      // narrow + validate params
      const params = [];
      for (let i = 0; i < (tool.params || []).length; i++) {
        const p = tool.params[i];
        let v = args[i];
        if (v === undefined) {
          if (p.required !== false) {
            throw new ToolError("INVALID_ARGUMENT", `missing ${p.name}`);
          }
          v = null;
        }
        if (typeof p.type === "string" && p.type.startsWith("ID:")) {
          v = narrowId(v);
        }
        params.push(v);
      }
      for (const eff of tool.effects || []) {
        if (DESTRUCTIVE.has(eff) && !approval && !preview) {
          // not logged: the reference (approved) run has no such entry and
          // unnecessary_destructive diffs the two logs
          throw new EffectBlocked(sym, eff, clone(params));
        }
      }
      const entry = { tool: sym, name: tool.name, args: clone(params),
                      ok: true, error: null };
      calls.push(entry);
      const injected = checkInjection(tool);
      if (injected) {
        entry.ok = false;
        entry.error = injected;
        throw new ToolError(injected, `injected on ${tool.name}`);
      }
      try {
        return runImpl(tool, params);
      } catch (e) {
        entry.ok = false;
        entry.error = e instanceof ToolError ? e.code : String(e.message);
        throw e;
      }
    },
    constant(sym) {
      if (!(sym in constants)) throw new Error(`unknown constant ${sym}`);
      return clone(constants[sym]);
    },
    initial(rname) {
      const regs = input.initial_registers || {};
      return rname in regs ? clone(regs[rname]) : undefined;
    },
    now: () => now,
    fld(x, sym) {
      budget();
      if (x === null || x === undefined) {
        throw new Error(`field access ${sym} on NULL`);
      }
      const f = fieldsBySym.get(sym);
      if (!f) throw new Error(`unknown field ${sym}`);
      const v = x[f.name];
      return v === undefined ? null : v;
    },
    setF(x, sym, v) {
      budget();
      const f = fieldsBySym.get(sym);
      if (!f) throw new Error(`unknown field ${sym}`);
      return { ...x, [f.name]: v };
    },
    cmp(op, a, b) {
      budget();
      a = narrowId(a);
      b = narrowId(b);
      // spec 0.4.0 §4: STR comparison is case-folded and trimmed. The
      // model never chooses this, so it cannot get it wrong; "cyrus"
      // finds "Cyrus" and the NOT_FOUND path stops depending on the
      // user's shift key. Ids and enum values are canonical already.
      const norm = (v) => (typeof v === "string" ? v.trim().toLowerCase() : v);
      switch (op) {
        case "EQ":
          if (typeof a === "string" && typeof b === "string") return norm(a) === norm(b);
          return a === b;
        case "LT": return a < b;
        case "GT": return a > b;
        case "CONTAINS":
          // substring only since 0.6.0; membership is IN
          if (typeof a === "string") return norm(a).includes(norm(String(b)));
          return false;
        case "IN":
          if (!Array.isArray(b)) return false;
          return b.map(narrowId).some((x) => (typeof x === "string" &&
            typeof a === "string" ? norm(x) === norm(a) : x === a));
        default: throw new Error(`bad cmp ${op}`);
      }
    },
    // spec 0.4.0 §4 EMPTY: an empty list or NULL
    empty(v) {
      budget();
      return v === null || v === undefined || (Array.isArray(v) && v.length === 0);
    },
    // spec 0.4.0 §4 MOST / LEAST (sign +1 / -1): the value of `sym` shared
    // by the most / fewest elements of `list`. With `cands`, the keys are
    // the candidates' ids (in candidate order, zeros included) and elements
    // keyed elsewhere are ignored; without, keys are the values present in
    // element order. Ties go to the first key seen. NULL when no keys.
    extremeBy(list, sym, cands, sign) {
      budget();
      const f = fieldsBySym.get(sym);
      if (!f) throw new Error(`unknown field ${sym}`);
      const counts = new Map();
      if (Array.isArray(cands)) {
        for (const c of cands) counts.set(narrowId(c), 0);
      }
      for (const x of list) {
        const k = narrowId(x[f.name] === undefined ? null : x[f.name]);
        if (Array.isArray(cands)) {
          if (counts.has(k)) counts.set(k, counts.get(k) + 1);
        } else {
          counts.set(k, (counts.get(k) || 0) + 1);
        }
      }
      let best = null;
      let bestN = null;
      for (const [k, n] of counts) {
        if (bestN === null || (sign > 0 ? n > bestN : n < bestN)) {
          best = k;
          bestN = n;
        }
      }
      return best;
    },
    mapF(list, sym) {
      budget();
      const f = fieldsBySym.get(sym);
      if (!f) throw new Error(`unknown field ${sym}`);
      return list.map((x) => (x[f.name] === undefined ? null : x[f.name]));
    },
    count: (list) => list.length,
    // spec §4 FORMAT: fill {i} slots. TIME renders as an ISO date (UTC);
    // objects render as their id; null as empty.
    format(template, values, kinds) {
      budget();
      const render = (v, kind) => {
        if (v === null || v === undefined) return "";
        if (kind === "TIME" && typeof v === "number") {
          return new Date(v * 1000).toISOString().slice(0, 10);
        }
        if (typeof v === "object") return v.id !== undefined ? String(v.id) : JSON.stringify(v);
        return String(v);
      };
      return String(template).replace(/\{(\d+)\}/g, (_m, i) =>
        render(values[Number(i)], (kinds || [])[Number(i)]));
    },
    sortBy(list, sym, dir) {
      budget();
      const f = fieldsBySym.get(sym);
      if (!f) throw new Error(`unknown field ${sym}`);
      const out = list.slice();
      out.sort((x, y) => {
        const a = x[f.name];
        const b = y[f.name];
        let c = 0;
        if (a < b) c = -1;
        else if (a > b) c = 1;
        return dir === "DESC" ? -c : c;
      });
      return out;
    },
    select(list, i) {
      budget();
      if (i < 0 || i >= list.length) {
        throw new ToolError("INDEX_OUT_OF_RANGE", `index ${i} of ${list.length}`);
      }
      return list[i];
    },
    first: (list) => (list.length ? list[0] : null),
    ret: (v) => ({ __kind: "return", value: v }),
    stop: () => ({ __kind: "stop" }),
    // spec §4 ABORT: the program declines to act; reason is a closed enum,
    // refs are the symbols it is about (checkable by the harness, askable
    // by a UI).
    abort: (reason, refs) => ({ __kind: "abort", reason, refs: refs || [] }),
    pause(regs) {
      const out = {};
      for (const [k, v] of Object.entries(regs)) {
        if (v !== undefined) out[k] = v;
      }
      return { __kind: "pause", regs: out };
    },
  };

  const context = vm.createContext(Object.create(null));
  const script = new vm.Script(input.js + "\nmain;", { filename: "program.js" });
  const programMain = script.runInContext(context, { timeout: 2000 });

  // Runs after the program ends (normally or on a tool error), before the
  // result is written: the world's per-turn phase, e.g. the RPG's enemies.
  // Never on effect_blocked — nothing executed there, and a preview run's
  // state is thrown away.
  const runPostHook = (status) => {
    const hook = input.post_hook;
    if (!hook || status === "effect_blocked") return null;
    try {
      const mod = loadEngine(hook.module);
      const fn = mod[hook.fn];
      if (typeof fn !== "function") {
        return { code: "INVALID_ARGUMENT",
                 message: `engine ${hook.module} has no ${hook.fn}` };
      }
      fn(state, { now, ToolError, clone });
      return null;
    } catch (e) {
      return { code: e instanceof ToolError ? e.code : "JS_ERROR",
               message: String(e && e.message) };
    }
  };

  // Everything the run changed, in order, with the effect that changed it.
  // DESTRUCTIVE wins over WRITE on a tool that carries both.
  const consequences = () => {
    const out = [];
    for (const c of calls) {
      if (!c.ok) continue;
      const effects = toolsBySym.get(c.tool)?.effects || [];
      const effect = effects.find((e) => DESTRUCTIVE.has(e))
        || (effects.includes("WRITE") ? "WRITE" : null);
      if (effect) out.push({ tool: c.tool, name: c.name, args: c.args, effect });
    }
    return out;
  };

  // What an unapproved run needs a person to look at: any destructive call,
  // or more writes than the caller's bulk limit. Both report the whole list,
  // which is exact because the program ran to the end.
  const gate = () => {
    const changes = consequences();
    const destructive = changes.filter((c) => c.effect !== "WRITE");
    if (destructive.length) {
      return { code: "DESTRUCTIVE", effect: destructive[0].effect };
    }
    const limit = input.bulk_write_limit;
    if (typeof limit === "number" && changes.length > limit) {
      return { code: "BULK_WRITE", effect: "WRITE" };
    }
    return null;
  };

  const finish = (result) => {
    if (preview && !approval &&
        (result.status === "ok" || result.status === "paused")) {
      const blocked = gate();
      if (blocked) {
        const changes = consequences();
        result = {
          status: "effect_blocked",
          error: { ...blocked, tool: null, count: changes.length,
                   calls: changes },
          state, calls, ops,
        };
      }
    }
    const hookError = runPostHook(result.status);
    if (hookError) result.post_hook_error = hookError;
    process.stdout.write(JSON.stringify(result) + "\n");
    process.exit(0);
  };

  const watchdog = setTimeout(() => {
    finish({ status: "error", error: { code: "TIMEOUT", message: "watchdog" },
             state, calls, ops });
  }, input.external_url ? 120000 : WATCHDOG_MS); // external tools block on a model

  Promise.resolve()
    .then(() => programMain(rt))
    .then((marker) => {
      clearTimeout(watchdog);
      const base = { state, calls, ops };
      if (marker && marker.__kind === "pause") {
        finish({ status: "paused", registers: marker.regs, ...base });
      } else if (marker && marker.__kind === "return") {
        finish({ status: "ok", return_value: marker.value, ...base });
      } else if (marker && marker.__kind === "abort") {
        finish({ status: "aborted", reason: marker.reason,
                 refs: marker.refs || [], ...base });
      } else {
        finish({ status: "ok", return_value: null, ...base });
      }
    })
    .catch((e) => {
      clearTimeout(watchdog);
      // a failed segment reports where it stopped and what it had bound, so
      // a planner can continue from the state the sandbox left (reads are
      // free; a failed write did not happen)
      const base = { state, calls, ops, line, registers: boundRegisters() };
      if (e instanceof EffectBlocked) {
        finish({ status: "effect_blocked",
                 error: { code: "EFFECT_BLOCKED", tool: e.tool,
                          effect: e.effect, args: e.args }, ...base });
      } else if (e instanceof ToolError) {
        finish({ status: "error",
                 error: { code: e.code, message: e.message }, ...base });
      } else {
        finish({ status: "error",
                 error: { code: "JS_ERROR", message: String(e && e.message) },
                 ...base });
      }
    });
}

{
  const chunks = [];
  process.stdin.on("data", (c) => chunks.push(c));
  process.stdin.on("end", () => {
    let input;
    try {
      input = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    } catch (e) {
      process.stdout.write(JSON.stringify({
        status: "error", error: { code: "BAD_INPUT", message: String(e) },
      }) + "\n");
      process.exit(0);
    }
    try {
      main(input);
    } catch (e) {
      process.stdout.write(JSON.stringify({
        status: "error", error: { code: "JS_ERROR", message: String(e && e.message) },
      }) + "\n");
      process.exit(0);
    }
  });
}
