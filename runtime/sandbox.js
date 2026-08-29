#!/usr/bin/env node
"use strict";
/*
 * Agent Core sandbox (F3).
 *
 * Reads one JSON job from stdin:
 *   { js, state, tools, fields, constants, now, approval,
 *     error_injection, initial_registers, max_ops }
 * Executes the compiled program in a bare `vm` context (no require, no fs,
 * no network — the only host object handed in is `rt`), against an in-memory
 * world state via a generic, data-driven tool engine.
 *
 * Writes one JSON result to stdout:
 *   { status: ok|paused|error|effect_blocked,
 *     return_value, registers, state, calls, error, ops }
 *
 * Effect gate: any DELETE/SEND/PAY call without `approval` halts the run
 * with EFFECT_BLOCKED (not catchable by TRY).
 */

const vm = require("node:vm");

const DESTRUCTIVE = new Set(["DELETE", "SEND", "PAY"]);
const WATCHDOG_MS = 5000;

class ToolError extends Error {
  constructor(code, message) {
    super(message || code);
    this.code = code;
  }
}

class EffectBlocked extends Error {
  constructor(tool, effect) {
    super(`EFFECT_BLOCKED ${effect} ${tool}`);
    this.tool = tool;
    this.effect = effect;
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
  const maxOps = input.max_ops || 100000;
  const injections = (input.error_injection || []).map((e) => ({
    tool: e.tool || null, name: e.name || null, code: e.code,
    times: e.times === undefined ? 1 : e.times,
  }));

  const calls = [];
  let ops = 0;
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
          rec[fname] = params[i];
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
          rec[k] = params[i];
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
      case "pay": {
        const entry = {};
        (impl.param_map || []).forEach((key, i) => { entry[key] = params[i]; });
        state.payments.push(entry);
        return null;
      }
      default:
        throw new ToolError("INVALID_ARGUMENT", `bad impl op ${impl.op}`);
    }
  }

  const rt = {
    ToolError,
    isToolError: (e) => e instanceof ToolError,
    async call(sym, args) {
      budget();
      const tool = toolsBySym.get(sym);
      if (!tool) throw new ToolError("NOT_FOUND", `unknown tool ${sym}`);
      for (const eff of tool.effects || []) {
        if (DESTRUCTIVE.has(eff) && !approval) {
          throw new EffectBlocked(sym, eff);
        }
      }
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
      switch (op) {
        case "EQ": return a === b;
        case "LT": return a < b;
        case "GT": return a > b;
        case "CONTAINS":
          if (typeof a === "string") return a.includes(String(b));
          if (Array.isArray(a)) return a.map(narrowId).includes(b);
          return false;
        default: throw new Error(`bad cmp ${op}`);
      }
    },
    mapF(list, sym) {
      budget();
      const f = fieldsBySym.get(sym);
      if (!f) throw new Error(`unknown field ${sym}`);
      return list.map((x) => (x[f.name] === undefined ? null : x[f.name]));
    },
    count: (list) => list.length,
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

  const finish = (result) => {
    process.stdout.write(JSON.stringify(result) + "\n");
    process.exit(0);
  };

  const watchdog = setTimeout(() => {
    finish({ status: "error", error: { code: "TIMEOUT", message: "watchdog" },
             state, calls, ops });
  }, WATCHDOG_MS);

  Promise.resolve()
    .then(() => programMain(rt))
    .then((marker) => {
      clearTimeout(watchdog);
      const base = { state, calls, ops };
      if (marker && marker.__kind === "pause") {
        finish({ status: "paused", registers: marker.regs, ...base });
      } else if (marker && marker.__kind === "return") {
        finish({ status: "ok", return_value: marker.value, ...base });
      } else {
        finish({ status: "ok", return_value: null, ...base });
      }
    })
    .catch((e) => {
      clearTimeout(watchdog);
      const base = { state, calls, ops };
      if (e instanceof EffectBlocked) {
        finish({ status: "effect_blocked",
                 error: { code: "EFFECT_BLOCKED", tool: e.tool,
                          effect: e.effect }, ...base });
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
