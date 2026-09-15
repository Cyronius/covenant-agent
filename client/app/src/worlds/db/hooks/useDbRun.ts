// Database Analyst's agent-run loop. Same shape as
// client/app/src/worlds/kanban/hooks/useAgentRun.ts — model load, a
// freely-typed request -> POST /db_prompt -> grammar-constrained generation
// -> /validate (PAUSE continuation + the real approval gate) -> chat/state
// the UI reads — trimmed of what this world doesn't need: no preflight (no
// tool here assigns to a named person the way kanban's assign_card does),
// no writer/EXTERNAL tool. What it adds: `lastResult`, the last RETURN'd
// value, for the results table — kanban has no equivalent because its UI
// renders the board itself, not a query result.
import { useCallback, useEffect, useRef, useState } from 'react';
import { type LoadStage, type Planner } from '../../../../../shared/llm';
import {
  createPlannerFor,
  fetchModels,
  loadModelPreference,
  loadPreference,
  saveModelPreference,
  savePreference,
  type InferenceMode,
  type ModelInfo,
} from '../../../../../shared/inference';
import { buildFullPrompt, STOP, type Registers } from '../../../../../shared/prompt';
import { validate, type CallLogEntry, type ValidateResponse } from '../../../../../shared/validate';
import { fetchDbPrompt, describeCallSite } from '../lib/dbPrompt';
import { describeCall, describeArg } from '../lib/describe';
import { fetchDbState, TOOL_EFFECTS, type CrmState, type Effect } from '../data/crm';

const APP_KEY = 'db-ui';
const MAX_SEGMENTS = 4;

export type ChatMessage =
  | { kind: 'user'; id: string; text: string }
  | { kind: 'program'; id: string; text: string }
  | { kind: 'tool-call'; id: string; effect: Effect; name: string; detail: string }
  | { kind: 'gate'; id: string; text: string; items?: string[]; state: 'pending' | 'approved' | 'cancelled' }
  | { kind: 'agent-text'; id: string; text: string }
  | { kind: 'stats'; id: string; tokens: number; ms: number }
  | { kind: 'note'; id: string; text: string };

export type ModelStatus =
  | { phase: 'loading'; stage: LoadStage }
  | { phase: 'ready'; backend: 'wasm' | 'webgpu' | 'server'; loadMs: number }
  | { phase: 'error'; message: string };

let nextId = 0;
const mkId = () => `d${++nextId}`;

function effectFor(call: CallLogEntry): Effect {
  return TOOL_EFFECTS[call.name] ?? 'READ';
}

const ABORT_COPY: Record<string, string> = {
  NOT_FOUND: "I couldn't find what that refers to.",
  AMBIGUOUS: 'That could mean more than one thing — which did you mean?',
  UNSUPPORTED: "I don't have a tool that does that.",
  NEEDS_INFO: 'I need more details before I can do that.',
};

const EFFECT_COPY: Record<string, string> = {
  DELETE: "This can't be undone.",
  SEND: 'This will send a real message.',
  PAY: 'This will move money.',
};

function summarize(calls: CallLogEntry[], returnValue: unknown): string {
  const counts: Record<string, number> = {};
  for (const c of calls) {
    const e = effectFor(c);
    counts[e] = (counts[e] ?? 0) + 1;
  }
  const parts = Object.entries(counts).map(([eff, n]) => `${n} ${eff.toLowerCase()}`);
  const base = parts.length ? `Done — ${parts.join(', ')}.` : "Done — didn't need to touch any tools for that.";
  if (Array.isArray(returnValue)) {
    return `${base} ${returnValue.length} result${returnValue.length === 1 ? '' : 's'}.`;
  }
  if (returnValue !== null && returnValue !== undefined) {
    return `${base} Returned: ${JSON.stringify(returnValue)}.`;
  }
  return base;
}

export function useDbRun() {
  const [state, setState] = useState<CrmState | null>(null);
  const [lastResult, setLastResult] = useState<unknown>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ phase: 'loading', stage: 'grammar' });
  const [busy, setBusy] = useState(false);
  const [inference, setInference] = useState<InferenceMode>(() => loadPreference(APP_KEY));
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string | null>(() => loadModelPreference(APP_KEY));

  const plannerRef = useRef<Planner | null>(null);
  const gateResolveRef = useRef<((approved: boolean) => void) | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  const loadStartedRef = useRef(false);

  const append = useCallback((m: ChatMessage) => {
    setMessages((prev) => [...prev, m]);
  }, []);

  const patch = useCallback((id: string, fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));
  }, []);

  const loadPlanner = useCallback(async (mode: InferenceMode, name: string) => {
    try {
      if (plannerRef.current) {
        await plannerRef.current.unload();
        plannerRef.current = null;
      }
      setModelStatus({ phase: 'loading', stage: mode === 'server' ? 'model' : 'grammar' });
      const planner = await createPlannerFor({
        mode,
        model: name,
        onStage: (stage) => setModelStatus({ phase: 'loading', stage }),
      });
      plannerRef.current = planner;
      setModelStatus({ phase: 'ready', backend: planner.backend, loadMs: planner.loadMs });
    } catch (err) {
      setModelStatus({ phase: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  useEffect(() => {
    if (loadStartedRef.current) return;
    loadStartedRef.current = true;
    (async () => {
      try {
        setState(await fetchDbState());
      } catch (err) {
        setModelStatus({ phase: 'error', message: err instanceof Error ? err.message : String(err) });
        return;
      }
      let name = model;
      try {
        const list = await fetchModels();
        setModels(list.models);
        if (!name || !list.models.some((m) => m.name === name)) {
          name = list.default ?? list.models[0]?.name ?? null;
          setModel(name);
        }
      } catch {
        // no /models (a plain static host): fall back to whatever is stored
      }
      if (!name) {
        setModelStatus({ phase: 'error', message: 'no model available from GET /models' });
        return;
      }
      loadPlanner(inference, name);
    })();
  }, [loadPlanner]);

  // See kanban's useAgentRun.ts for the full story: this hook can unmount
  // for real now (client-side routing between worlds), and without this,
  // the next world's model load fights this one for the same OPFS handle.
  useEffect(() => {
    return () => {
      plannerRef.current?.unload().catch(() => {});
    };
  }, []);

  const setInferenceMode = useCallback((mode: InferenceMode) => {
    setInference((prev) => {
      if (prev === mode) return prev;
      savePreference(APP_KEY, mode);
      return mode;
    });
  }, []);

  const selectModel = useCallback((name: string) => {
    setModel((prev) => {
      if (prev === name) return prev;
      saveModelPreference(APP_KEY, name);
      return name;
    });
  }, []);

  const switchMountedRef = useRef(false);
  useEffect(() => {
    if (!switchMountedRef.current) {
      switchMountedRef.current = true;
      return;
    }
    if (model) loadPlanner(inference, model);
  }, [inference, model, loadPlanner]);

  const approveGate = useCallback(() => {
    gateResolveRef.current?.(true);
    gateResolveRef.current = null;
  }, []);

  const cancelGate = useCallback(() => {
    gateResolveRef.current?.(false);
    gateResolveRef.current = null;
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const planner = plannerRef.current;
      const trimmed = text.trim();
      if (!planner || !trimmed || busy || !stateRef.current) return;

      setBusy(true);
      append({ kind: 'user', id: mkId(), text: trimmed });

      let totalTokens = 0;
      let totalGenMs = 0;

      try {
        const kp = await fetchDbPrompt(trimmed, stateRef.current);

        let registers: Registers = null;
        let pauseTypes: Record<string, string> | null = null;
        const prior: string[] = [];
        let approvalGranted = false;

        for (let segIdx = 0; segIdx < MAX_SEGMENTS; segIdx++) {
          const prompt = buildFullPrompt(kp.input_text, registers, prior, kp.system);
          const gen = await planner.generate(prompt, { maxTokens: 250, stop: STOP, grammar: kp.grammar });
          prior.push(gen.text);
          totalTokens += gen.tokensOut;
          totalGenMs += gen.genMs;
          append({ kind: 'program', id: mkId(), text: gen.text });

          let resp: ValidateResponse<CrmState> = await validate<CrmState>({
            context: kp.context,
            world: kp.world,
            now: kp.now,
            text: gen.text,
            state: stateRef.current,
            registers,
            pause_types: pauseTypes,
            approval: approvalGranted,
          });

          while (resp.status === 'effect_blocked') {
            const sym = resp.error?.tool;
            const tool = sym ? kp.context.tools.find((t) => t.sym === sym) : undefined;
            const cur = stateRef.current;
            const args = resp.error?.args
              ? resp.error.args.map((a) => describeArg(a, cur, cur))
              : sym
                ? describeCallSite(gen.text, sym, kp.context)
                : [];
            const changes = resp.error?.calls ?? [];
            const lead = (resp.error?.effect && EFFECT_COPY[resp.error.effect]) || 'This needs approval.';
            const gateId = mkId();
            append({
              kind: 'gate',
              id: gateId,
              text: changes.length
                ? `${changes.length} change${changes.length === 1 ? '' : 's'}. ${lead}`
                : `${lead} ${tool?.name ?? 'this action'}(${args.join(', ')})`,
              items: changes.length
                ? changes.map((c) => `${c.name} — ${c.args.map((a) => describeArg(a, cur, cur)).join(' · ')}`)
                : undefined,
              state: 'pending',
            });
            const approved = await new Promise<boolean>((resolve) => {
              gateResolveRef.current = resolve;
            });
            patch(gateId, (m) => (m.kind === 'gate' ? { ...m, state: approved ? 'approved' : 'cancelled' } : m));
            if (!approved) {
              append({ kind: 'note', id: mkId(), text: 'Cancelled — nothing changed.' });
              return;
            }
            approvalGranted = true;
            resp = await validate<CrmState>({
              context: kp.context,
              world: kp.world,
              now: kp.now,
              text: gen.text,
              state: stateRef.current,
              registers,
              pause_types: pauseTypes,
              approval: true,
            });
          }

          if (resp.status === 'paused') {
            registers = resp.registers ?? null;
            pauseTypes = resp.pause_envs?.[0] ?? null;
            continue;
          }

          if (resp.status === 'aborted') {
            const text = ABORT_COPY[resp.reason ?? ''] ?? `Can't do that (${resp.reason || 'no reason given'}).`;
            append({ kind: 'agent-text', id: mkId(), text });
            return;
          }

          if (resp.status === 'ok') {
            const before = stateRef.current;
            const after = resp.final_state ?? before;
            for (const call of resp.calls) {
              append({
                kind: 'tool-call',
                id: mkId(),
                effect: effectFor(call),
                name: call.name,
                detail: describeCall(call, before, after),
              });
            }
            setState(after);
            setLastResult(resp.return_value);
            append({ kind: 'agent-text', id: mkId(), text: summarize(resp.calls, resp.return_value) });
            return;
          }

          append({
            kind: 'note',
            id: mkId(),
            text: resp.diagnostics?.length ? resp.diagnostics.join(' · ') : resp.error?.message ?? `run failed: ${resp.status}`,
          });
          return;
        }

        append({ kind: 'note', id: mkId(), text: 'Segment cap reached — stopping.' });
      } catch (err) {
        append({ kind: 'note', id: mkId(), text: err instanceof Error ? err.message : String(err) });
      } finally {
        if (totalTokens > 0) {
          append({ kind: 'stats', id: mkId(), tokens: totalTokens, ms: totalGenMs });
        }
        setBusy(false);
      }
    },
    [append, patch, busy]
  );

  return {
    state,
    lastResult,
    messages,
    modelStatus,
    busy,
    inference,
    setInferenceMode,
    models,
    model,
    selectModel,
    sendMessage,
    approveGate,
    cancelGate,
  };
}
