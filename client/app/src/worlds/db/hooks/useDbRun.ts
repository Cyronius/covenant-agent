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
import { createPlanner, type Planner } from '../../../../../shared/planner';
import { buildFullPrompt, STOP, type Registers } from '../../../../../shared/prompt';
import { validate, type CallLogEntry, type ValidateResponse } from '../../../../../shared/validate';
import { fetchDbPrompt, describeCallSite } from '../lib/dbPrompt';
import { describeCall, describeArg } from '../lib/describe';
import { fetchDbState, TOOL_EFFECTS, type CrmState, type Effect } from '../data/crm';

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
  | { phase: 'loading' }
  | { phase: 'ready'; model: string; loadMs: number }
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
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ phase: 'loading' });
  const [busy, setBusy] = useState(false);

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

  // Singleton load per real mount; see kanban's useAgentRun.ts for why
  // there is no cancellation flag.
  useEffect(() => {
    if (loadStartedRef.current) return;
    loadStartedRef.current = true;
    (async () => {
      try {
        setState(await fetchDbState());
        const planner = await createPlanner();
        plannerRef.current = planner;
        setModelStatus({ phase: 'ready', model: planner.model, loadMs: planner.loadMs });
      } catch (err) {
        setModelStatus({ phase: 'error', message: err instanceof Error ? err.message : String(err) });
      }
    })();
  }, []);

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
          const gen = await planner.generate(prompt, {
            maxTokens: 250,
            stop: STOP,
            grammar: kp.grammar,
            task: { request: trimmed, context: kp.context, registers, pauseTypes },
          });
          prior.push(gen.text);
          totalTokens += gen.tokensOut;
          totalGenMs += gen.genMs;
          if (gen.truncated && segIdx === 0) {
            append({
              kind: 'note',
              id: mkId(),
              text: `The planner only read the start of this request (${gen.requestTokens} tokens is past its budget).`,
            });
          }
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
    sendMessage,
    approveGate,
    cancelGate,
  };
}
