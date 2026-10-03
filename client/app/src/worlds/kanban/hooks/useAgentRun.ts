// Orchestrates the real thing: model load, a freely-typed request -> real
// prompt (POST /kanban_prompt) -> real grammar-constrained generation ->
// real /validate round-trip (PAUSE continuation loop + the real approval
// gate), and the chat/board state the UI reads. No preloaded conversation,
// no fixed set of requests — whatever gets typed goes to the model.
//
// The board is fake (src/data/board.ts) and persists across turns; the
// generation/compile/typecheck/effects/execution/approval-gate are real.
// See .claude/plans/understory-kanban-frontend.md.
import { useCallback, useEffect, useRef, useState } from 'react';
import { createPlanner, type Planner } from '../../../../../shared/planner';
import { buildFullPrompt, STOP, type Registers } from '../../../../../shared/prompt';
import { validate, type CallLogEntry, type TaskContextJson, type ValidateResponse } from '../../../../../shared/validate';
import { fetchKanbanPrompt, describeCallSite } from '../lib/kanbanPrompt';
import { describeCall, describeArg } from '../lib/describe';
import { initialState, userById, TOOL_EFFECTS, type KanbanState, type Effect } from '../data/board';

const MAX_SEGMENTS = 4; // PAUSE continuations per request (harness MAX_SEGMENTS is the eval-side cap)

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
const mkId = () => `m${++nextId}`;

function effectFor(call: CallLogEntry): Effect {
  return TOOL_EFFECTS[call.name] ?? 'READ';
}

// spec §4 ABORT reasons, as the person should read them.
const ABORT_COPY: Record<string, string> = {
  NOT_FOUND: "I couldn't find what that refers to on this board.",
  AMBIGUOUS: 'That could mean more than one thing — which did you mean?',
  UNSUPPORTED: "I don't have a tool that does that.",
  NEEDS_INFO: 'I need more details before I can do that.',
};

/** Turn `ABORT reason refs` into a sentence using the context's own
 *  descriptions. Returns undefined for a bare abort so the enum copy applies. */
function describeAbort(reason: string, refs: string[], ctx: TaskContextJson): string | undefined {
  if (!refs.length) return undefined;
  const desc = (sym: string): string => {
    const k = sym[0];
    const d = k === 'T' ? ctx.tools.find((t) => t.sym === sym)?.desc
      : k === 'F' ? ctx.fields.find((f) => f.sym === sym)?.desc
      : ctx.constants.find((c) => c.sym === sym);
    if (k === 'C' && d && typeof d === 'object') return `"${String((d as { value: unknown }).value)}"`;
    return typeof d === 'string' ? d : sym;
  };
  switch (reason) {
    case 'NOT_FOUND':
      return `I couldn't find anything matching ${desc(refs[0])}.`;
    case 'NEEDS_INFO':
      return `I need one more thing: ${desc(refs[0])}.`;
    case 'AMBIGUOUS':
      return refs.length > 1
        ? `That could mean ${desc(refs[0])} or ${desc(refs[1])} — which did you mean?`
        : `There's more than one match — which one? (tell me the ${desc(refs[0])})`;
    default:
      return undefined;
  }
}

const EFFECT_COPY: Record<string, string> = {
  DELETE: "This can't be undone.",
  SEND: 'This will send a real message.',
  PAY: 'This will move money.',
};

/** Headline for the approval gate's list. A turn that only sends messages
 *  hasn't changed the board, so don't say it has. */
function countChanges(changes: { name: string }[]): string {
  const n = changes.length;
  const s = n === 1 ? '' : 's';
  return changes.every((c) => TOOL_EFFECTS[c.name] === 'SEND')
    ? `${n} message${s}`
    : `${n} change${s} to the board`;
}

function summarize(calls: CallLogEntry[], returnValue: unknown): string {
  const counts: Record<string, number> = {};
  for (const c of calls) {
    const e = effectFor(c);
    counts[e] = (counts[e] ?? 0) + 1;
  }
  const parts = Object.entries(counts).map(([eff, n]) => `${n} ${eff.toLowerCase()}`);
  const base = parts.length ? `Done — ${parts.join(', ')}.` : "Done — didn't need to touch any tools for that.";
  if (returnValue !== null && returnValue !== undefined) {
    return `${base} Returned: ${JSON.stringify(returnValue)}.`;
  }
  return base;
}

export function useAgentRun() {
  const [board, setBoard] = useState<KanbanState>(() => initialState());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ phase: 'loading' });
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const plannerRef = useRef<Planner | null>(null);
  const gateResolveRef = useRef<((approved: boolean) => void) | null>(null);
  const toastTimerRef = useRef<number | undefined>(undefined);
  const boardRef = useRef(board);
  boardRef.current = board;
  // StrictMode double-invokes effects in dev (mount -> cleanup -> mount);
  // one load per real mount is enough.
  const loadStartedRef = useRef(false);

  const append = useCallback((m: ChatMessage) => {
    setMessages((prev) => [...prev, m]);
  }, []);

  const patch = useCallback((id: string, fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));
  }, []);

  const pushToast = useCallback((text: string) => {
    window.clearTimeout(toastTimerRef.current);
    setToast(text);
    toastTimerRef.current = window.setTimeout(() => setToast(null), 2400);
  }, []);

  // Loads on mount, no button in the way — see App.tsx's loading view.
  //
  // No cancellation flag: loadStartedRef makes this a true singleton load
  // for the component's real lifetime, and gating these setState calls on
  // "did the effect that started this get cleaned up" is wrong for
  // StrictMode's simulated cleanup, which fires right after the first mount
  // while the load is still running — it once left the UI on the loading
  // screen forever after the model had finished loading.
  useEffect(() => {
    if (loadStartedRef.current) return;
    loadStartedRef.current = true;
    createPlanner()
      .then((planner) => {
        plannerRef.current = planner;
        setModelStatus({ phase: 'ready', model: planner.model, loadMs: planner.loadMs });
      })
      .catch((err) => setModelStatus({ phase: 'error', message: err instanceof Error ? err.message : String(err) }));
  }, []);

  const approveGate = useCallback(() => {
    gateResolveRef.current?.(true);
    gateResolveRef.current = null;
  }, []);

  const cancelGate = useCallback(() => {
    gateResolveRef.current?.(false);
    gateResolveRef.current = null;
  }, []);

  const resetBoard = useCallback(() => {
    setBoard(initialState());
    setMessages([]);
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const planner = plannerRef.current;
      const trimmed = text.trim();
      if (!planner || !trimmed || busy) return;

      setBusy(true);
      append({ kind: 'user', id: mkId(), text: trimmed });

      let totalTokens = 0;
      let totalGenMs = 0;

      try {
        const kp = await fetchKanbanPrompt(trimmed, boardRef.current);

        // The request names somebody who isn't here. The server resolved
        // that against the board before anything was generated, so answer
        // and stop — a model asked to assign to a person who doesn't exist
        // picks one who does (2026-09-13: four cards to the wrong Bob).
        if (kp.preflight) {
          append({ kind: 'agent-text', id: mkId(), text: kp.preflight.message });
          return;
        }

        let registers: Registers = null;
        let pauseTypes: Record<string, string> | null = null;
        const prior: string[] = [];
        // Always try unapproved first — the real effect gate (DELETE/SEND/PAY
        // all count, spec §7) decides whether that was fine or needs a real
        // click, not the UI guessing from the request text.
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

          let resp: ValidateResponse<KanbanState> = await validate<KanbanState>({
            context: kp.context,
            world: kp.world,
            now: kp.now,
            text: gen.text,
            state: boardRef.current,
            registers,
            pause_types: pauseTypes,
            approval: approvalGranted,
          });

          // Same generated program re-sent with the approval flag flipped —
          // nothing executed yet, so there's nothing to "continue" from.
          while (resp.status === 'effect_blocked') {
            const sym = resp.error?.tool;
            const tool = sym ? kp.context.tools.find((t) => t.sym === sym) : undefined;
            // Prefer the sandbox's actual params (resolved values — for a SEND
            // that's the drafted text); fall back to the static call site.
            const board = boardRef.current;
            const args = resp.error?.args
              ? resp.error.args.map((a) => describeArg(a, board, board))
              : sym
                ? describeCallSite(gen.text, sym, kp.context)
                : [];
            const changes = resp.error?.calls ?? [];
            const lead = (resp.error?.effect && EFFECT_COPY[resp.error.effect]) || 'This needs approval.';
            const gateId = mkId();
            append({
              kind: 'gate',
              id: gateId,
              // The whole program ran unapproved against a throwaway copy of
              // the board, so this list is everything it would do — not the
              // first call of however many (which is what one click used to
              // authorize: "delete_card #2" deleted seven cards).
              text: changes.length
                ? `${countChanges(changes)}. ${lead}`
                : `${lead} ${tool?.name ?? 'this action'}(${args.join(', ')})`,
              items: changes.length
                ? changes.map(
                    (c) => `${c.name} — ${c.args.map((a) => describeArg(a, board, board)).join(' · ')}`
                  )
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
            resp = await validate<KanbanState>({
              context: kp.context,
              world: kp.world,
              now: kp.now,
              text: gen.text,
              state: boardRef.current,
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
            // The planner declined (ABORT) — nothing ran past any reads.
            // With a referent (spec §4 0.3.0) the abort is a question we can
            // put to the user in the schema's own words.
            const reason = resp.reason ?? '';
            const text = describeAbort(reason, resp.refs ?? [], kp.context)
              ?? ABORT_COPY[reason]
              ?? `Can't do that (${reason || 'no reason given'}).`;
            append({ kind: 'agent-text', id: mkId(), text });
            return;
          }

          if (resp.status === 'ok') {
            const before = boardRef.current;
            const after = resp.final_state ?? before;
            for (const call of resp.calls) {
              append({
                kind: 'tool-call',
                id: mkId(),
                effect: effectFor(call),
                name: call.name,
                detail: describeCall(call, before, after),
              });
              if (call.name === 'send_message' && call.ok) {
                const recipient = typeof call.args[0] === 'string' ? userById(after, call.args[0] as string) : null;
                pushToast(`Message sent to ${recipient?.name ?? 'user'}`);
              }
            }
            setBoard(after);
            append({ kind: 'agent-text', id: mkId(), text: summarize(resp.calls, resp.return_value) });
            return;
          }

          // static_error / error / server_error — surface diagnostics and stop.
          append({
            kind: 'note',
            id: mkId(),
            text: resp.diagnostics?.length
              ? resp.diagnostics.join(' · ')
              : resp.error?.message ?? `run failed: ${resp.status}`,
          });
          return;
        }

        append({ kind: 'note', id: mkId(), text: 'Segment cap reached — stopping.' });
      } catch (err) {
        append({ kind: 'note', id: mkId(), text: err instanceof Error ? err.message : String(err) });
      } finally {
        // Only if at least one generate() call actually produced tokens —
        // e.g. not when fetchKanbanPrompt() itself threw before generation.
        if (totalTokens > 0) {
          append({ kind: 'stats', id: mkId(), tokens: totalTokens, ms: totalGenMs });
        }
        setBusy(false);
      }
    },
    [append, patch, pushToast, busy]
  );

  return {
    board,
    messages,
    modelStatus,
    busy,
    toast,
    sendMessage,
    approveGate,
    cancelGate,
    resetBoard,
  };
}
