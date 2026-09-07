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
import { type LoadStage, type Planner } from '../../../shared/llm';
import {
  createPlannerFor,
  fetchModels,
  loadModelPreference,
  loadPreference,
  saveModelPreference,
  savePreference,
  type InferenceMode,
  type ModelInfo,
} from '../../../shared/inference';
import { buildFullPrompt, STOP, type Registers } from '../../../shared/prompt';
import { validate, type CallLogEntry, type TaskContextJson, type ValidateResponse } from '../../../shared/validate';
import { fetchKanbanPrompt, describeCallSite } from '../lib/kanbanPrompt';
import { describeCall, describeArg } from '../lib/describe';
import { initialState, userById, TOOL_EFFECTS, type KanbanState, type Effect } from '../data/board';

// Which checkpoint to load is the server's call (GET /models), not a
// hardcode here — see client/shared/inference.ts. This was pinned to the S1
// checkpoint until 2026-09-04, which quietly kept the demo a generation
// behind the eval suites.
const APP_KEY = 'kanban-ui';
const MAX_SEGMENTS = 4; // PAUSE continuations per request (harness MAX_SEGMENTS is the eval-side cap)

export type ChatMessage =
  | { kind: 'user'; id: string; text: string }
  | { kind: 'program'; id: string; text: string }
  | { kind: 'tool-call'; id: string; effect: Effect; name: string; detail: string }
  | { kind: 'gate'; id: string; text: string; state: 'pending' | 'approved' | 'cancelled' }
  | { kind: 'agent-text'; id: string; text: string }
  | { kind: 'stats'; id: string; tokens: number; ms: number }
  | { kind: 'note'; id: string; text: string };

export type ModelStatus =
  | { phase: 'loading'; stage: LoadStage }
  | { phase: 'ready'; backend: 'wasm' | 'webgpu' | 'server'; loadMs: number }
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
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ phase: 'loading', stage: 'grammar' });
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [inference, setInference] = useState<InferenceMode>(() => loadPreference(APP_KEY));
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string | null>(() => loadModelPreference(APP_KEY));

  const plannerRef = useRef<Planner | null>(null);
  const gateResolveRef = useRef<((approved: boolean) => void) | null>(null);
  const toastTimerRef = useRef<number | undefined>(undefined);
  const boardRef = useRef(board);
  boardRef.current = board;
  // StrictMode double-invokes effects in dev (mount -> cleanup -> mount) on
  // the same component instance; wllama's OPFS-backed model cache can't
  // have two concurrent loads open a sync access handle on the same file
  // ("Access Handles cannot be created if there is another open Access
  // Handle..."), so createPlanner() must only ever actually start once,
  // regardless of how many times the effect body runs.
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

  // Shared by the initial mount load and every mode/model switch below: both
  // want "unload whatever's loaded (if anything), load fresh under this
  // backend and checkpoint, report loading/ready/error the same way". Both
  // the backend and n_gpu_layers are load-time settings, so a switch always
  // means loading again.
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

  // Loads on mount, no button in the way — see App.tsx's loading view.
  //
  // No cancellation flag: loadStartedRef makes this a true singleton load
  // for the component's real lifetime (StrictMode's simulated dev-mode
  // mount->cleanup->mount can only ever start it once), so there's no
  // second attempt whose result would need to be discarded — and gating
  // these setState calls on "did the effect that started this get cleaned
  // up" is wrong for exactly that StrictMode cleanup: it fires immediately
  // after the first mount despite the singleton load still running, which
  // silently swallowed every later status update including 'ready' (caught
  // via an actual browser run: the model finished loading — confirmed in
  // wllama's own console output — but the UI sat on the loading screen
  // forever because `cancelled` had already flipped true).
  //
  // Which checkpoint to load comes from GET /models, so the first load waits
  // for that list; a stored preference that the server no longer has falls
  // back to the server's own default.
  useEffect(() => {
    if (loadStartedRef.current) return;
    loadStartedRef.current = true;
    (async () => {
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
    // Only the first mount's values matter here; the effect below owns every
    // reload after that, including ones this same state change triggers.
  }, [loadPlanner]);

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

  // Reload whenever the backend or the checkpoint changes after the initial
  // mount. Guarded so this doesn't also fire for the mount effect's own
  // initial values (that load is already in flight above).
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

        let registers: Registers = null;
        let pauseTypes: Record<string, string> | null = null;
        const prior: string[] = [];
        // Always try unapproved first — the real effect gate (DELETE/SEND/PAY
        // all count, spec §7) decides whether that was fine or needs a real
        // click, not the UI guessing from the request text.
        let approvalGranted = false;

        for (let segIdx = 0; segIdx < MAX_SEGMENTS; segIdx++) {
          const prompt = buildFullPrompt(kp.input_text, registers, prior);
          const gen = await planner.generate(prompt, { maxTokens: 250, stop: STOP });
          prior.push(gen.text);
          totalTokens += gen.tokensOut;
          totalGenMs += gen.genMs;
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
            const lead = (resp.error?.effect && EFFECT_COPY[resp.error.effect]) || 'This needs approval.';
            const gateId = mkId();
            append({
              kind: 'gate',
              id: gateId,
              text: `${lead} ${tool?.name ?? 'this action'}(${args.join(', ')})`,
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
    inference,
    setInferenceMode,
    models,
    model,
    selectModel,
    sendMessage,
    approveGate,
    cancelGate,
    resetBoard,
  };
}
