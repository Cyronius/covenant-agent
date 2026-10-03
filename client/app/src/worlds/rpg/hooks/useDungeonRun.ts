// One turn of the dungeon, driven by the real model.
//
//   POST /rpg_prompt  -> the turn's request text + constants (perception)
//   generate          -> an Agent Core program, grammar-constrained
//   POST /validate    -> real parse/typecheck/compile, real sandboxed rules,
//                        then the world's enemy phase (the post_hook)
//
// Ported from client/kanban-ui's useAgentRun, minus two things it does not
// need: there is no approval gate (no DELETE/SEND/PAY tool in this world),
// and there is no PAUSE continuation — every turn is a fresh single-segment
// program against freshly assigned symbols, so a `paused` result just ends
// the turn and its registers are dropped. See .claude/plans/rpg-demo-app.md.
import { useCallback, useEffect, useRef, useState } from 'react';
import { createPlanner, type Planner } from '../../../../../shared/planner';
import { buildFullPrompt, STOP } from '../../../../../shared/prompt';
import { validate, type CallLogEntry, type ValidateResponse } from '../../../../../shared/validate';
import {
  fetchRpgPrompt,
  newGame,
  player,
  type RpgPromptResponse,
  type RpgState,
} from '../lib/rpgApi';

export type ModelStatus =
  | { phase: 'loading' }
  | { phase: 'ready'; model: string; loadMs: number }
  | { phase: 'error'; message: string };

export interface TurnRecord {
  id: number;
  turn: number;
  /** what the planner read: the one-line brief, then each constant's
   *  description (the tiny planner never sees the grid) */
  request: string;
  /** the brief went past the planner's request budget */
  truncated: boolean;
  /** what it wrote back, verbatim Agent Core */
  program: string;
  status: string;
  calls: CallLogEntry[];
  diagnostics: string[];
  error: { code: string; message?: string } | null;
  reason?: string | null;
  events: string[];
  tokens: number;
  ms: number;
}

const ABORT_COPY: Record<string, string> = {
  NOT_FOUND: "It says it can't find what the goal refers to.",
  AMBIGUOUS: 'It says the situation is ambiguous.',
  UNSUPPORTED: 'It says it has no tool for that.',
  NEEDS_INFO: 'It says it needs more information.',
};

/** What the planner reads this turn, for the side panel. */
export interface Seen {
  brief: string;
  constants: string[];
}

function seenFrom(kp: RpgPromptResponse): Seen {
  return { brief: kp.observation.brief, constants: kp.observation.constants };
}

export function useDungeonRun() {
  const [state, setState] = useState<RpgState | null>(null);
  const [seen, setSeen] = useState<Seen | null>(null);
  const [turns, setTurns] = useState<TurnRecord[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ phase: 'loading' });
  const [busy, setBusy] = useState(false);
  const [auto, setAuto] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const plannerRef = useRef<Planner | null>(null);
  const stateRef = useRef<RpgState | null>(null);
  stateRef.current = state;
  const autoRef = useRef(auto);
  autoRef.current = auto;
  const busyRef = useRef(false);
  const nextTurnId = useRef(0);
  // StrictMode double-invokes effects in dev; one load per real mount.
  const loadStartedRef = useRef(false);

  const restart = useCallback(async (scenario?: string) => {
    setAuto(false);
    setTurns([]);
    setNote(null);
    const fresh = await newGame(scenario);
    const kp = await fetchRpgPrompt(fresh);
    setState(kp.state);
    setSeen(seenFrom(kp));
  }, []);

  useEffect(() => {
    if (loadStartedRef.current) return;
    loadStartedRef.current = true;
    restart().catch((err) => setNote(err instanceof Error ? err.message : String(err)));
    createPlanner()
      .then((planner) => {
        plannerRef.current = planner;
        setModelStatus({ phase: 'ready', model: planner.model, loadMs: planner.loadMs });
      })
      .catch((err) => setModelStatus({ phase: 'error', message: err instanceof Error ? err.message : String(err) }));
  }, [restart]);

  /** Play exactly one turn. Returns false when the game is over or the turn
   *  could not run, which is also what stops auto-play. */
  const step = useCallback(async (): Promise<boolean> => {
    const planner = plannerRef.current;
    const current = stateRef.current;
    if (!planner || !current || busyRef.current) return false;
    if (current.status !== 'playing') return false;
    if (current.turn >= current.max_turns) {
      setNote(`Out of turns (${current.max_turns}).`);
      return false;
    }

    busyRef.current = true;
    setBusy(true);
    try {
      const kp = await fetchRpgPrompt(current);
      setSeen(seenFrom(kp));

      const prompt = buildFullPrompt(kp.input_text, null, [], kp.system);
      const gen = await planner.generate(prompt, {
        maxTokens: 250,
        stop: STOP,
        grammar: kp.grammar,
        // the tiny planner reads the brief and the constants, not the
        // grid-and-legend observation a GGUF was prompted with
        task: { request: kp.observation.brief, context: kp.context },
      });

      const run = (text: string) =>
        validate<RpgState>({
          context: kp.context,
          world: 'rpg',
          now: kp.now,
          text,
          state: kp.state,
          registers: null,
          pause_types: null,
          approval: true, // nothing in this world is effect-gated
        });

      const resp: ValidateResponse<RpgState> = await run(gen.text);

      const before = kp.state;
      let after = resp.final_state ?? before;

      // A program that doesn't compile never reaches the sandbox, so the
      // enemy phase never fires and the clock never moves — auto-play would
      // sit on the same turn forever against a model that mostly fails to
      // compile (seen in the browser 2026-09-04: four log entries, all "turn
      // 1"). Burn the turn with a no-op instead, which is exactly what
      // harness/rpg_suite.py's _idle_turn does offline. An ABORT needs no
      // such thing: it compiles and runs, so its post_hook already fired.
      if (resp.status === 'static_error') {
        try {
          const idle = await run('STOP\n');
          after = idle.final_state ?? after;
        } catch {
          // leave the state alone; the turn is simply lost
        }
      }
      const record: TurnRecord = {
        id: nextTurnId.current++,
        turn: before.turn,
        request: [kp.observation.brief, ...kp.observation.constants].join('\n'),
        truncated: gen.truncated ?? false,
        program: gen.text,
        status: resp.status,
        calls: resp.calls ?? [],
        diagnostics: resp.diagnostics ?? [],
        error: resp.error ?? null,
        reason: resp.reason ?? null,
        events: after.log ?? [],
        tokens: gen.tokensOut,
        ms: gen.genMs,
      };
      setTurns((prev) => [...prev, record]);

      if (record.truncated) {
        setNote(`The planner only read the start of this turn's brief (${gen.requestTokens} tokens is past its budget).`);
      } else if (resp.status === 'static_error') {
        setNote("The program didn't compile — the turn passed anyway.");
      } else if (resp.status === 'aborted') {
        setNote(ABORT_COPY[record.reason ?? ''] ?? 'It declined to act.');
      } else {
        setNote(null);
      }

      setState(after);
      stateRef.current = after;
      // Re-render the observation for the new state so the "what the model
      // sees" panel matches the map instead of trailing it by a turn. This
      // is pure server-side rendering (no generation), and it advances
      // `memory` for the next prompt.
      try {
        const next = await fetchRpgPrompt(after);
        setSeen(seenFrom(next));
        setState(next.state);
        stateRef.current = next.state;
      } catch {
        // keep the state we already applied; the next step re-renders anyway
      }
      if (after.status === 'won') setNote('Reached the stairs down.');
      if (after.status === 'dead') setNote(`${player(after).hp} HP — killed on turn ${after.turn}.`);
      return after.status === 'playing';
    } catch (err) {
      setNote(err instanceof Error ? err.message : String(err));
      return false;
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, []);

  // Auto-play: one turn at a time, stopping the moment the game ends, the
  // toggle flips off, or a turn fails. Deliberately sequential — each turn's
  // prompt depends on the previous turn's state.
  useEffect(() => {
    if (!auto) return;
    let cancelled = false;
    (async () => {
      while (!cancelled && autoRef.current) {
        const ok = await step();
        if (!ok) break;
      }
      if (!cancelled) setAuto(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [auto, step]);

  return {
    state,
    seen,
    turns,
    modelStatus,
    busy,
    auto,
    note,
    step,
    startAuto: () => setAuto(true),
    stopAuto: () => setAuto(false),
    restart,
  };
}
