// Per turn: what the model was shown, what it wrote, and what the rules did
// with it. Both the observation and the program are the real text, collapsed
// rather than paraphrased — the point of the demo is that you can read what
// the model actually saw when it walked into a wall.
import { useState } from 'react';
import type { TurnRecord } from '../hooks/useDungeonRun';

function Disclosure({ label, text }: { label: string; text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="disclosure">
      <button type="button" className="disclosure-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? '▾' : '▸'} {label}
      </button>
      {open && <pre className="disclosure-body">{text}</pre>}
    </div>
  );
}

const STATUS_COPY: Record<string, string> = {
  ok: 'ran',
  error: 'stopped on a bad action',
  static_error: "didn't compile",
  aborted: 'declined',
  paused: 'paused (no continuation in a game turn)',
  effect_blocked: 'blocked by the effect gate',
};

export function TurnLog({ turns }: { turns: TurnRecord[] }) {
  if (!turns.length) {
    return (
      <p className="empty">
        Press <strong>Step</strong> for one turn, or <strong>Auto-play</strong> to let it run.
        Every turn is a fresh prompt: the model sees only the 5×5 window outlined on the map.
      </p>
    );
  }
  return (
    <ol className="turns">
      {turns.map((t) => (
        <li key={t.id} className={`turn turn-${t.status}`}>
          <header>
            <span className="turn-no">turn {t.turn}</span>
            <span className={`turn-status s-${t.status}`}>{STATUS_COPY[t.status] ?? t.status}</span>
            {t.tokens > 0 && (
              <span className="turn-stats">
                {t.tokens} tok · {(t.ms / 1000).toFixed(1)}s ·{' '}
                {(t.tokens / Math.max(t.ms / 1000, 0.001)).toFixed(1)} tok/s
              </span>
            )}
          </header>

          {t.calls.length > 0 && (
            <ul className="calls">
              {t.calls.map((c, i) => (
                <li key={i} className={c.ok ? 'call ok' : 'call bad'}>
                  <code>
                    {c.name}({c.args.map((a) => String(a)).join(', ')})
                  </code>
                  {!c.ok && <span className="call-err">{c.error?.code ?? 'failed'}</span>}
                </li>
              ))}
            </ul>
          )}

          {t.events.length > 0 && <p className="events">{t.events.join(' · ')}</p>}
          {t.diagnostics.length > 0 && <p className="diag">{t.diagnostics.join(' · ')}</p>}
          {t.error && t.calls.every((c) => c.ok) && (
            <p className="diag">
              {t.error.code}: {t.error.message}
            </p>
          )}

          <Disclosure label="program" text={t.program} />
          <Disclosure label="what the model was shown" text={t.request} />
        </li>
      ))}
    </ol>
  );
}
