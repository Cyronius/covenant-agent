import { useEffect, useRef } from 'react';
import { TOOLS } from '../data/tools';

const EFFECT_NOTE: Record<string, string> = {
  READ: 'Runs freely.',
  WRITE: 'Runs freely.',
  DELETE: "Needs your approval — can't be undone.",
  SEND: 'Needs your approval — sends a real message.',
  EXTERNAL: 'Runs freely — calls the writer model; you approve before anything it wrote is sent.',
};

export default function ToolsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    dialogRef.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal tools-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Tools available to the agent"
        tabIndex={-1}
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h2>What the agent can do</h2>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <p className="modal-sub">
          Every request is compiled down to calls against this fixed tool set — nothing else is reachable.
          DELETE and SEND calls pause for your approval before they run.
        </p>
        <ul className="tools-list">
          {TOOLS.map((t) => (
            <li key={t.name} className={`tool-entry effect-${t.effect.toLowerCase()}`}>
              <div className="tool-entry-head">
                <span className="effect-tag">{t.effect}</span>
                <span className="tool-entry-name">{t.name}</span>
                <span className="tool-entry-sig">
                  ({t.params.map((p) => p.name).join(', ')})
                  {t.returns ? ` → ${t.returns}` : ''}
                </span>
              </div>
              <p className="tool-entry-desc">{t.desc}</p>
              {t.params.length > 0 && (
                <ul className="tool-entry-params">
                  {t.params.map((p) => (
                    <li key={p.name}>
                      <code>{p.name}</code> <span className="param-type">{p.type}</span> — {p.desc}
                    </li>
                  ))}
                </ul>
              )}
              <p className="tool-entry-note">{EFFECT_NOTE[t.effect]}</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
