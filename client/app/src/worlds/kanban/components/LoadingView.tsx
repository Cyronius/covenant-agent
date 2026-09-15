import { Fragment } from 'react';
import type { ModelStatus } from '../hooks/useAgentRun';

const STEPS: { key: 'grammar' | 'model' | 'ready'; label: string }[] = [
  { key: 'grammar', label: 'Grammar' },
  { key: 'model', label: 'Model weights' },
  { key: 'ready', label: 'Ready' },
];

function stepState(status: ModelStatus, key: string): 'done' | 'active' | 'pending' {
  if (status.phase === 'ready') return key === 'ready' ? 'active' : 'done';
  if (status.phase !== 'loading') return 'pending';
  const order = ['grammar', 'model', 'ready'];
  const cur = order.indexOf(status.stage);
  const idx = order.indexOf(key);
  if (idx < cur) return 'done';
  if (idx === cur) return 'active';
  return 'pending';
}

function stageLine(status: ModelStatus): string {
  if (status.phase === 'error') return status.message;
  if (status.phase === 'ready') return `Ready · ${status.backend} · ${Math.round(status.loadMs)}ms load`;
  if (status.stage === 'grammar') return 'Fetching the grammar…';
  return 'Loading model weights — first run pulls ~800MB from the local dev server, give it a minute.';
}

function SproutIcon() {
  return (
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 21V11" />
      <path d="M12 12c0-4 -3-6.5-7.5-7C4.8 9.5 7 13 12 13Z" fill="currentColor" opacity=".22" stroke="none" />
      <path d="M12 12c0-4 -3-6.5-7.5-7C4.8 9.5 7 13 12 13Z" />
      <path d="M12 9c0-3 2.5-5 6.5-5.4C18.9 7.2 17 10 12 10Z" fill="currentColor" opacity=".3" stroke="none" />
      <path d="M12 9c0-3 2.5-5 6.5-5.4C18.9 7.2 17 10 12 10Z" />
    </svg>
  );
}

export default function LoadingView({ status }: { status: ModelStatus }) {
  return (
    <div className="loading-view">
      <div className={`loading-mark${status.phase === 'loading' ? ' pulse' : ''}`}>
        <SproutIcon />
      </div>
      <p className="loading-stage">{stageLine(status)}</p>
      <div className="loading-steps">
        {STEPS.map((s, i) => (
          <Fragment key={s.key}>
            <div className="loading-step">
              <span className={`loading-dot ${stepState(status, s.key)}`} />
              <span className="loading-step-label">{s.label}</span>
            </div>
            {i < STEPS.length - 1 && <span className="loading-rule" />}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
