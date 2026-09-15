import { Fragment } from 'react';
import type { ModelStatus } from '../hooks/useDbRun';

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

export default function LoadingView({ status }: { status: ModelStatus }) {
  return (
    <div className="loading-view">
      <div className={`loading-mark${status.phase === 'loading' ? ' pulse' : ''}`} />
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
