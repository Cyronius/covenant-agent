import { Fragment } from 'react';
import type { ModelStatus } from '../hooks/useDbRun';

const STEPS: { key: 'model' | 'ready'; label: string }[] = [
  { key: 'model', label: 'Planner' },
  { key: 'ready', label: 'Ready' },
];

function stepState(status: ModelStatus, key: string): 'done' | 'active' | 'pending' {
  if (status.phase === 'ready') return key === 'ready' ? 'active' : 'done';
  if (status.phase !== 'loading') return 'pending';
  return key === 'model' ? 'active' : 'pending';
}

function stageLine(status: ModelStatus): string {
  if (status.phase === 'error') return status.message;
  if (status.phase === 'ready') return `Ready · ${status.model} · ${Math.round(status.loadMs)}ms load`;
  return 'Loading the planner on the dev server…';
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
