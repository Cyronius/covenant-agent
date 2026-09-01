import { useEffect, useState } from 'react';
import type { ChatMessage } from '../hooks/useAgentRun';

function useEnter(): boolean {
  const [active, setActive] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setActive(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return active;
}

function ShieldIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l7 3v6c0 5-3 8-7 9-4-1-7-4-7-9V6l7-3Z" />
      <path d="M9.5 12.2l1.8 1.8 3.4-3.6" />
    </svg>
  );
}

export default function Message({
  message,
  onApprove,
  onCancel,
}: {
  message: ChatMessage;
  onApprove: () => void;
  onCancel: () => void;
}) {
  const active = useEnter();
  const enterCls = `enter${active ? ' enter-active' : ''}`;

  switch (message.kind) {
    case 'user':
      return (
        <div className={`msg msg-user ${enterCls}`}>
          <div className="msg-bubble">{message.text}</div>
        </div>
      );

    case 'program':
      return (
        <details className={`program-view ${enterCls}`}>
          <summary>Show generated program</summary>
          <pre className="program-code">{message.text}</pre>
        </details>
      );

    case 'stats':
      return (
        <div className={`msg-stats ${enterCls}`}>
          {message.tokens} tokens · {(message.ms / 1000).toFixed(1)}s ·{' '}
          {message.ms > 0 ? (message.tokens / (message.ms / 1000)).toFixed(1) : '0.0'} tok/s
        </div>
      );

    case 'agent-text':
      return (
        <div className={`msg msg-agent msg-closing ${enterCls}`}>
          <div className="msg-bubble">{message.text}</div>
        </div>
      );

    case 'note':
      return (
        <div className={`msg msg-note ${enterCls}`}>
          <div className="msg-bubble">{message.text}</div>
        </div>
      );

    case 'tool-call':
      return (
        <div className={`tool-call effect-${message.effect.toLowerCase()} ${enterCls}`}>
          <span className="effect-tag">{message.effect}</span>
          <span className="tool-name">{message.name}</span>
          <span className="tool-detail">{message.detail}</span>
        </div>
      );

    case 'gate':
      return (
        <div className={`approval-gate${message.state !== 'pending' ? ' gate-resolved' : ''} ${enterCls}`}>
          <div className="gate-head">
            <ShieldIcon />
            Approval required
          </div>
          <p className="gate-text">{message.text}</p>
          <div className="gate-actions">
            <button
              className="gate-btn gate-approve"
              type="button"
              disabled={message.state !== 'pending'}
              onClick={onApprove}
            >
              {message.state === 'approved' ? 'Approved' : 'Approve'}
            </button>
            <button className="gate-btn gate-cancel" type="button" disabled={message.state !== 'pending'} onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      );

    default:
      return null;
  }
}
