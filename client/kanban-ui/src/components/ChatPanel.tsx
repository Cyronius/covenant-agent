import { useEffect, useRef, useState } from 'react';
import Message from './Message';
import type { ChatMessage } from '../hooks/useAgentRun';

function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

function ChatIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l7 3v6c0 5-3 8-7 9-4-1-7-4-7-9V6l7-3Z" />
      <path d="M9.5 12.2l1.8 1.8 3.4-3.6" />
    </svg>
  );
}

export default function ChatPanel({
  messages,
  busy,
  onSend,
  onApprove,
  onCancel,
}: {
  messages: ChatMessage[];
  busy: boolean;
  onSend: (text: string) => void;
  onApprove: () => void;
  onCancel: () => void;
}) {
  const listRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState('');

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const submit = () => {
    const text = draft.trim();
    if (!text || busy) return;
    onSend(text);
    setDraft('');
  };

  return (
    <aside className="chat" aria-label="Agent chat">
      <div className="chat-head">
        <ChatIcon />
        Agent
      </div>
      <div className="chat-messages" ref={listRef}>
        {messages.length === 0 && (
          <p className="empty-note">
            Tell it what to do on the board — e.g. "archive the overdue cards assigned to Bob" or "message Priya about
            her overdue card."
          </p>
        )}
        {messages.map((m) => (
          <Message key={m.id} message={m} onApprove={onApprove} onCancel={onCancel} />
        ))}
        {busy && (
          <div className="msg msg-agent">
            <div className="msg-bubble thinking-bubble">
              <Spinner />
              Thinking…
            </div>
          </div>
        )}
      </div>
      <div className="chat-input-bar">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') submit();
          }}
          disabled={busy}
          placeholder="Ask the agent to do something on the board…"
        />
        <button type="button" disabled={busy || !draft.trim()} onClick={submit}>
          {busy ? (
            <>
              <Spinner />
              Working…
            </>
          ) : (
            'Send'
          )}
        </button>
      </div>
    </aside>
  );
}
