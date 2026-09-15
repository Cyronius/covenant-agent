import { useEffect, useRef, useState } from 'react';
import Message from './Message';
import type { ChatMessage } from '../hooks/useDbRun';

function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
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
      <div className="chat-head">Ask a question</div>
      <div className="chat-messages" ref={listRef}>
        {messages.length === 0 && (
          <p className="empty-note">
            Try "which customers are delinquent" or "close ticket 2" or "list open tickets sorted by priority."
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
          placeholder="Ask about customers, tickets, or invoices…"
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
