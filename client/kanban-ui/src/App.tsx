import { useState } from 'react';
import KanbanBoard from './components/KanbanBoard';
import ChatPanel from './components/ChatPanel';
import Toast from './components/Toast';
import LoadingView from './components/LoadingView';
import ToolsPanel from './components/ToolsPanel';
import { useAgentRun } from './hooks/useAgentRun';

function LeafIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 20c8-1 14-7 15-16-9 1-15 7-16 16Z" fill="currentColor" opacity=".2" stroke="none" />
      <path d="M4 20c8-1 14-7 15-16-9 1-15 7-16 16Z" />
      <path d="M6.5 17.5c3-4 6.5-7.5 11-10.5" />
    </svg>
  );
}

function ResetIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 1 2.6 6.4" />
      <path d="M3 4v6h6" />
    </svg>
  );
}

function ToolsIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14.7 6.3a4 4 0 0 1-5.4 5.4L4 17l3 3 5.3-5.3a4 4 0 0 1 5.4-5.4l-2.8 2.8-2-2Z" />
    </svg>
  );
}

export default function App() {
  const {
    board,
    messages,
    modelStatus,
    busy,
    toast,
    cpuOnly,
    toggleCpuOnly,
    sendMessage,
    approveGate,
    cancelGate,
    resetBoard,
  } = useAgentRun();

  const ready = modelStatus.phase === 'ready';
  const [toolsOpen, setToolsOpen] = useState(false);

  return (
    <>
      <header className="chrome">
        <div className="wordmark">
          <LeafIcon />
          Kanban Board Demo
        </div>
        <div className="chrome-right">
          <span className="tagline">board agent</span>
          {ready && (
            <button className="action-btn" type="button" onClick={() => setToolsOpen(true)}>
              <ToolsIcon />
              Tools
            </button>
          )}
          <label
            className="cpu-toggle"
            title="Force WASM-only decoding, skipping WebGPU offload. Changing this reloads the model."
          >
            <input
              type="checkbox"
              checked={cpuOnly}
              disabled={modelStatus.phase !== 'ready' || busy}
              onChange={toggleCpuOnly}
            />
            CPU only
          </label>
          {ready && (
            <span className="model-status ready">
              <span className="dot" />
              {modelStatus.backend} · {Math.round(modelStatus.loadMs)}ms load
            </span>
          )}
          {ready && messages.length > 0 && (
            <button className="replay-btn" type="button" onClick={resetBoard} disabled={busy}>
              <ResetIcon />
              Reset board
            </button>
          )}
        </div>
      </header>

      {!ready ? (
        <LoadingView status={modelStatus} />
      ) : (
        <main className="layout">
          <KanbanBoard board={board} />
          <ChatPanel messages={messages} busy={busy} onSend={sendMessage} onApprove={approveGate} onCancel={cancelGate} />
        </main>
      )}

      <Toast text={toast} />
      <ToolsPanel open={toolsOpen} onClose={() => setToolsOpen(false)} />
    </>
  );
}
