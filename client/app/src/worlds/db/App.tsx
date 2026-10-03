import { Link } from 'react-router-dom';
import ChatPanel from './components/ChatPanel';
import LoadingView from './components/LoadingView';
import SchemaPanel from './components/SchemaPanel';
import ResultsTable from './components/ResultsTable';
import { useDbRun } from './hooks/useDbRun';
// every top-level selector in here is scoped under .world-db — see
// ../kanban/App.tsx's matching comment.
import './styles.css';

export default function App() {
  const {
    lastResult,
    messages,
    modelStatus,
    busy,
    sendMessage,
    approveGate,
    cancelGate,
  } = useDbRun();

  const ready = modelStatus.phase === 'ready';

  return (
    <div className="world-db">
      <header className="chrome">
        <div className="wordmark">Database Analyst</div>
        <div className="chrome-right">
          <Link className="all-demos-link" to="/">
            ← All demos
          </Link>
          {ready && (
            <span className="model-status ready">
              <span className="dot" />
              {modelStatus.model} · {Math.round(modelStatus.loadMs)}ms load
            </span>
          )}
        </div>
      </header>

      {!ready ? (
        <LoadingView status={modelStatus} />
      ) : (
        <main className="layout">
          <SchemaPanel />
          <section className="results-pane">
            <h2>Results</h2>
            <ResultsTable value={lastResult} />
          </section>
          <ChatPanel messages={messages} busy={busy} onSend={sendMessage} onApprove={approveGate} onCancel={cancelGate} />
        </main>
      )}
    </div>
  );
}
