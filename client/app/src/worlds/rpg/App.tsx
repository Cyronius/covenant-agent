import { Link } from 'react-router-dom';
import { DungeonView } from './components/DungeonView';
import { InferenceControls } from './components/InferenceControls';
import { TurnLog } from './components/TurnLog';
import { useDungeonRun } from './hooks/useDungeonRun';
import { carried, player } from './lib/rpgApi';
// every top-level selector in here is scoped under .world-rpg (scope_css's
// one-time pass, plan example-host-and-new-worlds.md §3) — see
// ../kanban/App.tsx's matching comment.
import './styles.css';

function Hearts({ hp, max }: { hp: number; max: number }) {
  return (
    <span className="hearts" aria-label={`${hp} of ${max} HP`}>
      {Array.from({ length: max }, (_, i) => (
        <span key={i} className={i < hp ? 'heart full' : 'heart empty'} />
      ))}
    </span>
  );
}

export default function App() {
  const {
    state,
    window: view,
    nearby,
    turns,
    modelStatus,
    busy,
    auto,
    note,
    inference,
    setInferenceMode,
    models,
    model,
    selectModel,
    step,
    startAuto,
    stopAuto,
    restart,
  } = useDungeonRun();

  const ready = modelStatus.phase === 'ready';
  const over = state ? state.status !== 'playing' : false;

  return (
    <div className="world-rpg app">
      <header className="chrome">
        <div className="brand">Dungeon Agent</div>
        <div className="chrome-right">
          <Link className="all-demos-link" to="/">
            ← All demos
          </Link>
          <InferenceControls
            mode={inference}
            model={model}
            models={models}
            disabled={busy || auto || modelStatus.phase === 'loading'}
            onMode={setInferenceMode}
            onModel={selectModel}
          />
          {ready && (
            <span className="model-status ready">
              <span className="dot" />
              {modelStatus.backend} · {Math.round(modelStatus.loadMs)}ms load
            </span>
          )}
          {modelStatus.phase === 'loading' && (
            <span className="model-status loading">loading {modelStatus.stage}…</span>
          )}
          {modelStatus.phase === 'error' && (
            <span className="model-status error">{modelStatus.message}</span>
          )}
        </div>
      </header>

      <main>
        <section className="stage">
          {state ? (
            <>
              <div className="hud">
                <Hearts hp={player(state).hp} max={player(state).max_hp} />
                <span className="hud-item">turn {state.turn}/{state.max_turns}</span>
                <span className="hud-item">
                  carrying {carried(state).map((i) => i.kind).join(', ') || 'nothing'}
                </span>
                {over && <span className={`verdict ${state.status}`}>{state.status}</span>}
              </div>

              <DungeonView state={state} />

              <p className="quest">{state.quest}</p>

              <div className="controls">
                <button type="button" onClick={() => void step()} disabled={!ready || busy || auto || over}>
                  Step
                </button>
                {auto ? (
                  <button type="button" onClick={stopAuto}>
                    Stop
                  </button>
                ) : (
                  <button type="button" onClick={startAuto} disabled={!ready || busy || over}>
                    Auto-play
                  </button>
                )}
                <button type="button" onClick={() => void restart()} disabled={busy}>
                  New game
                </button>
                {busy && <span className="thinking">thinking…</span>}
              </div>
              {note && <p className="note">{note}</p>}

              <div className="seen">
                <h2>What the model sees this turn</h2>
                <pre className="window">{view.join('\n')}</pre>
                <p className="legend">
                  # wall · . floor · E stairs · D locked door · g goblin · k key · p potion
                </p>
                <p className="nearby">
                  {nearby.length
                    ? nearby
                        .map((n) => `${n.kind ?? 'door'} ${n.here ? 'here' : `${Math.abs(n.dx)}${n.dx > 0 ? 'E' : n.dx < 0 ? 'W' : ''}${Math.abs(n.dy)}${n.dy > 0 ? 'S' : n.dy < 0 ? 'N' : ''}`}`)
                        .join(' · ')
                    : 'nothing in view'}
                </p>
              </div>
            </>
          ) : (
            <p className="empty">Dealing a dungeon…</p>
          )}
        </section>

        <aside className="log">
          <h2>Turns</h2>
          <TurnLog turns={turns} />
        </aside>
      </main>
    </div>
  );
}
