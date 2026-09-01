import { useState } from 'react';
import Card from './Card';
import { NOW, type KanbanState } from '../data/board';

const COLUMNS: { status: 'todo' | 'doing' | 'done'; label: string }[] = [
  { status: 'todo', label: 'Todo' },
  { status: 'doing', label: 'Doing' },
  { status: 'done', label: 'Done' },
];

function BotanicalMark() {
  return (
    <svg className="botanical-mark" width="30" height="14" viewBox="0 0 28 14" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round">
      <path d="M1 7 H27" />
      <path d="M9 7 C9 3 12 2 14 2 C13 5 11 7 9 7Z" fill="currentColor" stroke="none" opacity=".55" />
      <path d="M19 7 C19 11 16 12 14 12 C15 9 17 7 19 7Z" fill="currentColor" stroke="none" opacity=".55" />
    </svg>
  );
}

export default function KanbanBoard({ board }: { board: KanbanState }) {
  const [archiveOpen, setArchiveOpen] = useState(false);
  const archived = board.entities.card.filter((c) => c.archived);

  return (
    <section className="board" aria-label="Kanban board">
      <div className="columns">
        {COLUMNS.map(({ status, label }) => {
          const cards = board.entities.card.filter((c) => c.status === status && !c.archived);
          return (
            <div className="column" key={status} data-status={status}>
              <div className="col-head">
                {label}
                <span className="count">{cards.length}</span>
                <BotanicalMark />
              </div>
              <div className="cards">
                {cards.length ? (
                  cards.map((c) => <Card key={c.id} card={c} board={board} now={NOW} />)
                ) : (
                  <p className="empty-note">Nothing here.</p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="archive-drawer">
        <button
          className={`archive-toggle${archiveOpen ? ' open' : ''}`}
          type="button"
          aria-expanded={archiveOpen}
          aria-controls="archiveBody"
          onClick={() => setArchiveOpen((v) => !v)}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 9l6 6 6-6" />
          </svg>
          Archived
          <span className="badge">{archived.length}</span>
        </button>
        <div
          id="archiveBody"
          className={`archive-body${archiveOpen ? ' open' : ''}`}
          style={{ maxHeight: archiveOpen ? 2000 : 0 }}
        >
          {archived.length ? (
            archived.map((c) => <Card key={c.id} card={c} board={board} now={NOW} />)
          ) : (
            <p className="empty-note">Empty.</p>
          )}
        </div>
      </div>
    </section>
  );
}
