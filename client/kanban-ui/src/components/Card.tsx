import { isOverdue, userById, type KanbanCard, type KanbanState } from '../data/board';

function humanDate(epoch: number): string {
  const d = new Date(epoch * 1000);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

export default function Card({ card, board, now }: { card: KanbanCard; board: KanbanState; now: number }) {
  const user = userById(board, card.assignee);
  const overdue = isOverdue(card, now);
  const cls = ['card', card.urgent ? 'urgent' : ''].filter(Boolean).join(' ');
  return (
    <div className={cls} data-id={card.id}>
      <div className="card-title">{card.title}</div>
      <div className="card-meta">
        <span className="avatar">{user ? initials(user.name) : '?'}</span>
        <span className="assignee-name">{user?.name ?? 'Unassigned'}</span>
        <span className={`due${overdue ? ' due-overdue' : ''}`}>
          {humanDate(card.due)}
          {overdue && <span className="sr-only"> — overdue</span>}
        </span>
      </div>
      {card.urgent && <span className="sr-only">Urgent</span>}
    </div>
  );
}

function initials(name: string): string {
  return name
    .split(' ')
    .map((p) => p[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();
}
