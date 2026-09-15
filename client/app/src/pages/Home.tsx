import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './home.css';

interface AppEntry {
  slug: string;
  path: string;
  title: string;
  blurb: string;
}

// Static fallback for a plain static host with no dev_server.py behind it
// (GET /apps 404s) — same three demos, just not fetched. Keep in sync with
// server/dev_server.py's APPS.
const FALLBACK: AppEntry[] = [
  { slug: 'kanban', path: '/kanban', title: 'Kanban Board', blurb: 'Free-typed requests against a fake team board.' },
  { slug: 'rpg', path: '/rpg', title: 'Dungeon Agent', blurb: 'A grid world, played turn by turn.' },
  { slug: 'db', path: '/db', title: 'Database Analyst', blurb: 'Ask questions of a fake CRM — customers, tickets, invoices.' },
];

export default function Home() {
  const [apps, setApps] = useState<AppEntry[]>(FALLBACK);

  useEffect(() => {
    fetch('/apps')
      .then((r) => r.json())
      .then((body) => {
        if (Array.isArray(body?.apps) && body.apps.length > 0) setApps(body.apps);
      })
      .catch(() => {
        // no /apps (plain static host) — the fallback list above is already showing
      });
  }, []);

  return (
    <div className="home">
      <header className="home-head">
        <h1>Agent Core Demos</h1>
        <p>
          Real in-browser inference, a real grammar-constrained IR, a real sandboxed runtime. The board/dungeon/CRM
          data is fake; everything else genuinely runs.
        </p>
      </header>
      <div className="home-grid">
        {apps.map((a) => (
          <Link key={a.slug} to={a.path} className="home-card">
            <span className="home-card-title">{a.title}</span>
            <span className="home-card-blurb">{a.blurb}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
