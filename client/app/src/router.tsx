import { lazy, Suspense } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import Home from './pages/Home';

// Each world is its own lazy-loaded chunk: its App, hook, and CSS only load
// when its route is visited, so at most one world's model-loading hook is
// ever mounted at once (react-router unmounts the previous route's tree on
// navigation, which is what makes useAgentRun.ts's/useDbRun.ts's/
// useDungeonRun.ts's unmount-unload cleanup effect actually run). See plan
// example-host-and-new-worlds.md §2-3.
const KanbanApp = lazy(() => import('./worlds/kanban/App'));
const RpgApp = lazy(() => import('./worlds/rpg/App'));
const DbApp = lazy(() => import('./worlds/db/App'));

function RouteLoading() {
  return <div style={{ padding: '2rem', fontFamily: 'sans-serif', color: '#666' }}>Loading…</div>;
}

const router = createBrowserRouter([
  { path: '/', element: <Home /> },
  {
    path: '/kanban',
    element: (
      <Suspense fallback={<RouteLoading />}>
        <KanbanApp />
      </Suspense>
    ),
  },
  {
    path: '/rpg',
    element: (
      <Suspense fallback={<RouteLoading />}>
        <RpgApp />
      </Suspense>
    ),
  },
  {
    path: '/db',
    element: (
      <Suspense fallback={<RouteLoading />}>
        <DbApp />
      </Suspense>
    ),
  },
]);

export default function AppRouter() {
  return <RouterProvider router={router} />;
}
