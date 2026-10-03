import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Proxies /validate, /kanban_prompt, /rpg_new, /rpg_prompt, /db_prompt,
// /db_new, /apps and /plan to the running `python server/dev_server.py`
// (default port 8080) so this app is effectively same-origin with the real
// planner/execution backend — see .claude/plans/example-host-and-new-worlds.md
// and server/README.md.
// COVENANT_BACKEND lets a second dev server (another port, another
// checkpoint) be driven without editing this file.
const BACKEND = process.env.COVENANT_BACKEND || 'http://localhost:8080';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // client/shared/ lives outside this app's root and there is no workspace
    // file, so Vite's default fs allow-list would 403 the /@fs/ request for
    // every shared module.
    fs: { allow: [import.meta.dirname + '/..'] },
    proxy: {
      '/validate': BACKEND,
      '/kanban_prompt': BACKEND,
      '/rpg_new': BACKEND,
      '/rpg_prompt': BACKEND,
      '/db_prompt': BACKEND,
      '/db_new': BACKEND,
      '/apps': BACKEND,
      '/plan': BACKEND,
    },
  },
});
