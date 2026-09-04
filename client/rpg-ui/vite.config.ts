import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Proxies /validate, /rpg_new, /rpg_prompt, /plan, /models*, /agent_core.gbnf to the
// running `python server/dev_server.py` (default port 8080) so
// the dungeon app
// is effectively same-origin with the real model/grammar/execution
// backend — see .claude/plans/rpg-demo-app.md and
// server/README.md. Also sets the same
// Cross-Origin-Opener-Policy/Cross-Origin-Embedder-Policy headers
// dev_server.py sends, required for wllama's multi-threaded WASM path
// (SharedArrayBuffer) — see src/lib/llm.md's "Multi-threading" note.
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
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      '/validate': BACKEND,
      '/rpg_new': BACKEND,
      '/rpg_prompt': BACKEND,
      '/plan': BACKEND,
      '/models': BACKEND,
      '/agent_core.gbnf': BACKEND,
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
