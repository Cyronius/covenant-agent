import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Proxies /validate, /kanban_prompt, /plan, /models/*, /agent_core.gbnf to the
// running `python server/dev_server.py` (default port 8080) so
// this app
// is effectively same-origin with the real model/grammar/execution
// backend — see .claude/plans/understory-kanban-frontend.md and
// server/README.md. Also sets the same
// Cross-Origin-Opener-Policy/Cross-Origin-Embedder-Policy headers
// dev_server.py sends, required for wllama's multi-threaded WASM path
// (SharedArrayBuffer) — see src/lib/llm.md's "Multi-threading" note.
const BACKEND = 'http://localhost:8080';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      '/validate': BACKEND,
      '/kanban_prompt': BACKEND,
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
