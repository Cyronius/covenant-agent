// Integration glue for the browser-inference stand-in POC. Wires together
// three pieces built independently against a fixed contract:
//   - ./llm.js     createPlanner({modelUrl, grammarUrl}) -> {backend, loadMs, generate}
//   - ./prompt.js  buildFullPrompt(taskInputText, registers, prior), STOP
//   - /fixtures/tasks.json  [{task_id, level, world, request, input_text}, ...]
// and the dev server's POST /validate (see server/README.md for the request/
// response contract, including the pause_types/pause_envs PAUSE-continuation
// protocol this file implements client-side).
//
// See .claude/plans/browser-inference-standin.md for scope and background.

import { createPlanner } from './llm.js';
import { buildFullPrompt, STOP } from './prompt.js';

// POC-only segment cap. harness/run.py uses MAX_SEGMENTS = 8; kept smaller
// here since this is a manual, one-task-at-a-time demo, not a batch eval.
const MAX_SEGMENTS = 4;

const els = {
  taskSelect: document.querySelector('#task-select'),
  loadBtn: document.querySelector('#load-btn'),
  genBtn: document.querySelector('#gen-btn'),
  resetBtn: document.querySelector('#reset-btn'),
  status: document.querySelector('#status'),
  log: document.querySelector('#log'),
  programOut: document.querySelector('#program-out'),
  stateOut: document.querySelector('#state-out'),
};

let tasks = [];
let planner = null; // set once, reused across tasks/runs
let run = null; // { taskId, prior: string[], registers, state, pauseTypes, segIdx }

function log(msg) {
  const line = document.createElement('div');
  const ts = new Date().toISOString().slice(11, 19);
  line.textContent = `[${ts}] ${msg}`;
  els.log.appendChild(line);
  els.log.scrollTop = els.log.scrollHeight;
}

function currentTask() {
  return tasks.find((t) => t.task_id === els.taskSelect.value) || null;
}

function resetRun() {
  const task = currentTask();
  run = task
    ? { taskId: task.task_id, prior: [], registers: null, state: null, pauseTypes: null, segIdx: 0 }
    : null;
  els.programOut.textContent = '';
  els.stateOut.textContent = '';
}

async function loadTasks() {
  const res = await fetch('/fixtures/tasks.json');
  if (!res.ok) {
    log(`ERROR fetching /fixtures/tasks.json: ${res.status} ${res.statusText}`);
    return;
  }
  tasks = await res.json();
  els.taskSelect.innerHTML = tasks
    .map((t) => `<option value="${t.task_id}">L${t.level} — ${t.request}</option>`)
    .join('');
  resetRun();
  log(`Loaded ${tasks.length} task fixtures.`);
}

async function loadModel() {
  els.loadBtn.disabled = true;
  els.status.textContent = 'loading… (first run fetches ~800MB, be patient)';
  log('Loading model from /models/qwen3.5-0.8b-condB-q8.gguf with grammar /agent_core.gbnf …');
  try {
    planner = await createPlanner({
      modelUrl: '/models/qwen3.5-0.8b-condB-q8.gguf',
      grammarUrl: '/agent_core.gbnf',
    });
    els.status.textContent =
      `loaded — backend: ${planner.backend}, load time: ${planner.loadMs.toFixed(0)} ms ` +
      `(backend is a capability check, not a confirmed per-call trace — see src/llm.md)`;
    log(`Model ready. backend=${planner.backend} loadMs=${planner.loadMs.toFixed(0)}`);
    els.genBtn.disabled = false;
  } catch (err) {
    els.status.textContent = `load failed: ${err.message}`;
    log(`ERROR loading model: ${err.stack || err}`);
    els.loadBtn.disabled = false;
  }
}

async function generateAndRunSegment() {
  if (!planner || !run) return;
  const task = tasks.find((t) => t.task_id === run.taskId);
  if (!task) return;

  if (run.segIdx >= MAX_SEGMENTS) {
    log(`Segment cap (${MAX_SEGMENTS}) reached — stopping without a final STOP.`);
    return;
  }

  const prompt = buildFullPrompt(task.input_text, run.registers, run.prior);
  log(`Generating segment ${run.segIdx}…`);
  const gen = await planner.generate(prompt, { maxTokens: 250, stop: STOP });
  log(`Generated (${gen.tokensOut} tok, ${gen.genMs.toFixed(0)} ms)`);

  const header = run.segIdx > 0 ? `\n--- segment ${run.segIdx} ---\n` : '';
  els.programOut.textContent += header + gen.text;
  run.prior.push(gen.text);

  const body = {
    task_id: run.taskId,
    text: gen.text,
    state: run.state,
    registers: run.registers,
    pause_types: run.pauseTypes,
  };
  log('POST /validate …');
  const t0 = performance.now();
  const res = await fetch('/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  const validateMs = performance.now() - t0;
  log(`/validate -> status=${data.status} (${validateMs.toFixed(0)} ms)`);
  if (data.diagnostics && data.diagnostics.length) {
    log(`diagnostics: ${JSON.stringify(data.diagnostics)}`);
  }

  els.stateOut.textContent = JSON.stringify(data, null, 2);

  run.state = data.final_state;
  run.registers = data.registers;
  run.pauseTypes = data.pause_envs ? data.pause_envs[0] : null;
  run.segIdx += 1;

  if (data.status === 'paused') {
    log('Task paused — generating continuation segment…');
    await generateAndRunSegment();
  } else {
    log(`Run finished with status: ${data.status}`);
  }
}

els.loadBtn.addEventListener('click', loadModel);
els.taskSelect.addEventListener('change', resetRun);
els.resetBtn.addEventListener('click', resetRun);
els.genBtn.addEventListener('click', async () => {
  if (!run) resetRun();
  els.genBtn.disabled = true;
  try {
    await generateAndRunSegment();
  } catch (err) {
    log(`ERROR: ${err.stack || err}`);
  } finally {
    els.genBtn.disabled = false;
  }
});

loadTasks();
