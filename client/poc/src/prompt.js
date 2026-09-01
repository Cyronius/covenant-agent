// Direct port of baselines/qwen/run_a.py:27-104 (SYSTEM string + build_prompt +
// make_planner's prompt assembly) to plain JS, for the browser inference POC.
// Plain ES module, no bundler, no Node-only APIs — loadable via
// `<script type="module">`.
//
// Keep SYSTEM's content byte-identical to the Python source (same
// instructions, same worked example). If run_a.py's SYSTEM string changes,
// update this file to match.

export const SYSTEM = `You translate task requests into Agent Core programs.

Agent Core instructions (one per line, two-space indent for block bodies):
CALL Tn args -> r        call tool Tn (args: registers, rX.Fn, Cn, NOW)
FILTER r pred -> r       keep list elements matching pred, e.g. F3 EQ C0 AND NOT F5 LT NOW
SORT r Fn ASC|DESC -> r  sort list by field
SELECT r i -> r          i-th element (0-based); FIRST r -> r; COUNT r -> r; MAP r Fn -> r
GET r.Fn -> r            extract field
LET x -> r               bind value
FOREACH r -> rElem       loop over list, body indented below
IF cond / ELSE           branch, bodies indented; cond compares operands, e.g. r0 EQ C1
PARALLEL                 body: CALL lines only, run concurrently
TRY [RETRY n] -> r       run body, catch tool errors; r gets OK or error code
STOP | RETURN x | PAUSE  end program (PAUSE = report back; a continuation follows later)

Rules: registers r0-r15 in order of first use. Use ONLY the T/F/C symbols
listed for the task; every literal value must be a C symbol. TIME fields are
timestamps: "more than N days ago" / "overdue" means Fx LT (cutoff/NOW).
Programs are SHORT — typically 2 to 8 lines — and always end with STOP
(or PAUSE when the request says to report back before acting).

Example:
TOOLS:
T0 () -> LIST OBJ:card [READ] :: List all cards.
T1 (F2:ID:card) -> - [DELETE] :: Delete a card.
FIELDS:
F1 card TIME :: card.due
F2 card ID:card :: card.id
F3 card BOOL :: card.urgent
CONSTANTS:
C0 BOOL :: true
REQUEST: Delete the overdue cards, but keep the urgent ones.
PROGRAM:
CALL T0 -> r0
FILTER r0 F1 LT NOW AND NOT F3 EQ C0 -> r1
FOREACH r1 -> r2
  CALL T1 r2.F2
STOP

Output ONLY the program, nothing else.`;

// taskInputText: the REQUEST:/TOOLS:/FIELDS:/CONSTANTS: block for one task
//   (produced server-side, equivalent to Python's serialize_context() — this
//   function just consumes the string as given).
// registers: object of currently-bound register values (or null/undefined on
//   the first segment).
// prior: array of previously-generated program-segment strings (empty on the
//   first segment).
export function buildPrompt(taskInputText, registers, prior) {
  const parts = [taskInputText.trim()];
  if (prior && prior.length) {
    parts.push("PROGRAM SO FAR (already executed, ended at PAUSE):");
    for (const p of prior) parts.push(p.trim());
    parts.push("REGISTERS NOW BOUND (values from the run):");
    parts.push(JSON.stringify(registers).slice(0, 1500));
    parts.push("Write ONLY the continuation program (registers above " +
                "are still bound; do not re-fetch).");
  }
  parts.push("PROGRAM:");
  return parts.join("\n");
}

// Full prompt text to hand the model, including the Qwen chat markup with
// the explicit empty <think></think> block — exactly matching run_a.py's
// make_planner() (baselines/qwen/run_a.py:93-95). Keeps the model in
// no-think mode (the grammar forbids <think> anyway, which would otherwise
// force "reasoning" in program syntax).
export function buildFullPrompt(taskInputText, registers, prior) {
  const user = buildPrompt(taskInputText, registers, prior);
  return `<|im_start|>system\n${SYSTEM}<|im_end|>\n` +
         `<|im_start|>user\n${user}<|im_end|>\n` +
         `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

export const STOP = ["<|im_end|>"];
