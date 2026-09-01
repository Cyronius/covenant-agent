# Covenant Agent Model — Simplified Experimental Plan

**Status:** Draft v1 — 2026-08-29
**Audience:** Claude Code (implementation agent) and the project owner
**Language:** Python is research scaffolding only — models, data generation, and the measurement harness. It is disposable. The shipping agent stack is TypeScript (see §0.2); JavaScript is the compiled program target and the sandbox runtime.

---

## 0. Purpose

Test one question as cheaply as possible:

> Given an English request and arbitrary tool schemas, can a small neural network reliably construct the correct executable program?

Everything in this document is in service of that question. Anything that does not move it forward is deferred (see §12).

**Project goal (owner, 2026-08-29):** a planner that runs in the end-user's
**browser**. Parameter budget **0–300M, as small as possible** — the
experiments exist to find the smallest model that clears the accuracy bar.
Qwen-class 0.5B+ models are baselines (R2), not candidates; they exceed the
budget. Size, download weight, and in-browser latency are first-class
success criteria alongside goal_success.

**Deployment tiers (owner, amended 2026-08-31):** the browser client is the
primary target and the 0–300M budget above still gates it — unchanged.
Separately, a **server-hosted fallback** ships for resource-constrained
clients (no WebGPU, low memory, etc.), running whatever model currently
passes the accuracy bar regardless of size — today that's the R2
Condition-B model. This tier has no size ceiling (server RAM/GPU, not a
browser tab) and is **not gated on R3**: it can ship independent of
whether the tiny-model track ever clears the browser budget. The two tiers
share the same harness/pipeline/sandbox; "ship server-side" is an API
wrapper around what already exists in `harness/run.py` +
`runtime/sandbox.js`, not new research. **Future work, not started** —
noted here so the decoupling is on record; the active track is the
browser tier (R3 onward, plus the §0.3 stand-in work).

This plan replaces the earlier 38-section draft. It keeps the foundation, the size curve, constrained decoding, compiler/execution feedback, tool grounding, English robustness, effect-typed safety, and real-trace distillation. It drops modular blocks, routers, Neural IR, Mixture of Widths, attention sparsity, and diffusion-style construction as premature.

### Core hypotheses actually being tested

| ID | Hypothesis | Tested by |
|----|-----------|-----------|
| H1 | A restricted executable IR dramatically reduces the model capacity needed for agentic behavior | R1, R3 |
| H2 | Grammar/type/schema-constrained decoding lets a smaller model match a larger unconstrained one | R3 |
| H3 | Compiler and runtime feedback can substitute for parameters | R4 |
| H4 | The model can ground unseen tools from schemas rather than memorize APIs | R5 |
| H5 | The English request distribution can be narrow enough for a tiny encoder | R6 |
| H6 | Explicit effects give deterministic safety that a model-based check cannot | R7 |
| H7 | Real-agent traces close the synthetic-to-real gap | R8 |

Strategy: try hard to disprove each one cheaply. Surviving hypotheses feed the next prototype.

### 0.1 Relationship to Covenant

Agent Core is a from-scratch rewrite of the ideas in
[Covenant](https://github.com/Cyronius/covenant), the experimental AI-first
language. We have wide latitude to diverge from it; the divergences below are
deliberate, not drift.

**Kept from Covenant:**
- Machine-first IR — no human authors it; deterministic structure with one
  valid way to write everything.
- Explicit declared effects, statically computable and enforced at runtime.
- Canonical form with round-trip guarantees (our round-trip tests are the
  analogue of Covenant's canonical text printer).

**Deliberately dropped:**
- SSA snippets with generated names → numbered registers `r0…r15`. A tiny
  model needs a tiny, fixed vocabulary.
- Node IDs, symbol graph, query engine — retrieval infrastructure for large
  codebases; irrelevant to single-program generation.
- WASM target (Deno/Node/browser via bytecode) → direct JavaScript. The
  execution model here is SandwichTS-style: generated JS calling tool stubs
  in a browser iframe/worker or Node sandbox (§0.2). WASM adds a toolchain
  for no benefit at this program size.

If a future change re-imports a dropped concept, note here which result
motivated it (§14 applies).

### 0.2 End-state stack (TypeScript)

The production agent stack is TypeScript. The deliverable is a TS pipeline —
parse → typecheck → effects → compile to JS (or interpret the IR directly) →
effect gate → PAUSE protocol — running in the browser (iframe/worker,
SandwichTS-style) or server-side Node. The Python `core/` is the reference
implementation used to iterate on the IR cheaply; it never ships.

**Timing:** port after R1 passes. R1 may force IR revisions; porting a moving
IR is wasted work. When the port lands, add differential tests: the TS and
Python compilers must emit byte-identical JS for every `spec/examples/`
program and for property-generated programs from F4. Until then, no
production code depends on the Python pipeline.

### 0.3 In-browser inference runtime (not yet built — R3+ deliverable)

Decided 2026-08-31 (not yet started; blocked on R3 producing a model to
export): **onnxruntime-web**, not WebLLM/MLC. WebLLM's compiled-model
registry targets chat-scale (1B+) architectures MLC has pre-compiled with
TVM; it does not help a custom <300M architecture, which would need its own
from-scratch MLC compilation to use WebLLM at all. onnxruntime-web gives
both target backends — WebGPU execution provider and WASM fallback — from
one library, against a plain ONNX export of whatever custom architecture R3
lands on.

**Real unsolved piece:** grammar-constrained decoding. `agent_core.gbnf`
constrains generation token-by-token today via llama.cpp's `grammar=` param
(`baselines/qwen/run_a.py:96-98`, R2 only). Neither onnxruntime-web nor
transformers.js ships GBNF-style constrained decoding; porting the Agent
Core grammar into an in-browser per-step logit mask (via onnxruntime-web's
custom logits-processor hook) is real R3+ engineering that must be budgeted
for, not something either library provides for free.

**Not applicable:** Vercel AI SDK (`useChat` et al.) — the model's I/O is
the symbolic, non-chat `serialize_context` format (`harness/context.py:133`),
not conversational text; a chat abstraction adds nothing here. Chrome's
built-in Prompt API only exposes Gemini Nano, not custom weights, so it's
out regardless of architecture.

**Depends on:** R3 (a trained, exportable model). Revisit this section when
R3 lands; until then it is a direction, not a build.

---

## 1. Non-goals (do not build these)

- General knowledge, multilingual support, user-facing prose generation
- Arbitrary programming languages or broad JS knowledge
- Maximum context length, tokenizer compression, GPU-specific kernels
- Modular neural blocks, shared state contracts, architecture routers, Neural IR
- Mixture of Widths (separate track, separate repo)
- Attention sparsity / linear attention replacements
- Diffusion-style or non-causal program construction
- Quantization below BF16 (until a model exists that is worth quantizing)

If a task seems to require one of these, stop and flag it rather than building it.

---

## 2. Foundation (prerequisite for every result)

The foundation is not a result. It is the measurement infrastructure. **No model training begins until F1–F5 run end to end on curriculum Levels 0–6.**

### F1 — Agent Core IR specification

A restricted, register-based IR that the model emits. Deterministically compiles to JavaScript.

**Vocabulary (initial):**

```
CALL GET SET LET FILTER MAP COUNT SORT SELECT FIRST FOREACH
IF ELSE AND OR NOT EQ LT GT CONTAINS
PARALLEL TRY RETRY RETURN STOP PAUSE
```

**Design rules:**

- Numbered registers `r0 … r15`, never generated variable names.
- Tools and fields are **dynamic runtime symbols** (`T0 T1 …`, `F0 F1 …`) assigned per request. Tool names are never tokenized permanently. Descriptions are supplied separately in the input.
- Every tool declares an effect set from `{READ, WRITE, DELETE, SEND, PAY, EXTERNAL}`. A program's effect set is computable statically before execution.
- `PAUSE` is a first-class execution boundary. The runtime executes up to `PAUSE`, returns register contents, and the planner resumes. This is the default execution model, not an ablation.
- Constants are referenced symbolically (`C0 C1 …`) with values held by the runtime.

**Example:**

```
CALL T4 -> r0
FILTER r0 F7 LT NOW -> r1
FOREACH r1 -> r2
  CALL T9 r2.F0 C3
STOP
```

**Deliverables:**

- `spec/agent_core.md` — grammar (EBNF), typing rules, effect rules, register semantics
- `spec/examples/` — at least 30 hand-written programs across Levels 0–10

### F2 — Parser, type checker, effect checker, compiler

- `core/parser.py` — Agent Core text → AST. Structured errors, not exceptions.
- `core/typecheck.py` — register binding, argument types, required fields, field existence on register contents.
- `core/effects.py` — static effect set for a program.
- `core/compile.py` — AST → JavaScript. Must be deterministic: same AST → byte-identical JS.
- Structured diagnostic vocabulary (used later as model feedback in R4):

```
UNBOUND <reg>
TYPE_ERROR <site> <expected> <got>
UNKNOWN_TOOL <sym>
UNKNOWN_FIELD <reg> <sym>
MISSING_ARG <tool> <field>
UNREACHABLE <line>
EFFECT_UNDECLARED <effect>
```

**Tests:** round-trip every `spec/examples/` program. Property test: random valid programs from F4 always parse, typecheck, and compile.

### F3 — Sandbox runtime and harness

- Synthetic worlds: Kanban/cards, CRM/customers/tickets, projects/users/managers. Each world has a schema, an initial state, and a tool set with schemas and effects.
- Sandboxed JS execution (Node, no network, no fs) with a deterministic tool implementation over in-memory state.
- Effect gate: any DELETE/SEND/PAY effect requires an approval token or the run halts with `EFFECT_BLOCKED`.
- Error injection: tools can be configured to return `NOT_FOUND`, `PERMISSION_DENIED`, `RATE_LIMITED`, `INVALID_ARGUMENT`, `PARTIAL_DATA`.
- `PAUSE` support: execution returns register state; the harness re-invokes the model with that state.

**Pipeline:**

```
request + tool schemas
  → model
  → Agent Core text
  → parse → typecheck → effects
  → compile → JS
  → sandbox execute (with PAUSE round-trips)
  → final state
  → metrics
```

**Metrics (per task, logged as JSONL):**

| Metric | Definition |
|--------|-----------|
| goal_success | final state == expected final state (**primary**) |
| parse_ok | syntactically valid |
| compile_ok | passed typecheck + effects + compile |
| tool_valid | every referenced tool exists |
| arg_valid | all argument types and required fields correct |
| exec_ok | JS ran without runtime error |
| tool_efficiency | actual tool calls / minimum required |
| program_efficiency | instructions / reference instructions |
| recovery_ok | on injected error, did the final state still match |
| unnecessary_destructive | count of DELETE/SEND/PAY effects not present in the reference |
| pauses | number of execution round-trips |
| latency_* | encode, generate, validate, compile, execute (ms) |
| tokens_in, tokens_out | |

### F4 — Backtranslation data generator

This is where the project lives or dies. Synthetic data with exact labels, unlimited volume.

```
random world (schema + state)
  → random valid Agent Core program (grammar-guided, level-targeted)
  → execute → expected final state
  → teacher model writes the English request
  → training pair: (request, tool schemas, program, expected state, level, effects)
```

**Requirements:**

- Level-targeted generation: a knob selects curriculum level 0–10.
- Phrasing diversity: the teacher is prompted for N distinct paraphrases per program; sample across formal/casual/terse styles.
- Adversarial English subset (for R6): negation, exceptions, quantifiers, temporal references, ordinal references. Tag these pairs.
- Tool renaming: every generated example gets fresh random `T`/`F` symbol assignments. Tool descriptions carry the meaning.
- Held-out tool families and held-out worlds are reserved at generation time and never enter training (for R5).
- Store with provenance: seed, generator version, teacher model, level, tags.

**Deliverable:** `data/gen/` with a CLI: `python -m data.gen --level 3 --n 10000 --seed 42 --out data/L3.jsonl`

### F5 — Curriculum

Task families of increasing difficulty. Each level needs ≥1,000 generated examples before R3 begins.

| Level | Name | Example | Requires |
|-------|------|---------|----------|
| 0 | Direct call | Delete card 14. | tool selection, arg extraction |
| 1 | Simple chain | Find the customer's email and send the invoice. | bind result → second call |
| 2 | Filtering | Archive every completed card. | query, filter, iterate |
| 3 | Multiple constraints | Archive overdue cards assigned to Bob except urgent ones. | boolean composition |
| 4 | Aggregation | Notify the account manager of the customer with the most open tickets. | group, count, compare, select |
| 5 | Branching | If delinquent notify billing, else send renewal. | IF/ELSE |
| 6 | Multi-stage dependency | Transfer projects owned by inactive users to their managers. | nested lookups |
| 7 | Parallel | Fetch customer, invoices, and history, then… | PARALLEL |
| 8 | Error recovery | (injected tool errors) | TRY/RETRY, replanning |
| 9 | Ambiguous scope | Clean up the old projects. | schema-grounded interpretation |
| 10 | Long loops | discovery → partial execution → replan | PAUSE round-trips |

**Foundation exit criterion:** a hand-written reference program for every level runs through F2→F3 and produces `goal_success = true`, and F4 produces valid data at every level.

---

## 3. R1 — Does the IR fit the job?

**Question:** Can Agent Core express the curriculum compactly, without escape hatches?

**Setup:**
- 1,000 tasks sampled across Levels 0–10 (weighted toward 2–8).
- Reference programs written by a strong model with a few-shot prompt, then spot-checked by hand (≥100 by hand).
- Also write the JavaScript equivalent for each.

**Measure:**
- Fraction that compile and pass goal_success with no escape hatch.
- Mean and p95 instructions per task.
- Mean and p95 tokens per task vs. the JS equivalent.
- Catalog of every case where the IR was awkward. Candidate missing primitives are listed for review, not added automatically.

**Pass:** ≥95% compile cleanly; median AC tokens **under the planner's own
vocabulary** (one token per IR keyword/symbol — the vocabulary R3 trains
with) ≤ 40% of the JS equivalent under a standard subword tokenizer.
*(Restated by owner 2026-08-30, motivated by R1: the original "same
tokenizer both sides" wording measured GPT-2's vocabulary — which splits
every IR symbol in two and has no relation to the planner's actual
emission cost. Both counts are stored per task in `results/r1_metrics.jsonl`;
original-wording result 0.545, restated result 0.269. See results/R1.md.)*
**Fail action:** revise F1 and rerun. **Nothing downstream proceeds until R1 passes.**
**Depends on:** Foundation.

---

## 4. R2 — What is the bar to beat?

**Question:** What does an off-the-shelf small model do on this harness, and how fast is it? This is the decision gate for the entire custom-model program.

**Setup:**
- Current smallest Qwen-class instruct models (≈0.5B and ≈1.5B). Verify current model names before starting; do not assume.
- Condition A: zero-shot with tool schemas in the prompt, grammar-constrained decoding (use F2's grammar with a constrained-decoding library).
- Condition B: LoRA fine-tune on 50k synthetic pairs from F4, same constrained decoding.
- Evaluate on Levels 0–10, plus the held-out tool set.

**Measure:** goal_success per level, p50/p95 latency on the **target hardware** (state it explicitly in the results file), tokens out, memory footprint.

**Deployment target (owner decision, 2026-08-29):** the planner runs in the
end-user's **browser**. Model download size and in-browser inference are the
binding constraints, not server latency. A 0.6B model (~300+ MB even
quantized) does not fit a web-app download budget, so R2 accuracy alone
cannot kill the custom-model track; the Qwen numbers are the bar to beat,
not a stop condition.

**Decision gate (write the answer in `results/R2.md` before continuing):**

1. Define the targets now: model download budget, in-browser p95 latency (proxy-measured on the dev CPU until a WASM/WebGPU runtime exists — direction set in §0.3), resident memory. Whatever they are, write them down.
2. Condition B's per-level goal_success is the **reference bar**: R3's pass criterion is defined relative to it, and a custom model justifies itself by approaching that accuracy within the browser budget.
3. If some off-the-shelf model meets the browser budget **and** ≥95% on Levels 0–8, then — and only then — stop the custom track and ship it with the harness, effect gate, and PAUSE runtime. Otherwise record the gap (accuracy and/or size/latency) that R3–R6 must close.

**Depends on:** Foundation. Runs in parallel with R1.

---

## 5. R3 — Size curve, with and without constraints

**Question:** Does constrained decoding shift the parameters→success curve left, and how small can the planner be?

**Setup:**
- Dense decoder-only transformers at 5M, 10M, 20M, 40M, 80M, 150M. Same tokenizer, same data, same training budget in tokens.
- Input: normalized request + serialized tool schemas with `T`/`F` symbols. Output: Agent Core.
- Training data: F4, all levels, ≥1M pairs. Include mutation-repair pairs (corrupt a valid program, target is the fixed program) at ~20% of the mix — this is the only piece of the old "Experiment V" that survives, and it belongs here.
- Decoding conditions:
  - **C1** unconstrained
  - **C2** syntax-constrained (grammar only)
  - **C4** syntax + type + tool-schema constrained (only valid tool IDs after `CALL`, only fields present on the register after `FILTER r0`, etc.)
  - Skip C3; it is a checkpoint, not a result.

**Measure:** goal_success by level and by condition; parse/compile rates; tokens out; training convergence (steps to plateau); latency.

**Outputs:**
- `results/R3.md` with the curve `params × condition → success`.
- The best model at each size, checkpointed. These are shared by R4–R8.

**Pass:** The C4 curve is clearly left of C1 (same success at ≥2× fewer parameters), and some model at ≤40M reaches the R2 success level on Levels 0–6.
**Fail action:** If no model ≤150M reaches R2's level under C4, the tiny-planner thesis fails on this data. Report it and stop the custom track.
**Depends on:** R1.

---

## 6. R4 — Do feedback loops buy capacity?

**Question:** Can compiler feedback and execution-in-the-loop substitute for parameters?

**Setup:** Take the best R3 model at two sizes (e.g., 10M and 40M). Train/evaluate four conditions:

| Condition | Description |
|-----------|-------------|
| K0 | Single-shot generation, no feedback |
| K1 | Binary VALID/INVALID feedback, up to 4 repair rounds |
| K2 | Structured diagnostics (the F2 vocabulary) as input, up to 4 repair rounds |
| K3 | K2 + `PAUSE` execution round-trips with register state returned |

Optional fifth arm, only if time allows: shared-block recursion (one block applied 2/4/8 times) on top of K3. Expect little from this; the loop is doing the work.

**Measure:** goal_success, mean and p95 repair rounds, mean and p95 pauses, total latency including round-trips.

**Pass:** K3 lifts the smaller model to within 5 points of the larger model's K0 success on Levels 0–8, at acceptable total latency.
**Depends on:** R3. Independent of R5–R8.

---

## 7. R5 — Tool grounding or memorization?

**Question:** Does the model read schemas, or recognize familiar APIs?

**Setup:**
- Held-out tool families and worlds reserved by F4 and never seen in training.
- Adversarial tool names in the held-out set: `foo17`, `xq`, `operation_93`. Meaning is only in the description and schema.
- Also evaluate the known-tool test set with all tool symbols reshuffled.

**Measure:** goal_success on known vs. held-out vs. adversarial-named tools; tool_valid and arg_valid rates.

**Pass:** held-out and adversarial success within 5 points of known-tool success.
**Fail action:** the generalization claim is dead. The product becomes a per-domain planner trained on each customer's tool set. Say so in the results.
**Depends on:** R3. Independent of R4, R6–R8.

---

## 8. R6 — Where does English break?

**Question:** How narrow does the request distribution have to be for a tiny model to hold up?

**Setup:**
- Adversarial paraphrase suite (tagged by F4): negation, exceptions (`except`, `unless`, `all but`), quantifiers (`at least`, `no more than`, `neither`), temporal (`before`, `most recent`, `second oldest`), and reference resolution.
- Out-of-distribution phrasings: written by a human or a different teacher model, deliberately unlike the generator's style.
- Evaluate every R3 model size on both suites.

**Measure:** goal_success by phenomenon category by model size; confusion cases logged with the emitted program.

**Pass:** There is no pass/fail. The deliverable is an honest boundary: "at 20M, the model is reliable on categories X and Y within the generator's phrasing distribution, and unreliable on Z and on OOD phrasing." This result decides whether the system is described as a **domain planner** or an **agent**.
**Depends on:** R3. Independent of R4, R5, R7, R8.

---

## 9. R7 — Effect-typed safety

**Question:** Does declaring effects prevent destructive mistakes deterministically, and does training with an effect penalty reduce unnecessary destructive calls?

**Setup:**
- Runtime gate (F3) blocks DELETE/SEND/PAY without approval.
- Training variant: add a loss penalty when the emitted program's static effect set contains a destructive effect that the reference does not.
- Evaluate on Level 3 and Level 9 tasks, where scope errors are most likely to cause collateral damage.

**Measure:** unapproved destructive executions (must be zero, by construction); unnecessary_destructive count with vs. without the penalty; goal_success delta.

**Pass:** zero unapproved destructive executions; measurable drop in unnecessary destructive effects with no goal_success regression.
**Depends on:** Foundation for the gate; R3 for the penalty variant. Mostly engineering.

---

## 10. R8 — Contact with reality

**Question:** Does real-agent trace data close the synthetic-to-real gap?

**Setup:**
- Run a strong model through SandwichTS on realistic tasks. Record request, tools, observations, generated JS, results, corrections, final state.
- Convert successful JS traces to Agent Core (write the converter; expect it to be lossy — log what does not convert).
- Fine-tune the best R3/R4 model on synthetic + traces.
- Evaluate on a **held-out set of real tasks**, not synthetic.

**Measure:** goal_success on real tasks before vs. after; synthetic regression check.

**Pass:** meaningful improvement on real tasks with no synthetic regression.
**Depends on:** R3 for the model. Trace collection can start during the foundation phase and should.

---

## 11. Sequence and parallelism

```
Foundation (F1–F5)
   ├── R1  (IR fitness)          ─┐
   └── R2  (off-the-shelf bar)   ─┤  parallel
                                  ▼
                        R2 decision gate
                       (stop or continue)
                                  ▼
                        R3 (size curve)
                                  ▼
        ┌──────────┬──────────┬──────────┬──────────┐
        R4         R5         R6         R7         R8
     feedback   grounding   English    safety     traces
        └──────────┴──────────┴──────────┴──────────┘
                     all independent of each other
```

- R7's runtime gate and R8's trace collection begin during the foundation phase.
- Each result gets its own `results/Rn.md` with: question, setup as actually run, numbers, pass/fail, and what changed in the plan because of it.

---

## 12. Deferred (revisit only if R3+R4 produce a model worth optimizing)

Quantization (INT8/Q4 per component) · input/output vocabulary size · context length and symbolic state compression · shared-block recursion beyond the R4 arm · adaptive recursion depth · Mixture of Widths · attention sparsity · diffusion-style editing · modular reasoning blocks · shared state contracts · architecture router · Neural IR.

---

## 13. Repository layout

```
covenant-agent/
  spec/
    agent_core.md            # F1 grammar, types, effects, registers
    examples/                # hand-written reference programs
  core/
    parser.py  typecheck.py  effects.py  compile.py  diagnostics.py
  runtime/
    sandbox.js               # Node sandbox, effect gate, PAUSE protocol
    worlds/                  # kanban.py crm.py projects.py
    tools/                   # tool schemas + effects + in-memory impls
  harness/
    run.py  metrics.py  report.py
  data/
    gen/                     # F4 backtranslation generator
    holdout/                 # reserved tools/worlds, never in training
  models/
    tiny/                    # R3 dense models
    constrained_decode.py    # C2/C4 decoders
    feedback_loop.py         # R4 repair + PAUSE driver
  baselines/
    qwen/                    # R2
  results/
    R1.md … R8.md
  tests/
```

---

## 14. Working conventions for Claude Code

- Read `spec/agent_core.md` before touching `core/` or `data/`.
- Every change to the IR grammar must update the spec, the parser, the typechecker, the compiler, the generator, and `spec/examples/` in the same PR, with round-trip tests passing.
- The harness is the source of truth for success. Do not report accuracy from any other path.
- Log every experiment run as JSONL with seed, git SHA, data version, model config, and decoding condition.
- When a result fails its pass bar, write the failure in `results/Rn.md` and stop. Do not quietly widen the bar.
- Do not add primitives to the IR, new model architectures, or new experiments without a note in this document explaining which result motivated it.
- Prefer boring choices: standard transformer blocks, standard tokenizer, standard optimizers. Novelty budget is spent on the IR, constrained decoding, and the feedback loop — nowhere else.
