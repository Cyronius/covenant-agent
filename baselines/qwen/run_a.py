"""R2 Condition A: off-the-shelf Qwen, in-context prompt, GBNF-constrained.

The model is wired in as a harness Planner, so scoring, PAUSE round-trips,
and latency come from harness.run.run_task — the same source of truth as R1.

  python -m baselines.qwen.run_a --model baselines/qwen/models/Qwen3.5-0.8B-Q8_0.gguf \
      --tasks data/r1_tasks.jsonl --out results/r2_a_0.8b.jsonl [--n 200] [--no-grammar]

--no-grammar gives the unconstrained arm for the same prompt (C1-style
reference point). Metrics rows are harness rows + tokens_out + gen config.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from harness.run import load_tasks, run_task  # noqa: E402
from harness import task_grammar  # noqa: E402
from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from core.ir import Abort  # noqa: E402
from harness.abort_check import check_abort  # noqa: E402

# PLAN.md §6 K2: the checker's structured diagnostics go back to the model as
# input, up to N repair rounds. The rendered forms are a stable contract
# (core/diagnostics.py) but they are terse and unlabelled — TYPE_ERROR prints
# expected then got with nothing saying which is which — so the legend is part
# of the prompt.
REPAIR_LEGEND = """\
The program does not compile. The checker reports:

%s

How to read these:
  TYPE_ERROR line:N A B   position on line N requires type A, you supplied B
  UNBOUND rN              rN is read before anything binds it; bind it with
                          `-> rN` on an earlier line (usually a CALL)
  UNKNOWN_FIELD rN FN     FN is not a field of the entity rN holds
  MISSING_ARG TN FN       tool TN requires parameter FN; supply it positionally
  UNKNOWN_TOOL TN         no such tool symbol in this task
  UNREACHABLE N           line N follows a terminator and can never run
  PARSE_ERROR line:N ...  line N is not valid Agent Core
  ABORT_UNFOUNDED R S ... your ABORT R S claimed something the task
                          contradicts (the detail says what); act instead,
                          or abort with a referent that holds

Rewrite the WHOLE program with these fixed, using only the T/F/C symbols
listed for the task. Line numbers are 1-based. Output ONLY the program."""


def repair_prompt(rendered: list) -> str:
    return REPAIR_LEGEND % "\n".join("  " + d for d in rendered[:8])


class GrammarCache:
    """Per-task GBNF, compiled once per distinct symbol table.

    `task` mode enumerates the T/F/C symbols the prompt declares; `typed`
    adds per-tool CALL rules whose slots admit only type-compatible symbols
    (PLAN.md §5 condition C4); `static` is the old single grammar, whose
    two-digit `num` made every symbol above 99 undecodable (results/S2.md
    §S3). `none` is the unconstrained arm.
    """

    def __init__(self, mode: str, base_path: Path):
        self.mode = mode
        self.base = base_path.read_text() if mode != "none" else None
        self._cache: dict = {}
        self._static = None

    @property
    def condition(self) -> str:
        return {"task": "A-grammar-task", "typed": "A-grammar-typed",
                "static": "A-grammar", "none": "A-unconstrained"}[self.mode]

    def for_task(self, task: dict):
        if self.mode == "none":
            return None
        from llama_cpp import LlamaGrammar
        if self.mode == "static":
            if self._static is None:
                self._static = LlamaGrammar.from_string(self.base,
                                                        verbose=False)
            return self._static
        typed = self.mode == "typed"
        key = (task_grammar.typed_signature(task) if typed
               else task_grammar.symbol_signature(task))
        if key not in self._cache:
            self._cache[key] = LlamaGrammar.from_string(
                task_grammar.grammar_for_task(task, self.base, typed=typed),
                verbose=False)
        return self._cache[key]

SYSTEM = """You translate task requests into Agent Core programs.

Agent Core instructions (one per line, two-space indent for block bodies):
CALL Tn args -> r        call tool Tn (args: registers, rX.Fn, Cn, NOW)
FILTER r pred -> r       keep list elements matching pred, e.g. F3 EQ C0 AND NOT F5 LT NOW
SORT r Fn ASC|DESC -> r  sort list by field
SELECT r i -> r          i-th element (0-based); FIRST r -> r; COUNT r -> r; MAP r Fn -> r
GET r.Fn -> r            extract field
FORMAT Ct args -> r      fill template constant Ct (slots {0} {1} ...) with values -> STR
LET x -> r               bind value
FOREACH r -> rElem       loop over list, body indented below
IF cond / ELSE           branch, bodies indented; cond compares operands, e.g. r0 EQ C1
PARALLEL                 body: CALL lines only, run concurrently
TRY [RETRY n] -> r       run body, catch tool errors; r gets OK or error code
STOP | RETURN x | PAUSE  end program (PAUSE = report back; a continuation follows later)
ABORT reason sym         decline without acting, naming what it is about:
                         NOT_FOUND Cn (nothing matches Cn) | NEEDS_INFO Fn (no value
                         for Fn) | AMBIGUOUS a b (cannot choose between) | UNSUPPORTED [Cn]

Rules: registers r0-r15 in order of first use. Use ONLY the T/F/C symbols
listed for the task; every literal value must be a C symbol. TIME fields are
timestamps: "more than N days ago" / "overdue" means Fx LT (cutoff/NOW).
Programs are SHORT — typically 2 to 8 lines — and always end with STOP
(or PAUSE when the request says to report back before acting). If the
request names something with no matching symbol, needs a tool that is
not listed, is missing required values, or could mean several things,
ABORT with the reason instead of guessing.

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

Output ONLY the program, nothing else."""


def build_prompt(task_input: str, registers: dict | None,
                 prior: list[str]) -> str:
    parts = [task_input.strip()]
    if prior:
        parts.append("PROGRAM SO FAR (already executed, ended at PAUSE):")
        parts += [p.strip() for p in prior]
        parts.append("REGISTERS NOW BOUND (values from the run):")
        parts.append(json.dumps(registers, default=str)[:1500])
        parts.append("Write ONLY the continuation program (registers above "
                     "are still bound; do not re-fetch).")
    parts.append("PROGRAM:")
    return "\n".join(parts)


def generate(llm, grammar, user: str, max_tokens: int = 250,
             template: str = "qwen", shots: list | None = None) -> dict:
    """One greedy, grammar-constrained completion. Returns
    {text, usage, finish_reason}.

    `shots` are worked examples as [(user, program), ...], prepended as prior
    chat turns. The SYSTEM prompt carries one example and was written against
    Qwen; an untuned model that fails on it may be failing on the format
    rather than the task, and shots separate the two. LFM2.5-8B-A1B one-shot
    never binds a register (`FILTER r0 …` with nothing bound, to the token
    cap); at four shots it writes the reference skeleton and misses on symbol
    choice instead.

    Templates are a per-model choice, not a global one:
      qwen  hand-rolled ChatML with an explicit empty think block — keeps the
            model in no-think mode (the grammar forbids <think> anyway, which
            would otherwise force "reasoning" in program syntax). Our tuned
            S1/S2 checkpoints were SFT'd against exactly this markup, so
            changing it changes the distribution they were measured on.
      chat  create_chat_completion, which applies the GGUF's own
            tokenizer.chat_template — the path for any other instruct model
            (harness/rpg_suite.py's cross-model comparison).
    """
    shots = shots or []
    if template == "chat":
        messages = [{"role": "system", "content": SYSTEM}]
        for shot_user, shot_program in shots:
            messages.append({"role": "user", "content": shot_user})
            messages.append({"role": "assistant", "content": shot_program})
        messages.append({"role": "user", "content": user})
        res = llm.create_chat_completion(
            messages=messages,
            grammar=grammar, temperature=0.0, max_tokens=max_tokens)
        choice = res["choices"][0]
        text = (choice["message"].get("content") or "").strip()
    else:
        turns = "".join(f"<|im_start|>user\n{u}<|im_end|>\n"
                        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        f"{p}<|im_end|>\n" for u, p in shots)
        prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n{turns}"
                  f"<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        res = llm.create_completion(
            prompt, grammar=grammar, temperature=0.0,
            max_tokens=max_tokens, stop=["<|im_end|>"])
        choice = res["choices"][0]
        text = choice["text"].strip()
    return {"text": text, "usage": res.get("usage", {}),
            "finish_reason": choice.get("finish_reason")}


def load_shots(path: Path, n: int, exclude: set) -> list:
    """[(user prompt, program)] worked examples for `generate(shots=...)`."""
    out = []
    if not n:
        return out
    for line in open(path):
        t = json.loads(line)
        if t["id"] in exclude or t["level"] < 2:
            continue
        out.append((build_prompt(t["input_text"], None, []),
                    "".join(t["reference"]["segments"]).strip()))
        if len(out) >= n:
            break
    return out


def make_planner(llm, grammar, task, max_tokens, usage_sink, template="qwen",
                 shots=None, repair=0, repair_sink=None):
    prior: list[str] = []

    def plan(request, ctx, seg_idx, registers):
        if seg_idx > 2:   # runaway guard: give up after 3 segments
            return None
        user = build_prompt(task["input_text"], registers, prior)
        res = generate(llm, grammar, user, max_tokens, template, shots)
        usage_sink.append(res["usage"])
        text = res["text"]

        # K2 repair rounds. Static diagnostics only: run_task owns execution,
        # and every failure worth repairing so far is a compile-time one.
        rounds = 0
        while rounds < repair:
            result = build(text, ctx)
            feedback = None
            if not result.compile_ok:
                feedback = result.rendered_diagnostics()
            elif (result.program.body
                  and isinstance(result.program.body[0], Abort)):
                # a first-line abort is a static claim about the task; check
                # its referent (spec §4). Aborts inside IF are runtime and
                # belong to run_task.
                a = result.program.body[0]
                msg = check_abort(ctx, task["state"], a.reason, a.refs)
                if msg:
                    feedback = [msg]
            if feedback is None:
                break
            rounds += 1
            res = generate(llm, grammar, repair_prompt(feedback),
                           max_tokens, template,
                           (shots or []) + [(user, text)])
            usage_sink.append(res["usage"])
            text = res["text"]
        if repair_sink is not None:
            repair_sink.append(rounds)

        prior.append(text)
        return text

    return plan


def main():
    ap = argparse.ArgumentParser(prog="baselines.qwen.run_a")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="first N tasks only (subset run)")
    ap.add_argument("--grammar", default="baselines/qwen/agent_core.gbnf")
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--repair", type=int, default=0, metavar="N",
                    help="up to N repair rounds per segment, feeding the "
                         "checker's diagnostics back as input (PLAN.md §6 "
                         "condition K2). 0 = single-shot (K0)")
    ap.add_argument("--shots", type=int, default=0, metavar="N",
                    help="prepend N worked examples as prior chat turns "
                         "(untuned arms only — a tuned checkpoint was SFT'd "
                         "on the zero-shot markup)")
    ap.add_argument("--shots-from", default="data/r1_tasks.jsonl")
    ap.add_argument("--grammar-mode", choices=["task", "typed", "static"],
                    default="task",
                    help="task: rebuild the grammar per task from the symbols "
                         "that task declares (default). typed: task plus "
                         "per-tool CALL rules with type-checked slots (C4). "
                         "static: the base file as-is — pre-2026-09-06 "
                         "behaviour, kept only to reproduce older runs")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--domains", default=None, metavar="DIR",
                    help="register generated domain themes (S0 suites)")
    ap.add_argument("--lora", default=None, metavar="GGUF",
                    help="attach a GGUF LoRA adapter at runtime instead of "
                         "using a merged checkpoint (plan "
                         "writer-adapter-experiment, E1)")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--template", choices=["qwen", "chat"], default="qwen",
                    help="qwen: hand-rolled ChatML + empty think block (the S1-S3 "
                         "markup); chat: the GGUF's own chat_template via "
                         "create_chat_completion (LFM2.5, any other instruct model)")
    args = ap.parse_args()
    if args.domains:
        from data.gen.domains import register_domains
        register_domains(args.domains)

    from llama_cpp import Llama
    grammars = GrammarCache("none" if args.no_grammar else args.grammar_mode,
                            Path(args.grammar))
    llm = Llama(model_path=args.model, n_ctx=args.ctx,
                n_threads=args.threads, verbose=False,
                lora_path=args.lora, lora_scale=args.lora_scale)

    tasks = load_tasks(Path(args.tasks))
    if args.n:
        tasks = tasks[:args.n]
    shots = load_shots(ROOT / args.shots_from, args.shots,
                       {t["id"] for t in tasks})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()
    with open(out, "w") as f:
        for i, task in enumerate(tasks):
            usage: list = []
            repairs: list = []
            row = run_task(task, make_planner(llm, grammars.for_task(task),
                                              task, args.max_tokens, usage,
                                              template=args.template,
                                              shots=shots, repair=args.repair,
                                              repair_sink=repairs))
            row["tokens_out"] = sum(u.get("completion_tokens", 0)
                                    for u in usage)
            row["tokens_in"] = sum(u.get("prompt_tokens", 0) for u in usage)
            row["condition"] = grammars.condition + (
                f"+K2repair{args.repair}" if args.repair else "")
            row["repair_rounds"] = sum(repairs)
            row["model"] = Path(args.model).name
            row["template"] = args.template
            if args.lora:
                row["lora"] = Path(args.lora).name
                row["lora_scale"] = args.lora_scale
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            if (i + 1) % 20 == 0:
                g = sum(r["goal_success"] for r in rows)
                c = sum(r["compile_ok"] for r in rows)
                el = time.time() - t_start
                print(f"{i+1}/{len(tasks)} goal={g} compile={c} "
                      f"({el/ (i+1):.1f}s/task)", flush=True)

    done = rows
    summary = {
        "model": Path(args.model).name,
        "lora": Path(args.lora).name if args.lora else None,
        "lora_scale": args.lora_scale if args.lora else None,
        "condition": rows[0]["condition"] if rows else None,
        "n": len(done),
        "compile_ok_rate": sum(r["compile_ok"] for r in done) / len(done),
        "goal_success_rate": sum(r["goal_success"] for r in done) / len(done),
        "parse_ok_rate": sum(r["parse_ok"] for r in done) / len(done),
        "by_level": {},
        "gen_ms_p50": statistics.median(
            r["latency_generate_ms"] for r in done),
        "gen_ms_p95": sorted(r["latency_generate_ms"]
                             for r in done)[int(0.95 * len(done))],
        "tokens_out_p50": statistics.median(r["tokens_out"] for r in done),
    }
    for r in done:
        b = summary["by_level"].setdefault(r["level"],
                                           {"n": 0, "compile": 0, "goal": 0})
        b["n"] += 1
        b["compile"] += r["compile_ok"]
        b["goal"] += r["goal_success"]
    summary["by_level"] = {k: summary["by_level"][k]
                           for k in sorted(summary["by_level"])}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
