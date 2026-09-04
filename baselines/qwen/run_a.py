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
from core.ir import TaskContext  # noqa: E402

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
ABORT reason             decline without acting: NOT_FOUND | AMBIGUOUS | UNSUPPORTED | NEEDS_INFO

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
             template: str = "qwen") -> dict:
    """One greedy, grammar-constrained completion. Returns
    {text, usage, finish_reason}.

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
    if template == "chat":
        res = llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            grammar=grammar, temperature=0.0, max_tokens=max_tokens)
        choice = res["choices"][0]
        text = (choice["message"].get("content") or "").strip()
    else:
        prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        res = llm.create_completion(
            prompt, grammar=grammar, temperature=0.0,
            max_tokens=max_tokens, stop=["<|im_end|>"])
        choice = res["choices"][0]
        text = choice["text"].strip()
    return {"text": text, "usage": res.get("usage", {}),
            "finish_reason": choice.get("finish_reason")}


def make_planner(llm, grammar, task, max_tokens, usage_sink, template="qwen"):
    prior: list[str] = []

    def plan(request, ctx, seg_idx, registers):
        if seg_idx > 2:   # runaway guard: give up after 3 segments
            return None
        user = build_prompt(task["input_text"], registers, prior)
        res = generate(llm, grammar, user, max_tokens, template)
        usage_sink.append(res["usage"])
        prior.append(res["text"])
        return res["text"]

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
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--domains", default=None, metavar="DIR",
                    help="register generated domain themes (S0 suites)")
    args = ap.parse_args()
    if args.domains:
        from data.gen.domains import register_domains
        register_domains(args.domains)

    from llama_cpp import Llama, LlamaGrammar
    grammar = None
    if not args.no_grammar:
        grammar = LlamaGrammar.from_string(
            Path(args.grammar).read_text(), verbose=False)
    llm = Llama(model_path=args.model, n_ctx=args.ctx,
                n_threads=args.threads, verbose=False)

    tasks = load_tasks(Path(args.tasks))
    if args.n:
        tasks = tasks[:args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()
    with open(out, "w") as f:
        for i, task in enumerate(tasks):
            usage: list = []
            row = run_task(task, make_planner(llm, grammar, task,
                                              args.max_tokens, usage))
            row["tokens_out"] = sum(u.get("completion_tokens", 0)
                                    for u in usage)
            row["tokens_in"] = sum(u.get("prompt_tokens", 0) for u in usage)
            row["condition"] = ("A-grammar" if grammar is not None
                                else "A-unconstrained")
            row["model"] = Path(args.model).name
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
