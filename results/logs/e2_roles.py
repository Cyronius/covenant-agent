"""E2, widened to three roles (.claude/plans/writer-adapter-experiment.md).

The decision number for `baselines/qwen/model_roles.py`: a role switch
clears the KV cache, so the next planner turn loses the shared system-prompt
prefix and re-prefills from zero. What does that cost, against the 775 MB
saved by dropping the second GGUF in server/dev_server.py?

Timed call is the planner with max_tokens=1 -- that is prefill plus one
token, which is exactly what a switch destroys. Generation speed itself is
unaffected by switching, so timing a full 128-token generation would only
add noise. One full generation per condition is kept as a spot check that
the planner still writes programs through this path.

Conditions, same tasks in the same order, one resident model throughout:

  C1  plan -> plan          today's planner instance: prefix stays warm
  C2  plan -> write -> plan adapter off for prose, back on
  C3  plan -> embed -> plan embeddings on, back off
  C0  cold                  explicit reset before each plan (worst case)

Also runs E2's correctness half: adapter-off output vs a dedicated base
instance, which must be identical or the whole idea is unsound.

  python results/logs/e2_roles.py [--n 8] [--out results/logs/e2_roles.jsonl]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baselines.qwen.model_roles import RoleModel  # noqa: E402
from baselines.qwen.run_a import SYSTEM, build_prompt  # noqa: E402

NL = chr(10)
MODELS = ROOT / "baselines" / "qwen" / "models"
BASE = MODELS / "Qwen3.5-0.8B-Q8_0.gguf"
ADAPTER = MODELS / "lora_s2_f16.gguf"
SUITE = ROOT / "data" / "holdout" / "e_demo_requests.jsonl"

WRITER_SYSTEM = ("You write short, plain workplace messages. Output only the "
                 "message text - no greeting line, no sign-off, no markdown.")
WRITE_BRIEF = ("Brief: tell the team which cards are overdue." + NL
               + "Items:" + NL + "- Fix login redirect, status doing" + NL
               + "- Ship invoice export, status todo" + NL + NL
               + "Write the message.")


def chat(system: str, user: str) -> str:
    return ("<|im_start|>system" + NL + system + "<|im_end|>" + NL
            + "<|im_start|>user" + NL + user + "<|im_end|>" + NL
            + "<|im_start|>assistant" + NL + "<think>" + NL + NL
            + "</think>" + NL + NL)


def planner_prompt(task: dict) -> str:
    return chat(SYSTEM, build_prompt(task["input_text"], None, []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="results/logs/e2_roles.jsonl")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(SUITE, encoding="utf-8")][:args.n]
    prompts = [planner_prompt(t) for t in tasks]
    print(f"{len(tasks)} tasks, prompt chars "
          f"{min(len(p) for p in prompts)}-{max(len(p) for p in prompts)}")

    t0 = time.perf_counter()
    m = RoleModel(BASE, ADAPTER, n_ctx=args.ctx, n_threads=args.threads)
    print(f"loaded base+adapter in {time.perf_counter()-t0:.1f}s")

    rows: list[dict] = []

    def timed_plan(task, prompt, cond, rep):
        r = m.plan(prompt, max_tokens=1)
        rows.append({"condition": cond, "rep": rep, "task_id": task["id"],
                     "ms": round(r["ms"], 1), "tokens_in": r["tokens_in"]})
        return r

    # llama.cpp mmaps the GGUF lazily, so the first real forward pass faults
    # ~800 MB in from disk. Measured cold, that lands entirely on whichever
    # condition runs first (40 s vs 6 s on the n=3 smoke) and reads as if
    # that condition were the slow one. Burn it before timing anything.
    t0 = time.perf_counter()
    for prompt in prompts[:2]:
        m.plan(prompt, max_tokens=1)
    print(f"warm-up (page faults in the mmap) {time.perf_counter()-t0:.1f}s")

    # Two passes, interleaved by pass rather than run back to back, so any
    # drift over the run shows up as pass-to-pass spread instead of being
    # silently attributed to a condition.
    for rep in range(args.reps):
        for cond in ("C1", "C2", "C3", "C0"):
            m.llm.reset()
            m.llm._ctx.kv_cache_clear()
            timed_plan(tasks[0], prompts[0], cond + "-prime", rep)
            for task, prompt in zip(tasks[1:], prompts[1:]):
                if cond == "C2":
                    m.write(WRITE_BRIEF, max_tokens=24)
                elif cond == "C3":
                    m.embed("which cards are overdue for the team")
                elif cond == "C0":
                    m.llm.reset()
                    m.llm._ctx.kv_cache_clear()
                timed_plan(task, prompt, cond, rep)
            got = [r["ms"] for r in rows
                   if r["condition"] == cond and r["rep"] == rep]
            print(f"  rep{rep} {cond}: planner prefill p50 "
                  f"{st.median(got):,.0f} ms ({len(got)} calls)")

    # spot check: does the planner still write a program through this path?
    spot = m.plan(prompts[0], max_tokens=96)
    print("spot-check program:", repr(spot["text"][:160]))

    # E2 correctness half: adapter-off must equal a dedicated base instance
    off = m.write(chat(WRITER_SYSTEM, WRITE_BRIEF), max_tokens=48)
    from llama_cpp import Llama
    ref = Llama(model_path=str(BASE), n_ctx=args.ctx, n_threads=args.threads,
                verbose=False)
    ref_out = ref.create_completion(chat(WRITER_SYSTEM, WRITE_BRIEF),
                                    temperature=0.0, max_tokens=48,
                                    stop=["<|im_end|>"])["choices"][0]["text"]
    same = off["text"] == ref_out
    print(f"adapter-off == dedicated base instance: {same}")
    if not same:
        print("  role-model:", repr(off["text"][:200]))
        print("  dedicated :", repr(ref_out[:200]))

    base_p50 = st.median(r["ms"] for r in rows if r["condition"] == "C1")
    summary = {"n_tasks": len(tasks), "ctx": args.ctx,
               "adapter_off_identical": same, "stats": m.stats()}
    for cond in ("C1", "C2", "C3", "C0"):
        ms = [r["ms"] for r in rows if r["condition"] == cond]
        per_rep = [round(st.median([r["ms"] for r in rows
                                    if r["condition"] == cond
                                    and r["rep"] == rep]), 1)
                   for rep in range(args.reps)]
        summary[cond] = {"p50_ms": round(st.median(ms), 1),
                         "delta_vs_C1_ms": round(st.median(ms) - base_p50, 1),
                         "per_rep_p50_ms": per_rep, "n": len(ms)}
    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + NL)
        f.write(json.dumps({"summary": summary}) + NL)
    print(json.dumps(summary, indent=2))
    m.close()


if __name__ == "__main__":
    main()
