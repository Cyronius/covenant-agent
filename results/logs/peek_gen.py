"""Print what a model actually writes for a few tasks, with diagnostics.

run_a.py stores scores, not programs; when a new model fails, the first
question is always "what did it write". Not a scored path.

  python results/logs/peek_gen.py --model <gguf> --tasks <jsonl> --n 3 \
      [--template chat] [--max-tokens 250]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baselines.qwen.run_a import build_prompt, generate  # noqa: E402
from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness import task_grammar  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tasks", required=True)
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--max-tokens", type=int, default=250)
ap.add_argument("--template", choices=["qwen", "chat"], default="chat")
ap.add_argument("--no-grammar", action="store_true")
ap.add_argument("--shots", type=int, default=0, metavar="N",
                help="prepend N worked examples as prior chat turns, drawn "
                     "from --shots-from. The SYSTEM prompt carries one "
                     "example and was written against Qwen; this asks "
                     "whether a model that fails one-shot fails on the task "
                     "or on the format.")
ap.add_argument("--shots-from", default="data/r1_tasks.jsonl")
args = ap.parse_args()

SHOTS = []
if args.shots:
    seen = {json.loads(l)["id"] for l in open(args.tasks)}
    for line in open(ROOT / args.shots_from):
        t = json.loads(line)
        if t["id"] in seen or t["level"] < 2:
            continue
        SHOTS.append(t)
        if len(SHOTS) >= args.shots:
            break

from llama_cpp import Llama, LlamaGrammar  # noqa: E402

llm = Llama(model_path=args.model, n_ctx=args.ctx, verbose=False)
base = task_grammar.load_base()

tasks = [json.loads(l) for l in open(args.tasks)][:args.n]
for t in tasks:
    grammar = None
    if not args.no_grammar:
        grammar = LlamaGrammar.from_string(
            task_grammar.grammar_for_task(t, base), verbose=False)
    t0 = time.perf_counter()
    if SHOTS:
        from baselines.qwen.run_a import SYSTEM
        msgs = [{"role": "system", "content": SYSTEM}]
        for s in SHOTS:
            msgs.append({"role": "user",
                         "content": build_prompt(s["input_text"], None, [])})
            msgs.append({"role": "assistant",
                         "content": "".join(s["reference"]["segments"]).strip()})
        msgs.append({"role": "user",
                     "content": build_prompt(t["input_text"], None, [])})
        raw = llm.create_chat_completion(messages=msgs, grammar=grammar,
                                         temperature=0.0,
                                         max_tokens=args.max_tokens)
        res = {"text": (raw["choices"][0]["message"].get("content") or "").strip(),
               "finish_reason": raw["choices"][0].get("finish_reason")}
    else:
        res = generate(llm, grammar, build_prompt(t["input_text"], None, []),
                       args.max_tokens, args.template)
    ms = (time.perf_counter() - t0) * 1000
    ctx = TaskContext.from_json(t["context"])
    r = build(res["text"], ctx)
    print("=" * 70)
    print("%s  L%s  %s  %.1fs  finish=%s"
          % (t["id"], t["level"], t["world"], ms / 1000,
             res.get("finish_reason")))
    print("REQUEST:", t["request"])
    print("-- reference " + "-" * 56)
    print("".join(t["reference"]["segments"]).rstrip())
    print("-- generated " + "-" * 56)
    print(res["text"].rstrip())
    print("-- diagnostics:", r.rendered_diagnostics()[:5])
