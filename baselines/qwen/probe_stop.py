"""Manual verification for agent_core.gbnf's terminator rule (no unit test is
possible: llama-cpp-python has no accept-string API without a loaded model,
and model files are gitignored).

Builds the kanban-ui demo prompt for a free-typed request (same code path
as POST /kanban_prompt), runs the gguf with and without the grammar, and
prints the next-token distribution right after the program's STOP with and
without a trailing newline.

  python -m baselines.qwen.probe_stop [--model PATH] [--grammar PATH] \
      ["delete all the cards owned by bob" ...]

Pass condition: every grammar run ends with finish=stop and the same text
as its unconstrained run, and P(<|im_end|>) after "STOP" is ~1.0. See
.claude/plans/grammar-terminator-newline.md for the failure this catches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "client" / "poc" / "server"))

from baselines.qwen.run_a import SYSTEM, build_prompt  # noqa: E402
from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from dev_server import handle_kanban_prompt  # noqa: E402

NOW = 1_760_000_000  # client/kanban-ui/src/data/board.ts
DAY = 86400
USERS = [("user_1", "Bob Alvarez"), ("user_2", "Priya Nandan"),
         ("user_3", "Theo Marsh"), ("user_4", "Kade Whitfield"),
         ("user_5", "Luz Ferreira")]
CARDS = [  # id, title, status, assignee, due (days from now), urgent
    ("card_1", "Fix OAuth redirect loop on staging", "doing", "user_1", -3, True),
    ("card_2", "Write release notes for v4.2", "todo", "user_2", 6, False),
    ("card_3", "Audit tool-call pause UX for DELETE effects", "doing", "user_3", 2, True),
    ("card_4", "Retire legacy webhook handler", "todo", "user_1", -8, False),
    ("card_5", "Design empty-state illustration for board", "todo", "user_2", 13, False),
    ("card_6", "Reproduce race condition in segment resume", "doing", "user_4", 1, True),
    ("card_7", "Onboard Luz to on-call rotation", "todo", "user_5", 8, False),
    ("card_8", "Ship grammar-constrained decode benchmarks", "done", "user_4", -6, False),
]
DEFAULT_REQUESTS = [
    "delete all the cards owned by bob",
    "message bob about card 4",
    "archive everything that is done",
]


def demo_state() -> dict:
    return {"entities": {
        "user": [{"id": i, "name": n, "email": f"{n.split()[0].lower()}@understory.test"}
                 for i, n in USERS],
        "card": [{"id": i, "title": t, "status": s, "assignee": a,
                  "due": NOW + d * DAY, "urgent": u, "archived": False,
                  "created": NOW - 20 * DAY}
                 for i, t, s, a, d, u in CARDS]},
        "outbox": [], "payments": []}


def chat_prompt(user: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def top_next(llm, text: str, k: int = 6) -> list:
    toks = llm.tokenize(text.encode(), add_bos=False, special=True)
    llm.reset()
    llm.eval(toks)
    logits = np.array(llm.scores[llm.n_tokens - 1], dtype=np.float64)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return [(llm.detokenize([int(i)], special=True).decode(errors="replace"),
             round(float(p[i]), 3)) for i in np.argsort(-p)[:k]]


def main() -> None:
    ap = argparse.ArgumentParser(prog="baselines.qwen.probe_stop")
    ap.add_argument("--model", default="baselines/qwen/models/qwen3.5-0.8b-s1-q8.gguf")
    ap.add_argument("--grammar", default="baselines/qwen/agent_core.gbnf")
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("requests", nargs="*", default=DEFAULT_REQUESTS)
    args = ap.parse_args()

    from llama_cpp import Llama, LlamaGrammar
    grammar = LlamaGrammar.from_string(Path(args.grammar).read_text(), verbose=False)
    llm = Llama(model_path=args.model, n_ctx=4096, verbose=False, logits_all=True)

    ok = True
    for req in args.requests:
        kp = handle_kanban_prompt({"request": req, "state": demo_state()})
        ctx = TaskContext.from_json(kp["context"])
        prompt = chat_prompt(build_prompt(kp["input_text"], None, []))
        print(f"=== {req}")
        texts = {}
        for g, label in ((None, "unconstrained"), (grammar, "grammar")):
            res = llm.create_completion(prompt, grammar=g, temperature=0.0,
                                        max_tokens=args.max_tokens, stop=["<|im_end|>"])
            ch = res["choices"][0]
            texts[label] = ch["text"].strip()
            r = build(ch["text"], ctx)
            print(f"  {label:13s} tokens_out={res['usage']['completion_tokens']:3d} "
                  f"finish={ch['finish_reason']:6s} compile_ok={r.compile_ok} "
                  f"diags={len(r.diagnostics)}")
        if texts["grammar"] != texts["unconstrained"]:
            ok = False
            print("  !! grammar output differs from unconstrained output")
        head = prompt + texts["unconstrained"]
        print("  next after STOP   :", top_next(llm, head))
        print("  next after STOP\\n :", top_next(llm, head + "\n"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
