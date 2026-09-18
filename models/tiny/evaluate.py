"""Score a trained model with covenant-agent's own harness.

Nothing here judges a program by comparing it to the reference text. The
program is compiled and executed in the sandbox, and the world state afterwards
is compared to the expected state. That is the same bar the real planner is held
to, and it means a program that reaches the goal a different way still counts.

Two modes, because the sandbox needs Node and a GPU pod does not:

  --generate   run the model, write programs to a JSONL. Needs torch only.
  --score      read that JSONL, execute each program. Needs Node and covenant-agent.

Running both at once is the normal path on a laptop.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from model import CanvasModel, Config
from sample import Trace, ar_sample, diffusion_sample, repair, to_text
from tok import OutVocab


def load_model(ckpt_path: Path, device, ov=None) -> CanvasModel:
    ck = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ck["cfg"])
    m = CanvasModel(cfg)
    if cfg.pointer and ov is not None:
        import re as _re
        m.set_symbol_ids([i for i, t in enumerate(ov.itos)
                          if _re.fullmatch(r"[TFCSNBDI]\d+", t)])
    m = m.to(device)
    # symbol_ids is derived from the vocabulary, not learned. Some checkpoints
    # were saved while it was still persistent, so drop it rather than relaxing
    # strictness -- every genuinely learned weight must still be accounted for.
    sd = {k: v for k, v in ck["model"].items() if k != "symbol_ids"}
    m.load_state_dict(sd)
    m.eval()
    return m


def generate(args):
    device = torch.device(args.device)
    cache = Path(args.cache)
    ov = OutVocab.load(cache / "out_vocab.json")
    model = load_model(Path(args.ckpt), device, ov)
    d = torch.load(cache / f"{args.split}.pt")
    meta = json.loads((cache / f"{args.split}_meta.json").read_text(encoding="utf-8"))

    n = min(args.limit or len(meta), len(meta))
    arm = "ar" if model.c.causal else "diffusion"

    # The compiler loop needs covenant-agent. Without it, generate plain and
    # let --score do the compiling later.
    build_fn = ctxs = None
    if args.repair_rounds and arm == "diffusion":
        from core.pipeline import build as build_fn          # noqa: F401
        from harness.context import TaskContext
        rows = pickle.load(open(cache / "rows.pkl", "rb"))[args.split]
        ctxs = [TaskContext.from_json(r["context"]) for r in rows]

    out = []
    t0 = time.time()
    for i in range(n):
        src = d["src"][i:i + 1].to(device)
        pad = d["pad"][i:i + 1].to(device)
        sym = d["sym"][i:i + 1].to(device) if "sym" in d and model.c.pointer else None
        tr = Trace()
        if arm == "diffusion":
            canvas, tr = diffusion_sample(model, src, pad, ov, steps=args.steps,
                                          temperature=args.temperature, trace=tr, sym=sym)
            if build_fn is not None:
                canvas, tr = repair(model, src, pad, ov, canvas, build_fn, ctxs[i],
                                    rounds=args.repair_rounds, steps=args.repair_steps,
                                    trace=tr, sym=sym)
        else:
            canvas, tr = ar_sample(model, src, pad, ov, trace=tr, sym=sym)
        out.append({
            **meta[i],
            "program": to_text(canvas, ov),
            "reference": to_text(d["tgt"][i:i + 1], ov),
            "passes": tr.passes, "steps": tr.steps, "repairs": tr.repairs,
            "unmask_step": tr.unmask_step,
            "canvas": canvas[0].tolist(),
            "compiled_inline": tr.compiled,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n}  {(time.time()-t0)/(i+1):.2f}s/example", flush=True)

    dest = Path(args.gen_out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} programs -> {dest}")


def score(args):
    """Execute each generated program against its task. Needs Node."""
    from sandbox import register_themes
    register_themes()
    from core.pipeline import build
    from harness.context import TaskContext
    from harness.run import run_task

    cache = Path(args.cache)
    rows = pickle.load(open(cache / "rows.pkl", "rb"))[args.split]
    by_id = {r["id"]: r for r in rows}
    gen = [json.loads(l) for l in open(args.gen_out, encoding="utf-8")]

    stats = Counter()
    by_level = defaultdict(Counter)
    passes = []
    results = []
    for g in gen:
        row = by_id.get(g["task_id"])
        if row is None:
            continue
        ctx = TaskContext.from_json(row["context"])
        res = build(g["program"], ctx)
        stats["n"] += 1
        stats["parse"] += bool(res.parse_ok)
        stats["compile"] += bool(res.compile_ok)
        stats["exact"] += g["program"].strip() == g["reference"].strip()
        passes.append(g["passes"])
        lvl = g.get("level", -1)
        by_level[lvl]["n"] += 1
        by_level[lvl]["compile"] += bool(res.compile_ok)

        goal = False
        if res.compile_ok:
            try:
                m = run_task(row, lambda *a, **k: g["program"])
                goal = bool(m.get("goal_success"))
            except Exception as exc:                      # a sandbox failure is not a model result
                stats["sandbox_error"] += 1
                results.append({"task_id": g["task_id"], "error": str(exc)[:200]})
        stats["goal"] += goal
        by_level[lvl]["goal"] += goal
        results.append({"task_id": g["task_id"], "level": lvl,
                        "compile": bool(res.compile_ok), "goal": goal,
                        "passes": g["passes"],
                        "diagnostics": res.rendered_diagnostics() if not res.compile_ok else []})

    n = max(stats["n"], 1)
    print(f"\n{args.gen_out}  ({stats['n']} tasks)")
    print(f"  parse    {stats['parse']/n:6.1%}")
    print(f"  compile  {stats['compile']/n:6.1%}")
    print(f"  goal     {stats['goal']/n:6.1%}")
    print(f"  exact    {stats['exact']/n:6.1%}")
    if stats["sandbox_error"]:
        print(f"  sandbox errors {stats['sandbox_error']} (not model failures)")
    print(f"  passes   mean {sum(passes)/max(len(passes),1):.1f}")
    print("\n  by level:")
    for lvl in sorted(by_level):
        c = by_level[lvl]
        print(f"    {lvl:3d}  n={c['n']:4d}  compile {c['compile']/c['n']:5.0%}  "
              f"goal {c['goal']/c['n']:5.0%}")

    summary = {"gen": str(args.gen_out), "split": args.split,
               "n": stats["n"], "parse": stats["parse"], "compile": stats["compile"],
               "goal": stats["goal"], "exact": stats["exact"],
               "sandbox_error": stats["sandbox_error"],
               "mean_passes": sum(passes) / max(len(passes), 1),
               "by_level": {str(k): dict(v) for k, v in by_level.items()}}
    dest = Path(args.gen_out).with_suffix(".score.json")
    dest.write_text(json.dumps({"summary": summary, "rows": results}, indent=1),
                    encoding="utf-8")
    print(f"\nwrote {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repair-rounds", type=int, default=0)
    ap.add_argument("--repair-steps", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen-out", default="runs/gen.jsonl")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not (args.generate or args.score):
        args.generate = args.score = True
    if args.generate:
        if not args.ckpt:
            raise SystemExit("--generate needs --ckpt")
        generate(args)
    if args.score:
        score(args)


if __name__ == "__main__":
    main()
