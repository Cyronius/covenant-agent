"""Score a trained model with covenant-agent's own harness.

Nothing here judges a program by comparing it to the reference text. The
program is compiled and executed in the sandbox, and the world state afterwards
is compared to the expected state. That is the same bar the real planner is held
to, and it means a program that reaches the goal a different way still counts.

Two modes, because the sandbox needs Node and a GPU pod does not:

  --generate   run the model, write programs to a JSONL. Needs torch only.
  --score      read that JSONL, execute each program. Needs Node and covenant-agent.

Running both at once is the normal path on a laptop.

Beside goal, compile and parse, `--score` reports two grounding measurements:
per-slot-kind accuracy of the generated canvas against the reference canvas
(same slot index; keyword, tool, field, constant, register), and the
line-aligned CALL agreement `diagnose.py` computes: on lines where both the
reference and the generation CALL, the same tool, and the same effect class,
each against exact chance.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from model import Config, build_model
from prep import STRUCT_KEYS
from sample import Trace, ar_sample, diffusion_sample, repair, to_text
from tok import OutVocab


def load_model(ckpt_path: Path, device, ov=None):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ck["cfg"])
    m = build_model(cfg)
    if cfg.binding == "flat" and cfg.pointer and ov is not None:
        m.set_symbol_ids([i for i, t in enumerate(ov.itos)
                          if re.fullmatch(r"[TFCSNBDI]\d+", t)])
    m = m.to(device)
    # symbol_ids is derived from the vocabulary, not learned. Some checkpoints
    # were saved while it was still persistent, so drop it rather than relaxing
    # strictness -- every genuinely learned weight must still be accounted for.
    sd = {k: v for k, v in ck["model"].items() if k != "symbol_ids"}
    m.load_state_dict(sd)
    m.eval()
    return m


class Split:
    """One cached split, and the per-example inputs and codec for either binding."""

    def __init__(self, cache: Path, name: str, device):
        self.cache, self.name, self.device = cache, name, device
        self.meta_cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
        self.binding = self.meta_cfg.get("binding", "flat")
        self.d = torch.load(cache / f"{name}.pt")
        self.meta = json.loads((cache / f"{name}_meta.json").read_text(encoding="utf-8"))
        if self.binding == "flat":
            self.ov = OutVocab.load(cache / "out_vocab.json")
        else:
            from canvas import Layout, load_keywords
            self.keywords = load_keywords(cache / "keywords.json")
            self.layout = Layout.from_dict(self.meta_cfg["layout"])

    def __len__(self) -> int:
        return len(self.meta)

    def codec(self, i: int):
        if self.binding == "flat":
            return self.ov
        from canvas import TaskCodec
        return TaskCodec(self.keywords, self.layout, **self.meta[i]["syms"])

    def inputs(self, i: int, model) -> dict:
        d = self.d
        if self.binding == "flat":
            out = {"src": d["src"][i:i + 1].to(self.device), "pad": d["pad"][i:i + 1].to(self.device)}
            if "sym" in d and getattr(model.c, "pointer", False):
                out["sym"] = d["sym"][i:i + 1].to(self.device)
            return out
        return {k: d[k][i:i + 1].to(self.device) for k in STRUCT_KEYS if k != "tgt"}

    def target(self, i: int) -> torch.Tensor:
        return self.d["tgt"][i:i + 1].long()


def generate(args):
    device = torch.device(args.device)
    cache = Path(args.cache)
    split = Split(cache, args.split, device)
    model = load_model(Path(args.ckpt), device, split.ov if split.binding == "flat" else None)
    if model.c.binding != split.binding:
        raise SystemExit(f"checkpoint binding={model.c.binding} but cache binding={split.binding}")
    if args.dec_loops:
        # The loop count is an inference-time dial (decision 9), not a shape, so
        # it can be turned after training. A model trained with --rand-loops has
        # seen every setting; one trained at a fixed L has not, and running it
        # somewhere else is a measurement of how far the dial travels.
        model.c.dec_loops = args.dec_loops
    print(f"loops={model.c.dec_loops} dec_layers={model.c.dec_layers} "
          f"weights={model.c.weights}", flush=True)

    n = min(args.limit or len(split), len(split))
    arm = "ar" if model.c.causal else "diffusion"
    # A level filter, for re-measuring one curriculum level without regenerating
    # the split. The tasks it picks are the same tasks whatever else changes, so
    # two runs over the same level are comparable slot for slot.
    which = [i for i in range(n)
             if args.level is None or split.meta[i].get("level") == args.level]
    if args.level is not None:
        print(f"level {args.level}: {len(which)} of {n} tasks", flush=True)

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
    for k, i in enumerate(which):
        inputs = split.inputs(i, model)
        ov = split.codec(i)
        tr = Trace()
        if arm == "diffusion":
            canvas, tr = diffusion_sample(model, inputs, ov, steps=args.steps,
                                          temperature=args.temperature, trace=tr,
                                          threshold=args.threshold)
            if build_fn is not None:
                canvas, tr = repair(model, inputs, ov, canvas, build_fn, ctxs[i],
                                    rounds=args.repair_rounds, steps=args.repair_steps,
                                    trace=tr)
        else:
            canvas, tr = ar_sample(model, inputs, ov, trace=tr)
        tgt = split.target(i)
        meta = {k: v for k, v in split.meta[i].items() if k != "syms"}
        out.append({
            **meta,
            "program": to_text(canvas, ov),
            "reference": to_text(tgt, ov),
            "passes": tr.passes, "steps": tr.steps, "repairs": tr.repairs,
            "unmask_step": tr.unmask_step,
            "canvas": canvas[0].tolist(),
            # Surface tokens per slot, so probe.py and --score can classify
            # slots without the cache and under either binding.
            "canvas_tokens": ov.decode(canvas[0].tolist()),
            "reference_tokens": ov.decode(tgt[0].tolist()),
            "compiled_inline": tr.compiled,
            # The compute this program cost, for the axis report.py draws:
            # one forward pass applies the block dec_layers x loops times.
            "loops": model.c.dec_loops, "dec_layers": model.c.dec_layers,
        })
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(which)}  {(time.time()-t0)/(k+1):.2f}s/example", flush=True)

    dest = Path(args.gen_out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} programs -> {dest}")


def slot_kind(tok: str) -> str:
    if re.fullmatch(r"T\d+", tok):
        return "tool"
    if re.fullmatch(r"F\d+", tok):
        return "field"
    if re.fullmatch(r"[CSNBDI]\d+", tok):
        return "const"
    if re.fullmatch(r"r\d+\.?", tok):
        return "reg"
    return "kw"


def slot_accuracy(gen: list[str], ref: list[str], acc: dict) -> None:
    """Per-kind agreement at the same slot index, over the reference's
    non-PAD slots. A misaligned program scores low here even when it is
    right; the line-aligned CALL agreement below is the complement."""
    for g, r in zip(gen, ref):
        if r == "PAD":
            break
        k = slot_kind(r)
        acc[k]["n"] += 1
        acc[k]["hit"] += g == r


def score(args):
    """Execute each generated program against its task. Needs Node."""
    from sandbox import register_themes
    register_themes()
    from core.pipeline import build
    from diagnose import compare_calls, parse as parse_context
    from harness.context import TaskContext, serialize_context
    from harness.run import run_task

    cache = Path(args.cache)
    rows = pickle.load(open(cache / "rows.pkl", "rb"))[args.split]
    by_id = {r["id"]: r for r in rows}
    gen = [json.loads(l) for l in open(args.gen_out, encoding="utf-8")]

    stats = Counter()
    by_level = defaultdict(Counter)
    slot_acc = defaultdict(Counter)
    calls = Counter()
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
        if "canvas_tokens" in g:
            slot_accuracy(g["canvas_tokens"], g["reference_tokens"], slot_acc)
        tools, _ = parse_context(serialize_context(row["request"], ctx))
        calls.update(compare_calls(g["reference"], g["program"], tools))

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
    if slot_acc:
        print("\n  slot accuracy against the reference canvas (same slot):")
        for k in ("kw", "tool", "field", "const", "reg"):
            c = slot_acc[k]
            if c["n"]:
                print(f"    {k:6s} {c['hit']/c['n']:6.1%}  (n={c['n']})")
    c = max(calls["compared"], 1)
    print(f"\n  CALL lines where both reference and generation CALL: {calls['compared']}")
    print(f"    same tool    {calls['same_tool']/c:6.1%}   chance {calls['chance_tool']/c:6.1%}")
    print(f"    same effect  {calls['same_effect']/c:6.1%}   chance {calls['chance_effect']/c:6.1%}")
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
               "loops": gen[0].get("loops", 1) if gen else 1,
               "dec_layers": gen[0].get("dec_layers") if gen else None,
               "slot_acc": {k: dict(v) for k, v in slot_acc.items()},
               "call_agreement": dict(calls),
               "by_level": {str(k): dict(v) for k, v in by_level.items()}}
    dest = Path(args.gen_out).with_suffix(".score.json")
    dest.write_text(json.dumps({"summary": summary, "rows": results}, indent=1),
                    encoding="utf-8")
    print(f"\nwrote {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cache", default="data_cache_struct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--dec-loops", type=int, default=None,
                    help="run the block this many times per pass, overriding "
                         "the checkpoint's setting")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="commit every slot the model is this confident of, and "
                         "at least one, instead of following the cosine "
                         "schedule; --steps becomes a cap on passes")
    ap.add_argument("--repair-rounds", type=int, default=0)
    ap.add_argument("--repair-steps", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--level", type=int, default=None,
                    help="generate only this curriculum level, for re-measuring "
                         "one level without regenerating the split")
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
