"""Convert F4 task JSONL into SFT chat pairs for R2 Condition B.

One example per program segment, using the SAME system prompt and user
format as eval (baselines.qwen.run_a), so train and test distributions
match. Multi-segment (PAUSE) tasks get a continuation example whose user
turn includes the executed segment and the actual register values — which
requires running segment 1 in the sandbox (Node) at conversion time.

  python -m baselines.qwen.make_sft --tasks data/train_50k.jsonl \
      --out data/sft_50k.jsonl [--no-continuations]

Output rows: {"messages": [{role, content} x3]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baselines.qwen.run_a import SYSTEM, build_prompt  # noqa: E402
from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness.context import sandbox_from_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from runtime.worlds import get_world  # noqa: E402


def pause_registers(task: dict, seg0: str) -> dict | None:
    """Execute segment 0 in the sandbox; return registers at PAUSE."""
    ctx = TaskContext.from_json(task["context"])
    result = build(seg0, ctx)
    if not result.compile_ok:
        return None
    world = get_world(task["world"])
    sandbox_ctx = task.get("sandbox") or sandbox_from_context(ctx, world)
    res = run_sandbox({
        "js": result.js, "state": task["state"],
        "tools": sandbox_ctx["tools"], "fields": sandbox_ctx["fields"],
        "constants": sandbox_ctx["constants"], "now": task["now"],
        "approval": task.get("approval", False),
        "error_injection": task.get("error_injection", []),
        "initial_registers": {}})
    if res.get("status") != "paused":
        return None
    return res.get("registers", {})


def main():
    ap = argparse.ArgumentParser(prog="baselines.qwen.make_sft")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-continuations", action="store_true")
    ap.add_argument("--domains", default=None, metavar="DIR",
                    help="register generated domain themes first")
    args = ap.parse_args()
    if args.domains:
        from data.gen.domains import register_domains
        register_domains(args.domains)

    from concurrent.futures import ThreadPoolExecutor

    tasks = [json.loads(l) for l in open(args.tasks)]
    multi = [] if args.no_continuations else [
        t for t in tasks if len(t["reference"]["segments"]) > 1]
    # sandbox runs are subprocess-bound; thread pool parallelizes Node spawns
    regmap: dict = {}
    if multi:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for t, regs in zip(multi, pool.map(
                    lambda t: pause_registers(
                        t, t["reference"]["segments"][0]), multi)):
                regmap[t["id"]] = regs

    n_seg0 = n_cont = n_skip = 0
    with open(args.out, "w") as out:
        for task in tasks:
            segs = task["reference"]["segments"]
            out.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": build_prompt(task["input_text"], None, [])},
                {"role": "assistant", "content": segs[0].strip()},
            ]}) + "\n")
            n_seg0 += 1
            if len(segs) > 1 and not args.no_continuations:
                regs = regmap.get(task["id"])
                if regs is None:
                    n_skip += 1
                    continue
                out.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",
                     "content": build_prompt(task["input_text"], regs,
                                             [segs[0].strip()])},
                    {"role": "assistant", "content": segs[1].strip()},
                ]}) + "\n")
                n_cont += 1
    print(f"wrote {n_seg0} seg0 + {n_cont} continuation examples "
          f"({n_skip} continuation skips) -> {args.out}")


if __name__ == "__main__":
    main()
