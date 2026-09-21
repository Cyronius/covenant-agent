"""Does the reference program survive the round trip, and still reach the goal?

This is the check that matters most before any training result is believed. It
takes the reference program out of the tensor cache, decodes it back to text
through the output vocabulary (flat) or the per-task codec (structural, where
`r0.F6` is two slots and every symbol is a pointer index), compiles it and
executes it in the sandbox. If the references do not score 100% here, then
every model number measured later is sitting on a broken data path and means
nothing.

    python test_pipeline.py --cache data_cache_smoke --n 60
"""
from __future__ import annotations

import argparse
from collections import Counter

import torch

from evaluate import Split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache_struct")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--id-contains", default=None, metavar="TEXT",
                    help="only rows whose task_id contains TEXT. A combined "
                         "holdout (data/s5_holdout_both.jsonl) stores the "
                         "plain rows first and the decoyed ones after, with "
                         "`+decoy` on their ids, so the first --n rows are "
                         "all plain and the decoyed half goes unchecked "
                         "without this")
    args = ap.parse_args()

    from sandbox import register_themes
    register_themes()
    import pickle
    from pathlib import Path
    from core.pipeline import build
    from harness.context import TaskContext
    from harness.run import run_task

    split = Split(Path(args.cache), args.split, torch.device("cpu"))
    rows = pickle.load(open(Path(args.cache) / "rows.pkl", "rb"))[args.split]
    by_id = {r["id"]: r for r in rows}
    print(f"binding: {split.binding}")

    stats = Counter()
    failures = []
    which = [i for i in range(len(split))
             if args.id_contains is None
             or args.id_contains in split.meta[i]["task_id"]]
    if args.id_contains is not None and not which:
        raise SystemExit(f"no {args.split} row's id contains "
                         f"{args.id_contains!r}")
    for i in which[:args.n]:
        row = by_id[split.meta[i]["task_id"]]
        text = split.codec(i).render(split.target(i)[0].tolist())
        ctx = TaskContext.from_json(row["context"])
        stats["n"] += 1

        # The reference has no EFFECTS header; ours is derived and prepended.
        # Compare against the reference with that line removed.
        original = row["reference"]["segments"][0]
        body = "\n".join(l for l in text.splitlines()
                         if not l.startswith("EFFECTS")) + "\n"
        stats["matches_reference"] += body.strip() == original.strip()

        res = build(text, ctx)
        stats["parse"] += bool(res.parse_ok)
        stats["compile"] += bool(res.compile_ok)
        if not res.compile_ok:
            failures.append((row["id"], res.rendered_diagnostics(), text))
            continue
        try:
            m = run_task(row, lambda *a, **k: text)
            ok = bool(m.get("goal_success"))
            stats["goal"] += ok
            if not ok:
                failures.append((row["id"], [f"goal_success=False status={m.get('status')}"], text))
        except Exception as exc:
            stats["sandbox_error"] += 1
            failures.append((row["id"], [f"sandbox: {exc}"], text))

    print(f"reference round-trip on {args.split}, {stats['n']} tasks")
    for k in ("matches_reference", "parse", "compile", "goal"):
        print(f"  {k:18s} {stats[k]:4d}/{stats['n']}")
    if stats["sandbox_error"]:
        print(f"  sandbox_error      {stats['sandbox_error']}")

    for tid, diags, text in failures[:5]:
        print(f"\n--- {tid} ---")
        for dg in diags[:4]:
            print(f"  {dg}")
        print("  program:")
        for line in text.splitlines():
            print(f"    {line}")

    ok = (stats["goal"] == stats["n"] and stats["compile"] == stats["n"]
          and stats["matches_reference"] == stats["n"])
    print("\nPASS" if ok else f"\nFAIL: {stats['n'] - stats['goal']} tasks did not reach the goal, "
          f"{stats['n'] - stats['matches_reference']} did not match the reference text")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
