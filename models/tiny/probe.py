"""When does the diffusion sampler decide each kind of token?

This is the measurement the experiment exists for. If the model fills keywords
first and leaves the effects header and result registers until last, it is
solving the program in an order of its own choosing rather than left to right.
If the unmask step just tracks slot position, there is no evidence here for the
non-linear-solving intuition, and that is worth knowing before anything is built
on top of it.

    python probe.py --gen runs/diffusion_test.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from tok import OutVocab

CLASSES = {
    "header": lambda t: t in ("EFFECTS", "READ", "WRITE", "DELETE", "SEND", "PAY", "EXTERNAL"),
    "keyword": lambda t: t in ("CALL", "LET", "GET", "SET", "FORMAT", "FILTER", "MAP",
                               "COUNT", "SORT", "MOST", "LEAST", "SELECT", "FIRST",
                               "FOREACH", "IF", "ELSE", "PARALLEL", "TRY", "RETURN",
                               "STOP", "PAUSE", "ABORT"),
    "tool": lambda t: t.startswith("T") and t[1:].isdigit(),
    "field": lambda t: t.startswith("F") and t[1:].isdigit(),
    "const": lambda t: t[0] in "SNBDIC" and t[1:].isdigit(),
    "register": lambda t: t.startswith("r") and t[1:].isdigit(),
    "structure": lambda t: t in ("NL", "IND", "->"),
    "pad": lambda t: t == "PAD",
}


def classify(tok: str) -> str:
    for name, fn in CLASSES.items():
        try:
            if fn(tok):
                return name
        except IndexError:
            pass
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="a generated JSONL from evaluate.py")
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--compiled-only", action="store_true",
                    help="only count programs that compiled, so the order reflects "
                         "successful solving rather than noise")
    args = ap.parse_args()

    ov = OutVocab.load(Path(args.cache) / "out_vocab.json")
    rows = [json.loads(l) for l in open(args.gen, encoding="utf-8")]
    if args.compiled_only:
        rows = [r for r in rows if r.get("compiled_inline")]
    if not rows:
        raise SystemExit("no rows to probe")

    by_class = defaultdict(list)
    by_position = defaultdict(list)
    max_step = 0
    for r in rows:
        toks = ov.decode(r["canvas"])
        for slot, (tok, step) in enumerate(zip(toks, r["unmask_step"])):
            if step < 0:
                continue
            max_step = max(max_step, step)
            by_class[classify(tok)].append(step)
            by_position[slot].append(step)

    print(f"{len(rows)} programs, {max_step + 1} distinct unmask steps\n")
    print(f"{'token class':12s} {'n':>7s} {'mean step':>10s} {'median':>7s}")
    for name in sorted(by_class, key=lambda k: mean(by_class[k])):
        v = by_class[name]
        print(f"{name:12s} {len(v):7d} {mean(v):10.2f} {median(v):7.1f}")

    # If the unmask step is essentially the slot index, the sampler is decoding
    # left to right with extra arithmetic and the whole premise is unsupported.
    slots = sorted(by_position)
    xs = [s for s in slots for _ in by_position[s]]
    ys = [v for s in slots for v in by_position[s]]
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    corr = cov / (vx * vy) if vx and vy else float("nan")
    print(f"\ncorrelation between slot position and unmask step: {corr:+.3f}")
    print("  near +1 means left-to-right; near 0 means the order is content-driven")


if __name__ == "__main__":
    main()
