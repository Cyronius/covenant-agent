"""Signature uniqueness, a lexical baseline, and what the model actually picked.

Run this on every generated file before believing anything about grounding. It
answers three questions from the reference programs and the task contexts alone:

  - does the tool's signature (everything before `::`) identify it without the
    description?  In s5_plain it does, for every call.
  - would a bag-of-words match between request and description pick the tool?
  - when the model wrote a CALL on the same line as the reference, did it pick
    the same tool, or at least one with the same effect class?  A model that
    reads signatures is far above chance on the effect row; one that does not
    bind symbols to lines sits at chance on both.

    python diagnose.py --gen out/ar_s0_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import COVENANT  # noqa: F401,E402  puts covenant-agent on sys.path
from harness.context import TaskContext, serialize_context  # noqa: E402

STOP = set("a an the of to for on in by with and or from is are be it its this that "
           "one all any each every please could you can i need me go ahead hey thanks "
           "then which what how where tell".split())


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_]+", s.lower()) if w not in STOP and len(w) > 1}


def parse(src: str):
    tools, fields = {}, {}
    for line in src.splitlines():
        if " :: " not in line:
            continue
        head, desc = line.split(" :: ", 1)
        sym, rest = head.split(" ", 1)
        if re.fullmatch(r"T\d+", sym):
            tools[sym] = (rest, desc)
        elif re.fullmatch(r"F\d+", sym):
            fields[sym] = (rest, desc)
    return tools, fields


def calls(prog: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(prog.splitlines()):
        p = line.strip().split()
        if len(p) >= 2 and p[0] == "CALL":
            out.append((i, p[1]))
    return out


def effect(sig: str) -> str:
    m = re.search(r"\[(.*?)\]", sig)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="a generated JSONL from evaluate.py")
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = pickle.load(open(Path(args.cache) / "rows.pkl", "rb"))[args.split]
    gen = {}
    for line in open(args.gen, encoding="utf-8"):
        r = json.loads(line)
        gen[r["task_id"]] = r
    strip = lambda s: re.sub(r"=F\d+", "", s)          # drop per-request field symbols

    n = 0
    uniq_full = uniq_stripped = lex = 0.0
    fld_n = fld_uniq = 0
    m = Counter()
    for r in rows:
        g = gen.get(r["id"])
        if g is None:
            continue
        ctx = TaskContext.from_json(r["context"])
        tools, fields = parse(serialize_context(r["request"], ctx))
        req_w = words(r["request"])
        ref_calls = calls(g["reference"])
        gen_calls = dict(calls(g["program"]))
        for line, sym in ref_calls:
            if sym not in tools:
                continue
            n += 1
            sig = tools[sym][0]
            uniq_full += sum(1 for s, _ in tools.values() if s == sig) == 1
            uniq_stripped += sum(1 for s, _ in tools.values() if strip(s) == strip(sig)) == 1
            sc = {s: len(req_w & words(d)) for s, (_, d) in tools.items()}
            best = max(sc.values())
            win = [s for s, v in sc.items() if v == best]
            lex += (1 / len(win)) if sym in win else 0
            pick = gen_calls.get(line)
            if pick is None:
                m["no_call_on_line"] += 1
                continue
            m["compared"] += 1
            m["same_tool"] += pick == sym
            if pick in tools:
                m["same_effect"] += effect(tools[pick][0]) == effect(sig)
            # Exact chance: a uniform pick over this task's declared tools.
            m["chance_tool"] += 1 / len(tools)
            m["chance_effect"] += sum(1 for s, _ in tools.values()
                                      if effect(s) == effect(sig)) / len(tools)
        for tok in set(re.findall(r"\bF\d+\b", g["reference"])):
            if tok in fields:
                fld_n += 1
                fld_uniq += sum(1 for s, _ in fields.values() if s == fields[tok][0]) == 1

    if n == 0:
        raise SystemExit("no reference CALLs matched; wrong --split or --cache?")
    c = max(m["compared"], 1)
    print(f"{n} reference CALLs over {len(rows)} tasks")
    print(f"  called tool unique by full signature:            {uniq_full / n:6.1%}")
    print(f"  unique by types and effects, field symbols out:  {uniq_stripped / n:6.1%}")
    print(f"  lexical overlap with the request picks it:       {lex / n:6.1%}")
    print(f"{fld_n} referenced fields, unique by entity and type:   {fld_uniq / max(fld_n, 1):6.1%}")
    print(f"\nmodel, on {m['compared']} CALL lines where it also wrote a CALL:")
    print(f"  same tool    {m['same_tool'] / c:6.1%}   chance {m['chance_tool'] / c:6.1%}")
    print(f"  same effect  {m['same_effect'] / c:6.1%}   chance {m['chance_effect'] / c:6.1%}")
    print("\n  same-effect at chance means the model is not reading signatures at all.")


if __name__ == "__main__":
    main()
