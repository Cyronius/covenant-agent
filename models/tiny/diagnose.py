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

`compare_calls` is the third measurement on its own; `evaluate.py --score`
reports it with every scored file.
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


def compare_calls(reference: str, program: str, tools: dict) -> Counter:
    """Line-aligned CALL agreement between a generation and its reference.

    Counts: `compared` (reference CALL lines where the generation also wrote
    a CALL), `same_tool`, `same_effect`, and the exact chance for each: a
    uniform pick over the task's declared tools.
    """
    m = Counter()
    gen_calls = dict(calls(program))
    for line, sym in calls(reference):
        if sym not in tools:
            continue
        sig = tools[sym][0]
        pick = gen_calls.get(line)
        if pick is None:
            m["no_call_on_line"] += 1
            continue
        m["compared"] += 1
        m["same_tool"] += pick == sym
        if pick in tools:
            m["same_effect"] += effect(tools[pick][0]) == effect(sig)
        m["chance_tool"] += 1 / len(tools)
        m["chance_effect"] += sum(1 for s, _ in tools.values()
                                  if effect(s) == effect(sig)) / len(tools)
    return m



# -- where a program goes wrong, by line ---------------------------------

# Token classes inside a FILTER predicate. Enough to say whether a wrong
# predicate is a wrong SYMBOL or a wrong SHAPE, which are different failures
# with different fixes.
#
# The operators are the spec's, not a guess: `cmp = EQ | LT | GT | CONTAINS |
# IN` (spec/agent_core.md:99). NE, LE and GE are not in the language.
OPS = {"EQ", "LT", "GT", "CONTAINS", "IN"}
JOIN = {"AND", "OR", "NOT"}


def token_class(tok: str) -> str:
    if tok in OPS:
        return "op"
    if tok in JOIN:
        return "join"
    if re.fullmatch(r"F\d+", tok):
        return "field"
    if re.fullmatch(r"r\d+\.?", tok):
        return "reg"
    if re.fullmatch(r"[CSNBDI]\d+|NOW|TRUE|FALSE", tok):
        return "const"
    return tok.lower()


def shape(line: str) -> tuple[str, ...]:
    """The line as a sequence of token classes: its grammatical shape."""
    return tuple(token_class(t) for t in line.split())


def adjacency_faults(line: str) -> list[str]:
    """Where a predicate breaks the grammar's own production, named.

    Straight off `spec/agent_core.md:95-99`:

        pred   = clause { ("AND" | "OR") clause }
        clause = [ "NOT" ] field cmp ( operand | field )

    Which makes every rule below local -- it constrains what may follow what,
    nothing more. That is the class of constraint a per-slot mask over the canvas
    could carry and decision 8's symbol mask does not: the symbol mask says which
    `F` symbols exist, not that a comparison needs a field on its left.
    `sample.local_mask` already carries one rule of exactly this kind (a receiver
    slot must be followed by a field), so the machinery is there.

    Note `AND NOT` and `OR NOT` are legal: a clause may open with NOT. An earlier
    version of this function counted them as faults and reported 30 per 67
    programs on both arms, which is the corpus obeying the grammar.
    """
    toks = line.split()
    if toks and toks[0] == "FILTER":
        toks = toks[1:]                      # drop the keyword and the source reg
        toks = toks[1:] if toks else toks
    if "->" in toks:
        toks = toks[:toks.index("->")]
    out = []
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        cls = token_class(t)
        nc = token_class(nxt) if nxt else None
        if t in ("AND", "OR"):
            if nxt is None or not (nxt == "NOT" or nc == "field"):
                out.append(f"{t} then {nxt or 'end'}")
        elif t == "NOT":
            if nc != "field":
                out.append(f"NOT then {nxt or 'end'}")
        elif cls == "op":
            prev = token_class(toks[i - 1]) if i else None
            if prev != "field":
                out.append(f"{cls} with {prev or 'nothing'} on its left")
            if nxt is None or nc in ("op", "join"):
                out.append(f"{cls} then {nc or 'end'}")
    return out


def line_report(gen: dict, levels: dict, scored: dict | None) -> None:
    """Which line breaks first, whether it breaks its shape or only its symbols,
    and whether the program still reached the goal.

    The distinction is the point. A predicate that names the wrong field is a
    binding failure. A predicate whose token classes do not compose -- two joins
    in a row, an operator where a field belongs -- is a grammar failure, and the
    two have nothing to do with each other.
    """
    first = Counter()
    kind = Counter()
    faults = Counter()
    equiv = Counter()
    for tid, g in gen.items():
        gl = [l.strip() for l in g["program"].strip().split("\n") if l.strip()]
        rl = [l.strip() for l in g["reference"].strip().split("\n") if l.strip()]
        goal = bool(scored.get(tid, {}).get("goal")) if scored else None
        if gl == rl:
            first["exact"] += 1
            continue
        for a, b in zip(gl + [""] * max(0, len(rl) - len(gl)),
                        rl + [""] * max(0, len(gl) - len(rl))):
            if a == b:
                continue
            key = (b.split(" ")[0] if b else "extra line") or "extra line"
            first[key] += 1
            if goal is not None:
                # A differing line that still reaches the goal was a different
                # way of saying the same thing, not an error.
                equiv[(key, "goal" if goal else "failed")] += 1
            if shape(a) == shape(b):
                kind[key + ": same shape, wrong symbols"] += 1
            else:
                kind[key + ": wrong shape"] += 1
            for f in adjacency_faults(a):
                faults[f] += 1
            break

    total = max(sum(first.values()), 1)
    print(f"\n  first differing line, over {total} programs"
          f" ({first.get('exact', 0)} exact):")
    for k, v in first.most_common(8):
        if k == "exact":
            continue
        print(f"    {k:14s} {v:5d}  {v/total:5.1%}")
    if equiv:
        print("\n  of those differing lines, did the program still reach the goal:")
        keys = sorted({k for k, _ in equiv})
        for k in keys:
            ok, bad = equiv[(k, "goal")], equiv[(k, "failed")]
            if ok + bad:
                print(f"    {k:14s} goal {ok:4d}  failed {bad:4d}  "
                      f"({ok/(ok+bad):.0%} of differences were harmless)")
    if kind:
        print("\n  was the shape wrong, or only the symbols in it:")
        for k, v in sorted(kind.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {k:48s} {v:5d}")
    if faults:
        print("\n  local adjacency rules broken (a per-slot mask could carry these):")
        for k, v in faults.most_common(8):
            print(f"    {k:22s} {v:5d}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="a generated JSONL from evaluate.py")
    ap.add_argument("--cache", default="data_cache_struct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lines", action="store_true",
                    help="which line of a program breaks first, whether its shape "
                         "or only its symbols, and whether the goal was reached "
                         "anyway; reads the .score.json beside --gen if it exists")
    ap.add_argument("--level", type=int, default=None,
                    help="restrict --lines to one curriculum level")
    args = ap.parse_args()

    rows = pickle.load(open(Path(args.cache) / "rows.pkl", "rb"))[args.split]
    gen = {}
    for line in open(args.gen, encoding="utf-8"):
        r = json.loads(line)
        gen[r["task_id"]] = r
    strip = lambda s: re.sub(r"=F\d+", "", s)          # drop per-request field symbols

    if args.lines:
        levels = {r["id"]: r.get("level") for r in rows}
        want = {t: g for t, g in gen.items()
                if args.level is None or levels.get(t) == args.level}
        sc_path = Path(args.gen).with_suffix(".score.json")
        scored = None
        if sc_path.exists():
            scored = {r["task_id"]: r for r in json.loads(
                sc_path.read_text(encoding="utf-8"))["rows"] if "task_id" in r}
        where = f"level {args.level}" if args.level is not None else "every level"
        print(f"{args.gen}  {where}, {len(want)} programs"
              + ("" if scored else "  (no .score.json beside it: no goal column)"))
        line_report(want, levels, scored)
        return

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
        for line, sym in calls(g["reference"]):
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
        m.update(compare_calls(g["reference"], g["program"], tools))
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
