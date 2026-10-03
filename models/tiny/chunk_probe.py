"""C0's $0 probe (.claude/plans/tiny-general-agent-menu.md, C0): does
Ternlight, reading the request in chunks, pick the constants a task's
reference program uses over the ones it doesn't? No training.

For every task with at least one used and one unused constant, each
constant gets a score from each method, and the probe reports:
  - pair: over every (used, unused) pair, how often the used one scores
    higher (ties count half). 50 is chance.
  - top1: how often the highest-scoring constant is a used one.

Methods:
  overlap     share of the constant's content words that appear in the
              request (lower case, a trailing s dropped). No model.
  whole       Ternlight: cosine of the whole request with the constant
              (what R22's reader has, except R22 never reads constants).
  chunk-K     Ternlight: best cosine over the request's windows of K words,
              moving K/2 words at a time (C0's input).
  overlap+X   overlap first, X breaking its ties: what X adds beyond
              matching words.

  python chunk_probe.py --reader reader [--out ../../results/logs/c0/probe.json]
  python chunk_probe.py --reader reader/role/r1/reader.pt --sizes 4   # a TernReader file

--reader is live_reader.open_reader's spec: the node dir, or a TernReader file.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from live_reader import open_reader
from prep import chunk_request, reader_text

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "demo": ROOT / "data" / "holdout" / "e_demo_requests.jsonl",
    "plain": ROOT / "data" / "s6_holdout_s500.jsonl",
    "clut": ROOT / "data" / "clt_holdout_s500.jsonl",
}
CONST_LINE = re.compile(r"^([CSNBDI]\d+) (.*?) :: (.*)$")
STOP = set("a an the of to in on for by with and or is are be it its this that "
           "all any every from at as into their them his her my me our your "
           "which who what when where".split())
ROLE_PAIRS = [
    ("move it to bob", "move it from bob"),
    ("assign card 6 to Bob", "assign Bob to card 6"),
    ("cards that are done", "cards that are not done"),
    ("orders placed before March", "orders placed after March"),
    ("send priya the list", "send the list about priya"),
    ("the oldest issue", "the newest issue"),
]
CALIBRATION = [
    ("bob", "Bob Alvarez"),
    ("bob", "Priya Nandan"),
    ("the release notes", "card 2 - Write release notes for v4.2"),
    ("the release notes", "card 4 - Retire legacy webhook handler"),
    ("finished", "the done status"),
    ("finished", "the todo status"),
]


def words(t: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", t.lower()):
        if w in STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def tasks(path: Path, dedupe: bool):
    seen = set()
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if dedupe:
            key = r["id"].rsplit("_s", 1)[0]
            if key in seen:
                continue
            seen.add(key)
        text = r["input_text"]
        consts, section = [], None
        for ln in text.splitlines():
            if ln.startswith("REQUEST: "):
                request = ln[len("REQUEST: "):]
            if ln in ("TOOLS:", "FIELDS:", "CONSTANTS:", "REGISTERS:"):
                section = ln
                continue
            if section == "CONSTANTS:":
                m = CONST_LINE.match(ln)
                if m:
                    consts.append((m.group(1), m.group(2), m.group(3)))
        prog = "\n".join(r["reference"]["segments"])
        used = set(re.findall(r"\b([CSNBDI]\d+)\b", prog))
        yield r["id"], request, consts, used


def kind(typ: str) -> str:
    head = typ.split()[0]
    if head.startswith("ID"):
        return "id"
    if "enum" in typ:
        return "enum"
    return "text" if head == "STR" else head.lower()


def rank_stats(scores: dict[str, list[float]], used: set[str], syms: list[str]):
    """(pair wins, pair count, top1 hit) for one task and one method."""
    u = [i for i, s in enumerate(syms) if s in used]
    n = [i for i, s in enumerate(syms) if s not in used]
    wins = 0.0
    for i in u:
        for j in n:
            a, b = scores[i], scores[j]
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    top = max(scores)
    tied = [i for i in range(len(syms)) if scores[i] == top]
    top1 = sum(1 for i in tied if syms[i] in used) / len(tied)
    return wins, len(u) * len(n), top1


def probe(rd, sizes: list[int], quiet: bool = False) -> dict:
    """Every suite's pair and top1 per method, and the role and calibration
    pairs' cosines, for one reader (anything with .vectors())."""
    say = (lambda *a: None) if quiet else print
    report = {}

    for suite, path in SUITES.items():
        agg = defaultdict(lambda: [0.0, 0, 0.0, 0])        # wins, pairs, top1, tasks
        by_kind = defaultdict(lambda: [0.0, 0])            # (method, kind) -> wins, pairs
        n_tasks = 0
        for tid, request, consts, used in tasks(path, dedupe=(suite == "demo")):
            syms = [s for s, _, _ in consts]
            if not (set(syms) & used) or not (set(syms) - used):
                continue
            n_tasks += 1
            ctexts = [reader_text(v) for _, _, v in consts]
            cv = rd.vectors(ctexts)
            rv = rd.vectors([request])
            ov = [len(words(t) & words(request)) / max(1, len(words(t))) for t in ctexts]
            methods = {"overlap": ov, "whole": (cv @ rv[0]).tolist()}
            for k in sizes:
                ch = rd.vectors(chunk_request(request, k))
                methods[f"chunk-{k}"] = (cv @ ch.T).max(1).values.tolist()
            for name in list(methods):
                if name != "overlap":
                    methods[f"overlap+{name}"] = [o + 1e-3 * s for o, s in zip(ov, methods[name])]
            for name, sc in methods.items():
                w, p, t1 = rank_stats(sc, used, syms)
                a = agg[name]
                a[0] += w; a[1] += p; a[2] += t1; a[3] += 1
                for i, (s, typ, _) in enumerate(consts):
                    if s not in used:
                        continue
                    for j, (s2, _, _) in enumerate(consts):
                        if s2 in used:
                            continue
                        x, y = sc[i], sc[j]
                        bk = by_kind[(name, kind(typ))]
                        bk[0] += 1.0 if x > y else 0.5 if x == y else 0.0
                        bk[1] += 1
        rows = {m: {"pair": round(100 * a[0] / a[1], 1), "top1": round(100 * a[2] / a[3], 1)}
                for m, a in agg.items()}
        kinds = sorted({k for _, k in by_kind})
        report[suite] = {"tasks": n_tasks, "methods": rows,
                         "pair_by_used_kind": {m: {k: (round(100 * by_kind[(m, k)][0] / by_kind[(m, k)][1], 1),
                                                       by_kind[(m, k)][1])
                                                   for k in kinds if by_kind[(m, k)][1]}
                                               for m in rows}}
        say(f"\n== {suite}: {n_tasks} tasks with used and unused constants")
        say(f"  {'method':22s} {'pair':>6s} {'top1':>6s}   pair by the used constant's kind (pairs)")
        for m, v in rows.items():
            bk = report[suite]["pair_by_used_kind"][m]
            say(f"  {m:22s} {v['pair']:6.1f} {v['top1']:6.1f}   "
                + "  ".join(f"{k} {p:.0f} ({n})" for k, (p, n) in bk.items()))

    say("\n== role pairs (cosine; 1.0 = the reader can't tell them apart)")
    pairs = {}
    for a, b in ROLE_PAIRS + CALIBRATION:
        v = rd.vectors([a, b])
        c = float(v[0] @ v[1])
        pairs[f"{a} | {b}"] = round(c, 3)
        say(f"  {c:.3f}  {a!r} vs {b!r}")
    report["pairs"] = pairs
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, metavar="DIR|FILE")
    ap.add_argument("--tier", default="mini")
    ap.add_argument("--sizes", default="2,3,4,6")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rd = open_reader(args.reader, args.tier)
    report = probe(rd, [int(k) for k in args.sizes.split(",")])
    report["reader"] = rd.name
    rd.close()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
