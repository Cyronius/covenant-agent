"""What do real requests ask for that this language cannot say?

Step 2c of `.claude/plans/agent-loop-and-ir-review.md`. The admission rule
(`spec/agent_core.md` §11) asks four things of a candidate instruction, and
only one of them — "is it expressible today" — can be answered at a desk.
The other three are measurements, and `CONTAINS`'s list arm is what happens
when they are skipped: admitted on intuition, retired unused.

The corpus cannot answer the demand question, because every request in it is
expressible by construction — the recipes only write what the grammar can
say. Two sources on disk are not built that way: the real mobi session turns
(`data/holdout/e_real_sessions.jsonl`, and the wider scrubbed pool under
`data/real_sessions/`) and the imported human-phrased requests in
`data/open_pairs/`.

Three passes, in the order they cost money:

  patterns    what the request *wording* asks for. No model, no GPU. A
              lower bound per candidate: wording that matches can only be
              answered by that candidate, wording that paraphrases is
              missed. Prints the matches so a human can price the error.
  workarounds what our own reference programs do the long way, over any
              task file. No model. Tier B/C evidence: a candidate whose
              workaround nothing uses is not being paid for.
  label       what a model actually failed to say, over a finished
              `baselines/qwen/run_a.py` output: every `ABORT UNSUPPORTED`,
              every non-compiling attempt, joined with the request's
              pattern labels. This is the pass that needs a pod, and it
              needs the run first — it does not call a model itself.

The candidates are the ones already on record in `results/R1.md`'s holding
pen. The census does not admit anything; it produces the demand number the
admission rule's first test asks for and nobody has.

  python -m harness.expressibility_census patterns [--source real|open|all]
        [--show N] [--out FILE]
  python -m harness.expressibility_census workarounds --tasks FILE [--show N]
  python -m harness.expressibility_census label --run FILE --tasks FILE
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ir import (Abort, Filter, Foreach, If, MapF, Select,  # noqa: E402
                     Sort, TaskContext, Try)
from core.pipeline import build  # noqa: E402

# ------------------------------------------------------------------ sources
# Real requests, in the two shapes on disk. The eval file is the reviewed,
# committed 234; the pool is the wider scrubbed mining set (gitignored — raw
# sessions carry PII, `harness/real_requests.py`), used when present.
REAL_SOURCES = (
    Path("data/holdout/e_real_sessions.jsonl"),
    Path("data/real_sessions/b2_pool.jsonl"),
)
OPEN_SOURCES = (
    Path("data/open_pairs/glaive_12k.jsonl"),
    Path("data/open_pairs/hermes_full.jsonl"),
    Path("data/open_pairs/toolace_4k.jsonl"),
)

_NUM = r"(?:[2-9]|\d\d+|two|three|four|five|six|seven|eight|nine|ten|dozen)"
# "the last 24 hours" and "the first 30 minutes" are time windows, not the
# prefix of a sorted list; they were 20 of the first 66 `first-n` matches on
# the imported requests before this guard
_NOT_TIME = (r"(?!\s*(?:hours?|hrs?|days?|minutes?|mins?|seconds?|secs?|"
             r"weeks?|months?|years?|quarters?|decades?)\b)")
_SUP = (r"(?:most|least|oldest|newest|latest|largest|smallest|biggest|"
        r"highest|lowest|earliest|recent|longest|shortest|worst|best|top)")

# Each candidate is a list of (probe name, pattern). A probe is deliberately
# narrow: it should only fire on wording that *needs* the candidate, because
# the false positives are what a human has to price and the false negatives
# are what the `label` pass is for.
CANDIDATES: dict[str, list[tuple[str, re.Pattern]]] = {
    # Tier A: the first n of an ordered list. SORT+FIRST takes one, SELECT
    # takes an index, nothing takes a prefix.
    "take": [
        ("top-n", re.compile(rf"\btop\s+{_NUM}\b{_NOT_TIME}", re.I)),
        ("n-superlative", re.compile(rf"\b{_NUM}\s+(?:{_SUP})\b", re.I)),
        ("first-n", re.compile(rf"\b(?:first|last|next)\s+{_NUM}\b"
                               rf"{_NOT_TIME}", re.I)),
        ("n-of-them", re.compile(rf"\b{_NUM}\s+(?:of\s+(?:the|them|those))\b",
                                 re.I)),
    ],
    # Tier A: the distinct values of a field. MAP keeps duplicates, MOST
    # returns one. Narrow on purpose: "which X have Y" is often answered
    # acceptably with the duplicated list, so it is not counted here.
    "unique": [
        ("distinct", re.compile(r"\b(?:distinct|de-?dupl?i?c?a?t?e?d?)\b",
                                re.I)),
        ("unique-values", re.compile(r"\bunique\s+(?:values?|list|set|"
                                     r"names?|entries)\b", re.I)),
        ("how-many-different", re.compile(r"\bhow many (?:different|distinct|"
                                          r"unique)\b", re.I)),
        ("what-different", re.compile(r"\bwhat(?:'s| is| are)?\s+(?:the\s+)?"
                                      r"(?:different|distinct)\b", re.I)),
    ],
    # Tier C: a table keyed by something — the result is a list of pairs and
    # the type system has no pair. The 3-line nested-FOREACH workaround is
    # what `workarounds` counts.
    "group_join": [
        ("grouped-by", re.compile(r"\bgroup(?:ed)?\s+by\b", re.I)),
        ("breakdown", re.compile(r"\bbreak(?:down|\s+down)\b", re.I)),
        ("count-per", re.compile(r"\b(?:how many|count|total|number|sum|"
                                 r"average)\b[^.?!]{0,40}?\b(?:per|for each|"
                                 r"by)\s+\w+", re.I)),
        ("each-possessive", re.compile(r"\beach\s+\w+(?:'s|s')\s+\w+", re.I)),
    ],
    # Tier B: NOT over a group. AND binds tighter than OR, so some forms are
    # reachable by De Morgan and negated groups are not.
    "predicate_group": [
        ("not-both", re.compile(r"\bnot both\b", re.I)),
        ("neither-nor", re.compile(r"\bneither\b[^.?!]{0,60}\bnor\b", re.I)),
        ("unless-and", re.compile(r"\bunless\b[^.?!]{0,60}\band\b", re.I)),
        ("except-when", re.compile(r"\bexcept (?:when|if|where)\b", re.I)),
    ],
    # Step 5 / the resumable ABORT: a request that only makes sense against
    # the previous turn. Not an opcode — evidence for letting new English
    # into a running task, and for an ABORT the host can answer.
    "followup": [
        ("do-the-same", re.compile(r"\b(?:same|likewise|ditto)\b[^.?!]{0,30}"
                                   r"\b(?:for|with|to)\b", re.I)),
        ("now-also", re.compile(r"^\s*(?:and\s+)?(?:now|also|then)\b", re.I)),
        ("instead", re.compile(r"\binstead\b", re.I)),
        ("that-one", re.compile(r"\b(?:that|this|those|these)\s+one\b", re.I)),
    ],
    # Not a candidate instruction at all — 1g made these tools. Counted
    # because the count decides how many answering recipes are worth
    # writing (step 2b's weighting, `results/REFLEX.md` §6).
    "aggregate_tool": [
        ("total", re.compile(r"\b(?:total|sum)\b", re.I)),
        # not bare "mean": "I mean lesson titles only" is not an average
        ("average", re.compile(r"\b(?:average|avg)\b|\bthe mean\b", re.I)),
        ("how-much", re.compile(r"\bhow much\b", re.I)),
        ("extreme", re.compile(r"\b(?:maximum|minimum|max|min|highest|"
                               r"lowest)\b", re.I)),
    ],
}


# Not a person typing. The product appends a continuity line of its own when
# a run spans modules ("Earlier modules in this run updated 6 titles. Apply
# the same intent and tone to this module."), which reads exactly like a
# human follow-up and would otherwise carry the `followup` count on its own.
# Counted separately rather than dropped: it is real input the agent gets.
SCAFFOLD = re.compile(r"Earlier modules in this run|"
                      r"Apply the same intent and tone", re.I)


def load_requests(paths) -> list[dict]:
    """[{request, source, id}] over whichever of `paths` exist."""
    rows = []
    for path in paths:
        p = ROOT / path
        if not p.exists():
            print(f"  (absent, skipped: {path})", file=sys.stderr)
            continue
        n = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("request") or row.get("text") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                rows.append({"request": text.strip(), "source": path.name,
                             "id": row.get("id") or row.get("turn_id") or "",
                             "scaffolded": bool(SCAFFOLD.search(text))})
                n += 1
        print(f"  {n} requests from {path}", file=sys.stderr)
    return rows


def label_request(text: str) -> dict[str, list[tuple[str, str]]]:
    """{candidate -> [(probe, matched text), ...]} for one request."""
    hits: dict[str, list[tuple[str, str]]] = {}
    for candidate, probes in CANDIDATES.items():
        for name, pattern in probes:
            m = pattern.search(text)
            if m:
                hits.setdefault(candidate, []).append((name, m.group(0)))
    return hits


# ----------------------------------------------------------- pass 1: wording
def patterns_pass(sources: str, show: int, out: Path | None) -> int:
    paths = {"real": REAL_SOURCES, "open": OPEN_SOURCES,
             "all": REAL_SOURCES + OPEN_SOURCES}[sources]
    rows = load_requests(paths)
    if not rows:
        print("no requests found", file=sys.stderr)
        return 1

    by_source = collections.Counter(r["source"] for r in rows)
    counts: dict[str, collections.Counter] = {
        c: collections.Counter() for c in CANDIDATES}
    human: dict[str, int] = {c: 0 for c in CANDIDATES}
    probe_counts: dict[str, collections.Counter] = {
        c: collections.Counter() for c in CANDIDATES}
    examples: dict[str, list] = {c: [] for c in CANDIDATES}
    labelled = []
    for row in rows:
        hits = label_request(row["request"])
        for candidate, found in hits.items():
            counts[candidate][row["source"]] += 1
            human[candidate] += 0 if row["scaffolded"] else 1
            for probe, text in found:
                probe_counts[candidate][probe] += 1
            if len(examples[candidate]) < show:
                examples[candidate].append(
                    (row["source"], found[0][1], row["request"]))
        if hits:
            labelled.append({**row, "candidates": sorted(hits),
                             "matches": {c: v for c, v in hits.items()}})

    scaffolded = sum(1 for r in rows if r["scaffolded"])
    typed = len(rows) - scaffolded
    print(f"\n{len(rows)} requests: "
          + ", ".join(f"{n} {s}" for s, n in by_source.most_common()))
    if scaffolded:
        print(f"{scaffolded} of them carry the product's own continuity line; "
              f"the `typed` column is the other {typed}")
    print("\ncandidate        share   rows    typed          by probe")
    print("-" * 78)
    for candidate in CANDIDATES:
        n = sum(counts[candidate].values())
        probes = ", ".join(f"{p} {k}" for p, k in
                           probe_counts[candidate].most_common())
        print(f"{candidate:16s} {n / len(rows):5.1%} {n:6d}   "
              f"{human[candidate]:4d} ({human[candidate] / typed:5.1%})  "
              f"{probes}")
    for candidate in CANDIDATES:
        if not examples[candidate]:
            continue
        print(f"\n--- {candidate}: first {len(examples[candidate])} matches "
              f"(for the human pass)")
        for source, matched, request in examples[candidate]:
            print(f"  [{matched}] {request[:150]}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in labelled:
                fh.write(json.dumps(row) + "\n")
        print(f"\n{len(labelled)} labelled rows -> {out}")
    return 0


# ------------------------------------------------- pass 2: what we do by hand
def walk(body):
    """Every instruction in `body`, descending into the block-carrying ones."""
    for instr in body:
        yield instr
        if isinstance(instr, Foreach):
            yield from walk(instr.body)
        elif isinstance(instr, If):
            yield from walk(instr.then)
            yield from walk(instr.els or [])
        elif isinstance(instr, Try):
            yield from walk(instr.body)


def workarounds(program) -> list[str]:
    """Which candidate's absence this program is working around."""
    found = []
    body = list(walk(program.body))
    sorted_regs = {i.dst.n for i in body if isinstance(i, Sort)}
    picks = collections.Counter(i.src.n for i in body
                                if isinstance(i, Select))
    if any(picks[r] >= 2 for r in sorted_regs):
        # two or more SELECTs off one sorted register is a hand-written
        # prefix: TAKE is the one instruction that would say it
        found.append("take")
    for instr in body:
        if not isinstance(instr, Foreach):
            continue
        for inner in walk(instr.body):
            # the ~3-line join: iterate the outer list, then filter the inner
            # one *by the loop variable* (results/R1.md, Tier C). A FOREACH
            # with any old FILTER inside is not a join, so the clause has to
            # name the loop register for this to count.
            if isinstance(inner, Filter) and \
                    _mentions(inner.pred, instr.var.n):
                found.append("group_join")
    return sorted(set(found))


def _mentions(pred, reg: int) -> bool:
    """Does any clause of `pred` read register `reg`?"""
    for clause in pred.clauses:
        for side in (clause.left, clause.right):
            if getattr(side, "n", None) == reg:
                return True
            if getattr(getattr(side, "reg", None), "n", None) == reg:
                return True
    return False


def workarounds_pass(tasks: Path, show: int, limit: int,
                     stride: int) -> int:
    counts = collections.Counter()
    examples = collections.defaultdict(list)
    rows = built = seen = 0
    with tasks.open(encoding="utf-8") as fh:
        for line in fh:
            if rows >= limit:
                break
            seen += 1
            # a generated corpus is written level by level, so the first N
            # rows are one recipe's worth of the language and say nothing
            # about the rest; stride across the whole file instead
            if (seen - 1) % stride:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            segments = (row.get("reference") or {}).get("segments") or []
            if not segments or "context" not in row:
                continue
            rows += 1
            ctx = TaskContext.from_json(row["context"])
            for seg in segments:
                res = build(seg, ctx)
                if res.program is None:
                    continue
                built += 1
                for candidate in workarounds(res.program):
                    counts[candidate] += 1
                    if len(examples[candidate]) < show:
                        examples[candidate].append(
                            (row.get("request", ""), seg.strip()))
    if not built:
        print(f"no reference segments built from {tasks}", file=sys.stderr)
        return 1
    print(f"\n{built} reference segments over {rows} rows in {tasks}"
          + (f" (every {stride}th of {seen} lines)" if stride > 1 else ""))
    print("\ncandidate        segments  share")
    print("-" * 40)
    for candidate in sorted(CANDIDATES):
        n = counts[candidate]
        print(f"{candidate:16s} {n:8d}  {n / built:5.1%}")
    for candidate, rows_ in examples.items():
        print(f"\n--- {candidate}: {len(rows_)} example segments")
        for request, seg in rows_:
            print(f"  {request[:100]}")
            for line in seg.splitlines():
                print(f"      {line}")
    return 0


# ------------------------------------------- pass 3: what a model could not say
def label_pass(run: Path, tasks: Path, show: int) -> int:
    """Join a finished run_a.py output with the census labels.

    Needs the run first; this does not call a model. The pod recipe is:

        python baselines/qwen/run_a.py --model <gguf> --tasks <census tasks>
            --think 512 --out <run.jsonl>
        python -m harness.expressibility_census label --run <run.jsonl>
            --tasks <census tasks>
    """
    by_id = {}
    with tasks.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                by_id[row["id"]] = row
    outcome = collections.Counter()
    demand = collections.Counter()
    unsupported_examples = []
    n = 0
    with run.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = by_id.get(row.get("id"))
            if task is None:
                continue
            n += 1
            request = task.get("request", "")
            hits = sorted(label_request(request))
            programs = row.get("programs") or []
            ctx = TaskContext.from_json(task["context"])
            kind = "expressed"
            for text in programs:
                res = build(text, ctx)
                if not res.compile_ok:
                    kind = "did_not_compile"
                    break
                first = res.program.body[0] if res.program.body else None
                if isinstance(first, Abort) and first.reason == "UNSUPPORTED":
                    kind = "abort_unsupported"
                    break
                if workarounds(res.program):
                    kind = "worked_around"
            outcome[kind] += 1
            if kind != "expressed":
                for candidate in hits or ["unlabelled"]:
                    demand[candidate] += 1
                if kind == "abort_unsupported" and \
                        len(unsupported_examples) < show:
                    unsupported_examples.append((request, hits))
    if not n:
        print("no run rows matched the task file (ids disjoint?)",
              file=sys.stderr)
        return 1
    print(f"\n{n} rows of {run.name} against {tasks.name}")
    print("\noutcome            rows   share")
    print("-" * 40)
    for kind, k in outcome.most_common():
        print(f"{kind:18s} {k:5d}  {k / n:5.1%}")
    print("\nwhat the ones that were not expressed asked for:")
    for candidate, k in demand.most_common():
        print(f"  {candidate:16s} {k:5d}")
    for request, hits in unsupported_examples:
        print(f"\n  ABORT UNSUPPORTED [{','.join(hits) or '-'}] "
              f"{request[:140]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="harness.expressibility_census")
    sub = ap.add_subparsers(dest="pass_", required=True)

    p = sub.add_parser("patterns", help="demand from request wording")
    p.add_argument("--source", choices=["real", "open", "all"], default="all")
    p.add_argument("--show", type=int, default=8,
                   help="example matches to print per candidate")
    p.add_argument("--out", type=Path, default=None,
                   help="write the labelled rows for the human pass")

    p = sub.add_parser("workarounds", help="what references do the long way")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--show", type=int, default=2)
    p.add_argument("--limit", type=int, default=4000,
                   help="rows to build; a training corpus is 30k rows of "
                        "which every reference segment gets compiled")
    p.add_argument("--stride", type=int, default=1, metavar="N",
                   help="build every Nth row, so the sample spans a corpus "
                        "written level by level")

    p = sub.add_parser("label", help="join a finished run_a.py output")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--show", type=int, default=10)

    args = ap.parse_args()
    if args.pass_ == "patterns":
        return patterns_pass(args.source, args.show, args.out)
    if args.pass_ == "workarounds":
        return workarounds_pass(args.tasks, args.show, args.limit,
                                args.stride)
    return label_pass(args.run, args.tasks, args.show)


if __name__ == "__main__":
    sys.exit(main())
