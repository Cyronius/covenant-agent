"""The imported open-schema tools, as a distractor pool.

`data/open_pairs/*.jsonl` holds 6,893 converted rows from three open
function-calling sets (glaive / hermes / ToolACE, see `convert_open.py`).
The obvious use — recombine them into 18-60 tool worlds and train on those —
is the wrong one, and it is measured wrong in
`.claude/plans/imported-schemas-as-distractors.md` §2: a tf-idf baseline over
request-vs-tool text scores 94.2% on those worlds as-is and 77.0% after
crowding them to 60 tools, against 13.4% on `s5_plain`. Every imported
parameter is a bare scalar, every `returns` is null and every field carries
`entity: null`, so there is no `ID:`/`OBJ:` structure for selection to go
through. They are an easier task, not a harder one.

What they *are* is 7,263 tool names the encoder has never read, against a
themed vocabulary of 1,896 that they intersect in 8. So they enter the corpus
as distractors injected into themed worlds, where the program, the entities
and the typing stay the donor world's and the encoder has to read and reject
42 descriptions written in a vocabulary it has never seen. Injected that way
the same lexical baseline falls to 7.9%.

The pool is world-form tool dicts, reconstructed from each row's serialized
context: the param name lives on the row's FieldDecl (`fields[sym].name`),
the impl is the `external` stub `convert_open.py` gave it, which returns a
deterministic string, so a wrong-but-well-typed call executes rather than
being statically impossible.

Two things foreign text does that themed schemas never do, both of which
stop a `models/tiny/prep.py` run dead after the corpus is already generated:
a description with a newline in it (`one_line`), and a parameter list long
enough to overflow a schema line (`fit_params`). 7,257 of the 7,263 names
survive both.

  from data.gen.open_pool import load_pool
  pool = load_pool(holdout=False)     # the training side of the split

  python -m data.gen.open_pool        # inventory
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
OPEN_PAIRS = ROOT / "data" / "open_pairs"

# The three sources, and only them. `b3_draw.jsonl` is a re-draw of rows
# already in these files (`draw_open.py`), not a fourth source; counting it
# is what produced the inflated 8,242-row / ~10,000-name inventory the plan
# corrects. `*_pilot.jsonl` are exact subsets. See data/open_pairs/README.md.
SOURCES = ("glaive_12k.jsonl", "hermes_full.jsonl", "toolace_4k.jsonl")

# Longest parameter list a distractor may carry. The cap is the model's,
# not the schema's: a tool line renders one `S=F123?` slot per parameter and
# `models/tiny/prep.py` refuses to truncate a schema line past `--max-line`
# (64 tokens). Measured over 600 injected rows, tool-line tokens by parameter
# count run 49/55/64 (p50/p95/max) at 6, 52/58/59 at 7 and 59/64/68 at 8, so
# 7 is the largest list that always fits. Unset it and 46% of injected rows
# stop the prep run with "a schema line is 90 tokens, cap is 64" -- after the
# whole corpus is generated. Optional parameters past the cap are trimmed
# rather than the tool dropped: a distractor is never called, so its tail is
# not load-bearing, and trimming keeps 7,257 of the 7,263 names instead of
# 7,104. The 6 tools with more than 7 REQUIRED parameters are dropped.
MAX_PARAMS = 7

# Share of the pool reserved for holdout worlds, mirroring the themed domain
# split (41 eval of 143 themes, data/holdout/reserved_domains.json). A
# holdout world's distractors have to be unseen too, or the held-out split
# stops measuring whether the encoder can read an unfamiliar schema.
HOLDOUT_PCT = 30

_CACHE: Dict[tuple, List[dict]] = {}

_WS = re.compile(r"\s+")


def one_line(text: str) -> str:
    """A description that survives the line-oriented context serializer.

    `serialize_context` writes one line per tool and per field, and
    `models/tiny/prep.py` parses that back by line, so a description with a
    newline in it splits into a line the parser refuses ("unrecognized
    context line"). Four of the 7,263 imported descriptions have one, which
    means it lands rarely and then stops a whole prep run after the corpus
    is already generated. ` :: ` is the serializer's description separator
    and none of them carry it today; normalising both here is where foreign
    text enters, and it is the only place that has to know.
    """
    return _WS.sub(" ", str(text)).replace(" :: ", " - ").strip()


def fit_params(params: List[dict], cap: int = MAX_PARAMS) -> List[dict] | None:
    """At most `cap` parameters: every required one, then the earliest
    optional ones, all in the source's order.

    None when the required ones alone do not fit, which is a tool's only way
    out of the pool.
    """
    if len(params) <= cap:
        return params
    keep = {i for i, p in enumerate(params) if p["required"]}
    if len(keep) > cap:
        return None
    for i in range(len(params)):
        if len(keep) >= cap:
            break
        keep.add(i)
    return [p for i, p in enumerate(params) if i in keep]


def _side(name: str) -> int:
    """Stable 0-99 bucket for a tool name.

    Not `hash()`: that is salted per process, so eight parallel generator
    shards would disagree about which side of the split a tool is on.
    """
    return int.from_bytes(
        hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(),
        "big") % 100


def tools_of_row(row: dict) -> List[dict]:
    """World-form tools from one converted row's serialized context."""
    ctx = row["context"]
    field_name = {f["sym"]: f["name"] for f in ctx["fields"]}
    out = []
    for t in ctx["tools"]:
        out.append({
            "name": t["name"],
            "desc": one_line(t["desc"]),
            "params": [{"name": field_name[p["sym"]], "type": p["type"],
                        "required": p.get("required", True),
                        "desc": one_line(p.get("desc", ""))}
                       for p in t["params"]],
            "returns": t.get("returns"),
            "effects": list(t["effects"]),
            # convert_open.py:277 — the stub returns "[kind: arg]" with no
            # external_url set, which is what makes a distractor callable.
            "impl": {"op": "external", "kind": t["name"]},
        })
    return out


def load_pool(holdout: bool = False, root: Path | None = None,
              holdout_pct: int = HOLDOUT_PCT) -> List[dict]:
    """Distinct imported tools on one side of the train/eval split.

    Deduped by name, the first occurrence that fits winning: 41% of imported
    tool occurrences carry a name that appears elsewhere with a different
    signature (`search_books` has 59), and the collision policy is the one
    `harness/crowding.py` already uses — drop, never rename. A name whose
    first occurrence is too wide for `fit_params` can therefore still enter
    under a later, narrower signature; file order is fixed, so which one it
    is does not vary between runs.
    """
    root = Path(root) if root else OPEN_PAIRS
    key = (str(root), holdout, holdout_pct)
    if key in _CACHE:
        return _CACHE[key]
    by_name: Dict[str, dict] = {}
    for fname in SOURCES:
        path = root / fname
        if not path.exists():
            raise SystemExit(f"open pool source missing: {path}")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                for tool in tools_of_row(json.loads(line)):
                    if tool["name"] in by_name:
                        continue
                    if (_side(tool["name"]) < holdout_pct) != holdout:
                        continue
                    params = fit_params(tool["params"])
                    if params is None:
                        continue
                    tool["params"] = params
                    by_name[tool["name"]] = tool
    pool = [by_name[n] for n in sorted(by_name)]
    _CACHE[key] = pool
    return pool


def main() -> None:
    import collections
    for holdout in (False, True):
        pool = load_pool(holdout=holdout)
        params = collections.Counter(p["type"] for t in pool for p in t["params"])
        effects = collections.Counter(e for t in pool for e in t["effects"])
        side = "holdout" if holdout else "train"
        print(f"{side:8s} {len(pool):5d} tools  "
              f"params {dict(params)}  effects {dict(effects)}")
    both = len(load_pool(False)) + len(load_pool(True))
    print(f"total    {both:5d} distinct names over {len(SOURCES)} sources")


if __name__ == "__main__":
    main()
