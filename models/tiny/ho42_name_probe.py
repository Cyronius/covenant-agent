"""Is the margin on the decoyed half description-reading, or name-reading?

`ho42_report.py` shows the model beating the shape-only null on the decoyed
holdout. That says it reads something the type signature does not carry -- but
a tool line carries two such things, the NAME and the DESCRIPTION, and the
grounding claim people will take from it is about the description.

`harness/decoys.py` separates them for free. A `--decoy-nonsense` share of
decoys are named `foo17` / `operation_93` / `do_thing_4`, with nothing in the
name pointing at what the tool does; the rest get plausible `verb_entity`
names built the same way the real tool's name is. So, per reference CALL, split
by what the reference tool is competing against:

  adversarial  every signature sibling is nonsense-named. The real tool is the
               only meaningful NAME in the group, so a name reader scores high
               here without reading a description.
  plausible    at least one sibling has a real verb_entity name. A name reader
               has no edge; only the description separates them.

WHAT THIS DOES NOT SEPARATE, stated because the obvious reading is wrong.
Both groups leave the name able to carry the choice. A plausible sibling is
`snooze_card` beside a real `archive_card`: different verbs, both meaningful,
and a request saying "archive this card" is matched by the name alone. So a
small gap does NOT show the model is reading descriptions -- it shows only
that the model does not *need* its competitors to be obviously fake, which
rules out one failure mode and not the interesting one.

The instrument that does separate them is `ablate_desc.py`: it permutes the
descriptions among tool lines, leaving every symbol, signature, effect and
NAME exactly where it was, and regenerates. A drop is description reading; no
change means the margin is name and structure. It needs a GPU but no
training, so it is one generation pass over the decoyed holdout.

    python ho42_name_probe.py out/ho42_ar_s0_holdout_decoy.jsonl [cache]
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness.decoys import NONSENSE_NAMES          # noqa: E402

CALL = re.compile(r"^\s*CALL\s+(T\d+)", re.M)


def called(text: str) -> dict[int, str]:
    """line index -> tool symbol, for every CALL line."""
    out = {}
    for i, line in enumerate(text.splitlines()):
        m = CALL.match(line)
        if m:
            out[i] = m.group(1)
    return out


def main() -> None:
    gen_path = Path(sys.argv[1])
    cache = Path(sys.argv[2] if len(sys.argv) > 2 else "data_cache_ho42")
    split = "holdout"
    rows = {r["id"]: r for r in pickle.load(open(cache / "rows.pkl", "rb"))[split]}

    tally: dict[str, Counter] = {"adversarial": Counter(), "plausible": Counter()}
    for line in gen_path.open(encoding="utf-8"):
        g = json.loads(line)
        row = rows.get(g["task_id"])
        if row is None:
            continue
        decoys = set((row.get("provenance") or {}).get("decoys") or [])
        if not decoys:
            continue
        tools = {t["sym"]: t for t in row["context"]["tools"]}

        def sig(t):
            return (tuple(p["type"] for p in t["params"]), t.get("returns"),
                    tuple(t["effects"]))

        ref_calls = called(g["reference"])
        gen_calls = called(g["program"])
        for idx, sym in ref_calls.items():
            real = tools.get(sym)
            if real is None:
                continue
            siblings = [t for s, t in tools.items()
                        if s != sym and sig(t) == sig(real) and t["name"] in decoys]
            if not siblings:
                continue                      # no discrimination posed here
            kind = ("adversarial"
                    if all(t["name"] in NONSENSE_NAMES for t in siblings)
                    else "plausible")
            tally[kind]["n"] += 1
            tally[kind]["hit"] += gen_calls.get(idx) == sym

    print(f"{gen_path.name}")
    print(f"  {'group':12s} {'calls':>6s} {'same_tool':>10s}")
    for kind in ("adversarial", "plausible"):
        c = tally[kind]
        if c["n"]:
            print(f"  {kind:12s} {c['n']:6d} {c['hit']/c['n']:10.1%}")
    a, p = tally["adversarial"], tally["plausible"]
    if a["n"] and p["n"]:
        gap = a["hit"] / a["n"] - p["hit"] / p["n"]
        print(f"\n  adversarial - plausible = {gap:+.1%}")
        print("  A large positive gap means the margin over the shape-only null")
        print("  is mostly the NAME, not the description. Near zero means the")
        print("  name is not carrying it.")


if __name__ == "__main__":
    main()
