"""Can the request tell each decoyed call's tool from its decoys?
(.claude/plans/decidable-decoys.md, "What the $0 check found").

Themes give decoys to all nine tool slots, but only the four flip slots
(delete, flag, re-parent, send) carry per-tool request wording. A list,
fetch or set-status decoy differs from the real tool only in where the
record lives, which no request mentions, so such a call is undecidable by
construction. A flip-slot call is decidable when `domains.request_fits`
says the request fits the called tool and none of the decoys in the row.

  python results/logs/decoy_decidable.py data/s6_holdout_decoy.jsonl data/s6_flip.jsonl ...

`label(row)` is the importable form: 'decidable', 'undecidable', or None
for a row with no decoyed call (or a world without a theme).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.gen import domains  # noqa: E402

_loaded = False


def _themes() -> dict:
    global _loaded
    if not _loaded:
        domains.register_domains(str(ROOT / "data/gen/themes"))
        _loaded = True
    return domains.THEMES


def calls(row: dict) -> list[tuple[str, str, bool]]:
    """(tool name, slot, decidable) for every called tool with a decoy in the row."""
    th = _themes().get(row["world"])
    if th is None:
        return []
    slots = {}
    for slot, spec in th["tools"].items():
        sibs = [(spec["name"], spec)] + [(d["name"], d) for d in spec.get("decoys") or []]
        for nm, s in sibs:
            slots[nm] = (slot, s, sibs)
    present = {t["name"] for t in row["context"]["tools"]}
    out = []
    for c in row["reference"]["call_log"]:
        if c["name"] not in slots:
            continue
        slot, spec, sibs = slots[c["name"]]
        decoys = [s for nm, s in sibs if nm != c["name"] and nm in present]
        if not decoys:
            continue
        ok = (slot in domains.FLIP_VERBS
              and domains.request_fits(row["request"], slot, spec)
              and not any(domains.request_fits(row["request"], slot, s) for s in decoys))
        out.append((c["name"], slot, ok))
    return out


def label(row: dict) -> str | None:
    cs = calls(row)
    if not cs:
        return None
    return "decidable" if all(ok for _, _, ok in cs) else "undecidable"


def main() -> int:
    for fn in sys.argv[1:]:
        rows = [json.loads(line) for line in open(fn)]
        labels = Counter(label(r) for r in rows if "opaque-names" not in r["tags"])
        slots = Counter()
        for r in rows:
            if "opaque-names" in r["tags"]:
                continue
            for _, slot, ok in calls(r):
                slots[(slot in domains.FLIP_VERBS, ok)] += 1
        n = sum(v for k, v in labels.items() if k)
        print(f"{Path(fn).name}: rows with a decoyed call {n}, decidable {labels['decidable']} "
              f"({labels['decidable'] / max(n, 1):.0%}), undecidable {labels['undecidable']}")
        tot = sum(slots.values())
        for (flip, ok), v in sorted(slots.items()):
            print(f"   {'flip slot' if flip else 'list/fetch/set-status'} call, "
                  f"{'decidable' if ok else 'undecidable'}: {v} ({v / max(tot, 1):.0%} of decoyed calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
