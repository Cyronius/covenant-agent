"""R15: what is left of the "wrong entity's tool" failure after R14.

fail_kinds.py calls a failure "tool: another signature" when the first slot
where the program departs from the reference holds a tool whose parameter and
return types differ from the reference tool's. R14 read that as "another
entity's tool". This splits it by what the two tools are actually about:

  same entity, other action    both tools act on the same entity type
  other entity: ...            the tools' entity types differ, and then
    repeats an earlier call      the tool was already called earlier
    a tool the reference ...     the reference calls it too, later (order)
    named in the request         the request names its entity
    not named in the request     it does not

A tool's entity is its return type's object, else its first ID parameter's.
Then the per-world rate of "other entity" failures for every model scored on
the S6 plain exam, to see whether the same worlds fail across models.

  cd models/tiny && python ../../results/logs/residual_r15.py
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from fail_kinds import kind, signature  # noqa: E402

ROWS = {r["id"]: r for r in pickle.load(open("data_cache_s6off/rows.pkl", "rb"))["holdout"]}
RUN3 = "runs/pod_s6split/out/s6off_A0_plain"
MODELS = {
    "run 3: A0, S6 options-off": RUN3,
    "run 1: A0, S5": "runs/pod_s6split/out/s5g_A0_s6plain",
    "run 2: SPt, S5": "runs/pod_s6split/out/s5g_SPt_s6plain",
    "R13: A0, S6": "runs/pod_planner/out/s6_A0_holdout_plain",
    "R13: SPt, S6": "runs/pod_planner/out/s6_SPt_holdout_plain",
}


def entity(t: dict) -> str | None:
    m = re.search(r"OBJ:(\w+)", t.get("returns") or "")
    if m:
        return m.group(1)
    for p in t.get("params") or []:
        m = re.match(r"ID:(\w+)", p.get("type") or "")
        if m:
            return m.group(1)
    return None


def named(ent: str | None, request: str) -> bool:
    if not ent:
        return False
    words = set(re.findall(r"[a-z]+", request.lower()))
    head = ent.split("_")[-1]
    return bool(words & {head, head + "s", head + "es", head[:-1] + "ies"})


def tool_failures(run: str):
    """(row, generated sym, reference sym, prefix, reference syms) for every failed
    task whose first departure is a tool of another signature."""
    gens = {r["task_id"]: r for r in map(json.loads, open(run + ".jsonl", encoding="utf-8"))}
    for s in json.load(open(run + ".score.json"))["rows"]:
        tid = s["task_id"].replace("+s6", "")
        row = ROWS.get(tid.split("#")[0]) or ROWS.get(tid)
        if row is None or s["goal"]:
            continue
        a, b = gens[s["task_id"]]["canvas_tokens"], gens[s["task_id"]]["reference_tokens"]
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if i is None or kind(a[i]) != "tool" or kind(b[i]) != "tool":
            continue
        tools = {t["sym"]: t for t in row["context"]["tools"]}
        if a[i] not in tools or b[i] not in tools or signature(tools[a[i]]) == signature(tools[b[i]]):
            continue
        yield s, row, tools[a[i]], tools[b[i]], a[:i], {t for t in b if kind(t) == "tool"}


def category(row, gen, ref, prefix, ref_syms) -> str:
    if entity(gen) == entity(ref):
        return "same entity, other action"
    if gen["sym"] in prefix:
        return "other entity: repeats an earlier call"
    if gen["sym"] in ref_syms:
        return "other entity: a tool the reference calls later (order)"
    if named(entity(gen), row["request"]):
        return "other entity: named in the request"
    return "other entity: not named in the request"


def main():
    n = len(json.load(open(RUN3 + ".score.json"))["rows"])
    cats, compiled, pairs, by = Counter(), Counter(), Counter(), defaultdict(Counter)
    for s, row, gen, ref, prefix, ref_syms in tool_failures(RUN3):
        c = category(row, gen, ref, prefix, ref_syms)
        cats[c] += 1
        compiled[c] += s["compile"]
        by[c][row["provenance"].get("recipe")] += 1
        if not c.startswith("same"):
            pairs[(entity(ref), entity(gen))] += 1
    total = sum(cats.values())
    print(f"run 3, S6 plain exam: {total} tool-of-another-signature failures ({total / n:.1%} of tasks)")
    for c, k in cats.most_common():
        print(f"  {k:4d}  {k / n:5.1%} of tasks  compile {compiled[c]:3d}  {c}")
        print(f"        recipes {by[c].most_common(4)}")
    print("\nother entity: (reference entity, generated entity)", pairs.most_common(10))

    table = {}
    for name, run in MODELS.items():
        tot, bad = Counter(), Counter()
        for s in json.load(open(run + ".score.json"))["rows"]:
            tid = s["task_id"].replace("+s6", "")
            row = ROWS.get(tid.split("#")[0]) or ROWS.get(tid)
            if row is not None:
                tot[row["world"]] += 1
        for s, row, gen, ref, prefix, ref_syms in tool_failures(run):
            if entity(gen) != entity(ref):
                bad[row["world"]] += 1
        table[name] = (tot, bad)
        print(f"{name:28s} other-entity failures {sum(bad.values()) / sum(tot.values()):5.1%} of tasks")
    tot3, bad3 = table["run 3: A0, S6 options-off"]
    worlds = sorted(tot3, key=lambda w: -bad3[w] / tot3[w])
    print("\nper world, share of the world's tasks:")
    print(f"{'world':20s} {'n':>4s}  " + "  ".join(f"{k[:12]:>12s}" for k in table))
    for w in worlds[:10]:
        print(f"{w:20s} {tot3[w]:4d}  " + "  ".join(f"{b[w] / max(t[w], 1):12.1%}" for t, b in table.values()))
    rest = worlds[10:]
    print(f"{'other ' + str(len(rest)):20s} {sum(tot3[w] for w in rest):4d}  " + "  ".join(
        f"{sum(b[w] for w in rest) / max(sum(t[w] for w in rest), 1):12.1%}" for t, b in table.values()))


if __name__ == "__main__":
    main()
