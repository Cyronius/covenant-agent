"""Can the typed signature alone pick the tool? Run this before believing any
grounding number.

A tool line is `T8 (I:policyholder=F2 S=F17) -> - [SEND] :: Place a call to a
policyholder.` Everything before the `::` is the signature. If, in a task, no
other tool shares the reference tool's signature, then a model that maps the
request to a *type shape* and looks up the unique tool with that shape is right
without reading a single description. The task never poses the discrimination a
description exists to resolve.

That is not a hypothesis about any model. It is a property of the corpus, and
it is measurable from reference programs alone:

    python -m harness.signature_uniqueness data/s5_plain.jsonl
    python -m harness.signature_uniqueness --schema data/schemas/coursebuilder_tools.json

Three columns, each a stricter claim about what the suite can teach or test:

  full        unique by the whole signature, per-request field symbols included
  stripped    unique by types, arity and effects, field symbols removed
  effect      unique by property set alone (mutates/irreversible/external)

`full` at 100% means the suite cannot distinguish tool grounding from type
inference, and no accuracy number taken on it supports a claim about reading
descriptions or schemas. See `results/GROUNDING.md` for the measured table and
what it invalidates.

The suites that can carry a grounding claim are the ones built by
`harness/decoys.py`, where siblings share a signature by construction.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from harness.context import TaskContext, serialize_context

# per-request symbols carry no type information the model can generalize from
_FIELD_SYM = re.compile(r"=[A-Z]+\d+")
_TOOL_SYM = re.compile(r"T\d+")
# spec 0.7.0 §7: the bracket holds a compact property code (M/!/X in a fixed
# order, "" for none of the three) rather than comma-joined effect words.
_EFFECTS = re.compile(r"\[([A-Z!]*)\]")


def strip_field_symbols(sig: str) -> str:
    return _FIELD_SYM.sub("", sig)


def effect_class(sig: str) -> str:
    m = _EFFECTS.search(sig)
    return m.group(1) if m else "?"


def tool_lines(context_text: str) -> dict[str, tuple[str, str]]:
    """symbol -> (signature, description), from a serialized context."""
    tools = {}
    for line in context_text.splitlines():
        if " :: " not in line:
            continue
        head, desc = line.split(" :: ", 1)
        parts = head.split(" ", 1)
        if len(parts) == 2 and _TOOL_SYM.fullmatch(parts[0]):
            tools[parts[0]] = (parts[1], desc)
    return tools


def called_tools(reference) -> list[str]:
    """Tool symbols on CALL lines of a reference program (all segments)."""
    if isinstance(reference, dict):
        text = "\n".join(reference.get("segments") or [])
    else:
        text = str(reference or "")
    out = []
    for line in text.splitlines():
        p = line.strip().split()
        if len(p) >= 2 and p[0] == "CALL" and _TOOL_SYM.fullmatch(p[1]):
            out.append(p[1])
    return out


def context_text(row: dict) -> str:
    """What the model is shown. `input_text` is the generator's own serialized
    context, so prefer it: it is faithful by construction and avoids
    re-serializing. Fall back for rows written before it existed."""
    text = row.get("input_text")
    if text:
        return text
    return serialize_context(row.get("request", ""),
                             TaskContext.from_json(row["context"]))


def measure_tasks(path: Path, limit: int | None = None) -> dict | None:
    n = full = stripped = eff = tasks = tools_seen = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and tasks >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "context" not in row or "reference" not in row:
                continue
            tasks += 1
            tools = tool_lines(context_text(row))
            tools_seen += len(tools)
            sigs = [s for s, _ in tools.values()]
            for sym in called_tools(row["reference"]):
                if sym not in tools:
                    continue
                n += 1
                sig = tools[sym][0]
                full += sum(1 for s in sigs if s == sig) == 1
                stripped += sum(1 for s in sigs
                                if strip_field_symbols(s)
                                == strip_field_symbols(sig)) == 1
                ec = effect_class(sig)
                eff += sum(1 for s in sigs if effect_class(s) == ec) == 1
    if n == 0:
        return None
    return {"tasks": tasks, "calls": n, "tools_per_task": tools_seen / max(tasks, 1),
            "full": full / n, "stripped": stripped / n, "effect": eff / n}


def _agent_core_sig(tool: dict) -> str:
    """A tool declared in Agent Core form (params/returns/effects)."""
    def ptype(p):
        return str(p.get("type") or p.get("t") or p) if isinstance(p, dict) else str(p)
    params = ",".join(sorted(ptype(p) for p in (tool.get("params") or [])))
    return (f"({params})->{tool.get('returns')} "
            f"[{','.join(tool.get('effects') or [])}]")


def _json_schema_sig(tool: dict) -> str:
    """A tool declared as an LLM function schema (JSON Schema parameters)."""
    props = (tool.get("parameters") or {}).get("properties") or {}
    required = set((tool.get("parameters") or {}).get("required") or [])

    def ptype(v):
        t = v.get("type", "?")
        return "|".join(sorted(t)) if isinstance(t, list) else str(t)
    req = ",".join(sorted(ptype(props[k]) for k in props if k in required))
    opt = ",".join(sorted(ptype(props[k]) for k in props if k not in required))
    return f"({req})" + (f"[{opt}]" if opt else "")


def measure_schema(path: Path) -> list[tuple[str, dict]]:
    """A raw tool-schema file: every list of tool dicts it holds, measured.

    Two declaration forms are in the tree. `params`/`returns`/`effects` is the
    Agent Core form and is directly comparable to a corpus number; JSON Schema
    `parameters` carries only string/number/object, which is coarser than a
    typed signature, so its uniqueness is a floor rather than the figure an
    Agent Core rendering of the same tools would give.
    """
    blob = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for key, value in (blob.items() if isinstance(blob, dict) else [("tools", blob)]):
        items = list(value.values()) if isinstance(value, dict) else value
        if not isinstance(items, list):
            continue
        tools = [t for t in items if isinstance(t, dict) and "name" in t]
        if not tools:
            continue
        agent_core = any("params" in t or "returns" in t for t in tools)
        sig = _agent_core_sig if agent_core else _json_schema_sig
        counts: dict[str, int] = {}
        for t in tools:
            counts[sig(t)] = counts.get(sig(t), 0) + 1
        colliding = sum(v for v in counts.values() if v > 1)
        groups = sorted(((v, s) for s, v in counts.items() if v > 1), reverse=True)
        out.append((key, {
            "form": "agent-core" if agent_core else "json-schema",
            "tools": len(tools), "unique": (len(tools) - colliding) / len(tools),
            "largest_group": max(counts.values()),
            "examples": [(s, [t["name"] for t in tools if sig(t) == s])
                         for _, s in groups[:3]]}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--schema", action="store_true",
                    help="the paths are raw tool-schema JSON, not task JSONL")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N tasks per file (the corpora are large)")
    ap.add_argument("--examples", action="store_true",
                    help="with --schema, print the colliding groups")
    args = ap.parse_args()

    if args.schema:
        for p in args.paths:
            for key, r in measure_schema(Path(p)):
                print(f"{Path(p).name}:{key:16s} {r['form']:11s} {r['tools']:4d} tools  "
                      f"{r['unique']:6.1%} unique  largest group {r['largest_group']:2d}")
                if args.examples:
                    for sig, names in r["examples"]:
                        print(f"    x{len(names):<2d} {sig}")
                        print(f"        {', '.join(names[:7])}")
        return

    print(f"{'file':38s} {'tasks':>6s} {'calls':>7s} {'tools':>6s} "
          f"{'full':>7s} {'stripped':>9s} {'effect':>7s}")
    for p in args.paths:
        path = Path(p)
        r = measure_tasks(path, args.limit) if path.exists() else None
        if r is None:
            print(f"{path.name:38s}  {'missing' if not path.exists() else 'no reference CALLs'}")
            continue
        print(f"{path.name:38s} {r['tasks']:6d} {r['calls']:7d} "
              f"{r['tools_per_task']:6.1f} {r['full']:6.1%} "
              f"{r['stripped']:8.1%} {r['effect']:6.1%}")


if __name__ == "__main__":
    main()
