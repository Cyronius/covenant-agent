"""Open tool-calling datasets -> Agent Core training pairs (plan
s2-consolidated-program §B3).

Sources (all Apache-2.0, ungated): glaiveai/glaive-function-calling-v2,
NousResearch/hermes-function-calling-v1, Team-ACE/ToolACE. What we take:
the REQUEST English and the TOOL SCHEMAS. Each example becomes its own
one-world context: every function is a tool with an `external` stub impl
(no state to mutate), each call argument becomes a typed constant, and the
reference program is the calls in order — `CALL @fn $i $j`. That is a
mechanical derivation, not the dataset's label as such: it is re-verified
through our typechecker and executed in our sandbox (external stub) before
it is kept, and anything whose values do not fit the IR (floats, arrays,
objects, values not present as literals) is dropped rather than coerced.
Glaive's "I don't have the capability" turns become `ABORT UNSUPPORTED`.

Only first-turn, single-request examples are used (clarification dialogues
are skipped): one request, the tools it saw, what a frontier model called.

  python -m data.gen.convert_open --source glaive --limit 2000 --out data/open_pairs/glaive.jsonl
  python -m data.gen.convert_open --source hermes --out data/open_pairs/hermes.jsonl
  python -m data.gen.convert_open --source toolace --out data/open_pairs/toolace.jsonl

Output rows carry both the SFT `messages` (make_sft.py's format, same SYSTEM
prompt as eval) and the task fields (context/state/reference) so they can be
audited and, with a registry entry, evaluated.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baselines.qwen.run_a import SYSTEM, build_prompt  # noqa: E402
from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, sandbox_from_context, serialize_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402

EFFECT_BY_VERB = [
    (re.compile(r"^(get|list|search|find|fetch|retrieve|check|lookup|look_up|calculate|compute|convert|translate|generate|analy[sz]e|query|read|view|show|count|estimate|predict|recommend|suggest|validate|verify)", re.I), "READ"),
    (re.compile(r"^(send|email|notify|message|post|publish|share|invite|alert|text|sms)", re.I), "SEND"),
    (re.compile(r"^(delete|remove|drop|purge|erase|cancel)", re.I), "DELETE"),
    (re.compile(r"^(pay|charge|refund|transfer|purchase|buy|order|book|reserve|donate|withdraw|deposit)", re.I), "PAY"),
]
REFUSAL = re.compile(r"(don't|do not|cannot|can't|unable to)[^.]{0,80}(capability|ability|perform|assist with|help with|book|order)", re.I)


def effect_for(name: str, desc: str = "") -> str:
    """Effect from the tool's name, else from the first verb of its description
    ("Retrieve detailed information…" is a READ even if the name is a noun)."""
    for rx, eff in EFFECT_BY_VERB:
        if rx.search(name):
            return eff
    first = re.sub(r"^(returns?|this (tool|function|api) )", "", desc.strip(), flags=re.I).strip()
    first = re.sub(r"^(returns?)", "get", first, flags=re.I)
    for rx, eff in EFFECT_BY_VERB:
        if rx.search(first):
            return eff
    if re.match(r"^(returns?|provides?|gives?|shows?|displays?|fetches|reads?)", desc.strip(), re.I):
        return "READ"
    return "WRITE"


def ir_type(schema: dict, value=None):
    t = (schema or {}).get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t in ("string", "str"):
        return "STR"
    if t in ("integer", "int"):
        return "INT"
    if t in ("number", "float"):
        # the IR has no float; keep only integral values
        if value is None or (isinstance(value, (int, float)) and float(value).is_integer()):
            return "INT"
        return None
    if t in ("boolean", "bool"):
        return "BOOL"
    return None


def coerce(value, ty):
    if ty == "STR":
        return str(value) if not isinstance(value, (dict, list)) else None
    if ty == "INT":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and float(value).is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
        return None
    if ty == "BOOL":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        return None
    return None


# ------------------------------------------------------------- parsers
def parse_glaive(row):
    """-> (tools, request, calls, refusal) or None."""
    sys_text = row["system"]
    tools = []
    for m in re.finditer(r"\{\s*\"name\"", sys_text):
        # each function is a pretty-printed JSON object; find its matching brace
        depth, i = 0, m.start()
        for j in range(m.start(), len(sys_text)):
            if sys_text[j] == "{":
                depth += 1
            elif sys_text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        tools.append(json.loads(sys_text[i:j + 1]))
                    except ValueError:
                        pass
                    break
    if not tools:
        return None
    chat = row["chat"]
    turns = re.split(r"\n\n\n(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat.strip())
    if len(turns) < 2 or not turns[0].startswith("USER:") or not turns[1].startswith("ASSISTANT:"):
        return None
    request = turns[0][len("USER:"):].strip()
    reply = turns[1][len("ASSISTANT:"):].replace("<|endoftext|>", "").strip()
    calls = []
    for fc in re.finditer(r"<functioncall>\s*(\{.*?\})\s*(?=<functioncall>|$)", reply, re.S):
        raw = fc.group(1)
        try:
            obj = json.loads(raw)
        except ValueError:
            # arguments are a single-quoted JSON string in this set
            try:
                obj = ast.literal_eval(raw)
            except Exception:
                return None
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        calls.append({"name": obj.get("name"), "args": args or {}})
    if not calls:
        if REFUSAL.search(reply):
            return tools, request, [], True
        return None  # a clarification question or plain chat
    return tools, request, calls, False


def parse_hermes(row):
    tools = row["tools"]
    tools = json.loads(tools) if isinstance(tools, str) else tools
    tools = [t.get("function", t) for t in tools]
    conv = row["conversations"]
    conv = json.loads(conv) if isinstance(conv, str) else conv
    msgs = [m for m in conv if m.get("from") in ("human", "gpt")]
    if len(msgs) < 2 or msgs[0]["from"] != "human" or msgs[1]["from"] != "gpt":
        return None
    request = msgs[0]["value"].strip()
    calls = []
    for tc in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", msgs[1]["value"], re.S):
        try:
            obj = json.loads(tc.group(1))
        except ValueError:
            return None
        calls.append({"name": obj.get("name"), "args": obj.get("arguments") or {}})
    if not calls:
        return None
    return tools, request, calls, False


_PY_CALL = re.compile(r"([A-Za-z_][\w .\-/]*?)\((.*?)\)(?=,\s*[A-Za-z_]|\]$)", re.S)


def parse_toolace(row):
    m = re.search(r"Here is a list of functions in JSON format that you can invoke:\s*(\[.*\])\.\s*\n", row["system"], re.S)
    if not m:
        m = re.search(r"(\[\{\"name\".*\}\])", row["system"], re.S)
    if not m:
        return None
    try:
        tools = json.loads(m.group(1))
    except ValueError:
        return None
    conv = row["conversations"]
    conv = json.loads(conv) if isinstance(conv, str) else conv
    if len(conv) < 2 or conv[0].get("from") != "user" or conv[1].get("from") != "assistant":
        return None
    request = conv[0]["value"].strip()
    reply = conv[1]["value"].strip()
    if not (reply.startswith("[") and reply.endswith("]")):
        return None
    calls = []
    for cm in _PY_CALL.finditer(reply):
        name, argtext = cm.group(1).strip(), cm.group(2)
        args = {}
        try:
            tree = ast.parse(f"f({argtext})", mode="eval")
            for kw in tree.body.keywords:
                args[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return None
        calls.append({"name": name, "args": args})
    if not calls:
        return None
    return tools, request, calls, False


PARSERS = {"glaive": parse_glaive, "hermes": parse_hermes, "toolace": parse_toolace}
LOADERS = {
    "glaive": ("glaiveai/glaive-function-calling-v2", {}),
    "hermes": ("NousResearch/hermes-function-calling-v1", {"data_files": "func-calling.json"}),
    "toolace": ("Team-ACE/ToolACE", {}),
}


# ------------------------------------------------------------- conversion
def to_world_and_program(tools, request, calls, refusal, rng):
    """Build a one-example world + constants + authoring-form program.
    Returns (world, constants, segments, expected_status) or a reason string."""
    def norm(name):
        n = re.sub(r"\W", "_", str(name)).strip("_")
        return ("f_" + n) if not n or n[0].isdigit() else n
    by_name = {}
    for t in tools:
        name = t.get("name") if isinstance(t, dict) else None
        if not isinstance(name, str) or not name.strip():
            return "bad tool name"
        by_name[name] = t
    if len({norm(n) for n in by_name}) != len(by_name):
        return "bad tool name"
    used = {c["name"] for c in calls}
    if not used <= set(by_name):
        return "call to undeclared tool"
    world_tools = []
    constants = []
    const_index = {}
    lines = []
    for t in tools:
        schema = (t.get("parameters") or {})
        if not isinstance(schema, dict):
            return "bad schema"
        props = schema.get("properties") or {}
        if not isinstance(props, dict) or not all(isinstance(v, dict) for v in props.values()):
            return "bad schema"
        required = set(schema.get("required") or [])
        # param order: called args first (positional CALL), then the rest
        call_args = next((c["args"] for c in calls if c["name"] == t["name"]), {})
        order = [p for p in call_args if p in props] + [p for p in props if p not in call_args]
        params = []
        for p in order:
            ty = ir_type(props[p], call_args.get(p))
            if ty is None:
                if p in call_args or p in required:
                    return "unsupported param type (float/array/object)"
                continue  # optional param of an unsupported type: leave it out
            params.append({"name": p, "type": ty,
                           "desc": str(props[p].get("description") or p)[:120],
                           "required": p in required})
        world_tools.append({
            "name": norm(t["name"]),
            "desc": str(t.get("description") or t["name"])[:300],
            "params": params, "returns": None,
            "effects": [effect_for(t["name"], str(t.get("description") or ""))],
            "impl": {"op": "external", "kind": norm(t["name"])},
        })
    if refusal:
        return ({"tools": world_tools}, constants, ["ABORT UNSUPPORTED\n"], "aborted")
    for c in calls:
        wt = next(w for w in world_tools if w["name"] == norm(c["name"]))
        refs = []
        given = c["args"]
        # positional: every param up to the last given one must be present
        last = max((i for i, p in enumerate(wt["params"]) if p["name"] in given), default=-1)
        for i, p in enumerate(wt["params"][:last + 1]):
            if p["name"] not in given:
                return f"gap in positional args ({p['name']})"
            v = coerce(given[p["name"]], p["type"])
            if v is None:
                return f"value not representable ({p['name']})"
            key = (p["type"], json.dumps(v))
            if key not in const_index:
                const_index[key] = len(constants)
                constants.append({"type": p["type"], "value": v,
                                  "desc": f"{p['desc'][:60]}: {v}"})
            refs.append(f"${const_index[key]}")
        for p in wt["params"][last + 1:]:
            if p["required"]:
                return f"required param not given ({p['name']})"
        lines.append(f"CALL @{wt['name']}" + ("".join(" " + r for r in refs)))
    lines.append("STOP")
    return ({"tools": world_tools}, constants, ["\n".join(lines) + "\n"], "ok")


def make_pair(source, idx, world, constants, segments, expected_status, request, rng):
    world = dict(world, name=f"{source}_{idx}", now=1_760_000_000,
                 entities={}, enums={}, default_state={"entities": {}, "outbox": [], "payments": []})
    ctx, sandbox_ctx = build_context(world, constants, rng)
    resolved = [resolve(s, ctx) for s in segments]
    res = build(resolved[0], ctx)
    if not res.compile_ok:
        return None, "typecheck: " + "; ".join(res.rendered_diagnostics()[:2])
    sres = run_sandbox({"js": res.js, "state": world["default_state"], "tools": sandbox_ctx["tools"],
                        "fields": sandbox_ctx["fields"], "constants": sandbox_ctx["constants"],
                        "now": world["now"], "approval": True, "error_injection": [],
                        "initial_registers": {}})
    if sres.get("status") != expected_status:
        return None, f"sandbox: {sres.get('status')} {sres.get('error')}"
    input_text = serialize_context(request, ctx)
    return {
        "id": f"open_{source}_{idx}", "source": source, "request": request,
        "context": ctx.to_json(), "input_text": input_text,
        "reference": {"segments": resolved, "authoring": segments, "calls": sres.get("calls", [])},
        "expected_status": expected_status,
        "effects": res.static_effects,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(input_text, None, [])},
            {"role": "assistant", "content": resolved[0].strip()},
        ],
        "tags": ["open", source] + (["abort"] if expected_status == "aborted" else []),
    }, None


def main() -> None:
    ap = argparse.ArgumentParser(prog="data.gen.convert_open")
    ap.add_argument("--source", choices=sorted(PARSERS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    from datasets import load_dataset
    path, kw = LOADERS[args.source]
    ds = load_dataset(path, split="train", **kw)
    parse = PARSERS[args.source]
    rng = random.Random(args.seed)
    reasons = {}
    n_ok = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            parsed = parse(row)
            if parsed is None:
                reasons["parse: not a first-turn call/refusal"] = reasons.get("parse: not a first-turn call/refusal", 0) + 1
                continue
            tools, request, calls, refusal = parsed
            if not (3 <= len(request.split()) <= 200):
                reasons["request length"] = reasons.get("request length", 0) + 1
                continue
            built = to_world_and_program(tools, request, calls, refusal, rng)
            if isinstance(built, str):
                key = re.sub(r"\(.*\)", "", built)
                reasons[key] = reasons.get(key, 0) + 1
                continue
            world, constants, segments, status = built
            try:
                pair, err = make_pair(args.source, i, world, constants, segments, status, request,
                                      random.Random(args.seed * 100003 + i))
            except Exception as e:  # noqa: BLE001 — one bad row must not stop the run
                pair, err = None, f"exception: {type(e).__name__}"
            if pair is None:
                key = err.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            f.write(json.dumps(pair) + "\n")
            n_ok += 1
    total = n_ok + sum(reasons.values())
    print(f"{args.source}: {n_ok}/{total} converted -> {out}")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {v:6d}  {k}")


if __name__ == "__main__":
    main()
