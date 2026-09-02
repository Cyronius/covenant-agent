"""Real user requests from mobi agent sessions (plan s2-consolidated-program
§B1/§B2): extract, scrub, tag, and sample.

Input: data/real_sessions/pages/*.json — pages of ai.AgnoSessions pulled
through the database skill CLI (results/logs/pull_pages.sh). Gitignored:
raw sessions carry PII.

What a "turn" is: one user message inside one run, with what the (frontier)
agent did next — its tool calls (name + arguments) and its final text. The
tool calls are the closest thing to a routing label we get for free.

  python -m harness.real_requests extract           # -> data/real_sessions/turns.jsonl (gitignored)
  python -m harness.real_requests histogram         # shape histogram over turns
  python -m harness.real_requests inventory         # tool -> arg keys/types (coursebuilder world draft)
  python -m harness.real_requests sample --n 250    # -> data/real_sessions/eval_candidates.jsonl
                                                    #    (scrubbed; human review before it becomes
                                                    #     data/holdout/e_real_sessions.jsonl)
                                                    #    + eval_session_ids.json (frozen; B2 excludes)
  python -m harness.real_requests pool              # -> data/real_sessions/b2_pool.jsonl (mining pool,
                                                    #    disjoint from the eval sessions by session id)

Routing labels: the sessions carry no tool *schemas* (the run's `tools` key
is the execution log), so a turn is classed by what the frontier agent
called - read / write / generative / knowledge / none - plus a multi-step
flag and a content-bearing flag (a write whose arguments carry HTML or long
prose, i.e. text the frontier model authored, which the planner can only
route to a writer tool, not produce).

Scrubbing is regex-only (emails, phones, URLs, long digit runs) plus a
capitalized-name flag for the human pass; it is a first filter, not a
guarantee. Nothing leaves data/real_sessions/ without the review step.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data" / "real_sessions" / "pages"
TURNS = ROOT / "data" / "real_sessions" / "turns.jsonl"
CANDIDATES = ROOT / "data" / "real_sessions" / "eval_candidates.jsonl"
EVAL_SESSIONS = ROOT / "data" / "real_sessions" / "eval_session_ids.json"
POOL = ROOT / "data" / "real_sessions" / "b2_pool.jsonl"
INVENTORY = ROOT / "data" / "real_sessions" / "tool_inventory.json"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://\S+")
_PHONE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_DIGITS = re.compile(r"\b\d{6,}\b")
_CAPS = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Shape heuristics — verbs the *request* uses, not what the agent did. Coarse
# on purpose: the point is the histogram, and the human pass corrects tags.
SHAPES = [
    ("create", re.compile(r"\b(create|add|new|make|generate|build|write|draft|insert)\b", re.I)),
    ("update", re.compile(r"\b(update|change|edit|rename|set|move|assign|reassign|fix|replace|revise|adjust)\b", re.I)),
    ("delete", re.compile(r"\b(delete|remove|archive|clear|get rid of)\b", re.I)),
    ("query", re.compile(r"\b(list|show|what|which|how many|find|search|look up|who|where|tell me|summari[sz]e)\b", re.I)),
    ("send", re.compile(r"\b(send|message|email|notify|remind|share|publish|invite)\b", re.I)),
    ("prose", re.compile(r"\b(write|draft|rewrite|reword|summari[sz]e|describe|explain|caption|intro|paragraph|text)\b", re.I)),
    ("image", re.compile(r"\b(image|picture|photo|illustration|icon|logo|thumbnail)\b", re.I)),
    ("chitchat", re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|cool|great)\b[\s!.?]*$", re.I)),
]


# Routing classes by frontier tool name. Knowledge tools are retrieval over
# docs; generative tools take a free-text instruction the frontier model
# wrote. Everything else is CRUD over course structure.
# Knowledge tools are the ones with capitalised names (LEARNER_HOW_TO,
# WCAG_ACCESSIBILITY_KB, user-named notebooks such as How_it_s_made) plus
# the raw KB query tools; every CRUD tool is lowercase snake_case.
_KNOWLEDGE = re.compile(r"(^query_(knowledge_base|notebook)_raw$|[A-Z])")
_GENERATIVE = {"generate_image", "edit_image", "apply_to_lessons"}
_READ = re.compile(r"^(get_|list_|fetch_|search_|find_|read_|show_|lookup)", re.I)
_HTML = re.compile(r"<\w+[^>]*>")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _content_bearing(args) -> bool:
    """True when a call argument carries authored text (HTML or >160 chars)."""
    if isinstance(args, str):
        return bool(_HTML.search(args)) or len(args) > 160
    if isinstance(args, dict):
        return any(_content_bearing(v) for v in args.values())
    if isinstance(args, list):
        return any(_content_bearing(v) for v in args)
    return False


def routing(tool_calls: list) -> dict:
    """Class a turn by the frontier agent's calls.
    primary: none | knowledge | read | write | generative
    multi_step: >= 2 distinct tool names (knowledge lookups excluded)
    content_bearing: a write/generative call whose args carry authored text
    """
    names = [c["name"] for c in tool_calls if c.get("name")]
    if not names:
        return {"primary": "none", "multi_step": False, "content_bearing": False}
    kinds = set()
    content = False
    for c in tool_calls:
        n = c.get("name") or ""
        if _KNOWLEDGE.search(n):
            kinds.add("knowledge")
        elif n in _GENERATIVE:
            kinds.add("generative")
            content = True
        elif _READ.search(n):
            kinds.add("read")
        else:
            kinds.add("write")
            content = content or _content_bearing(c.get("args"))
    for primary in ("generative", "write", "read", "knowledge"):
        if primary in kinds:
            break
    distinct = {n for n in names if not _KNOWLEDGE.search(n)}
    return {"primary": primary, "multi_step": len(distinct) >= 2, "content_bearing": content}


def teachability(r: dict) -> str:
    """What the planner could learn from this turn, given a coursebuilder
    world plus the writer tool:
      full     - routing and arguments are literals the IR can carry
      routing  - the call sequence is expressible; the text/image content is
                 the writer tool's job (arguments carry authored content)
      abstain  - no tool call: the reference is ABORT or a plain answer
    """
    if r["primary"] == "none":
        return "abstain"
    if r["content_bearing"]:
        return "routing"
    return "full"


def scrub(text: str) -> tuple[str, list]:
    flags = []
    t = _EMAIL.sub("<email>", text)
    if t != text:
        flags.append("email")
    t2 = _URL.sub("<url>", t)
    if t2 != t:
        flags.append("url")
    t3 = _PHONE.sub("<phone>", t2)
    if t3 != t2:
        flags.append("phone")
    t4 = _DIGITS.sub("<number>", t3)
    if t4 != t3:
        flags.append("digits")
    if _CAPS.search(t4):
        flags.append("name?")  # for the human pass; not auto-replaced
    return t4, flags


def shapes(text: str) -> list:
    out = [name for name, rx in SHAPES if rx.search(text)]
    return out or ["other"]


def iter_sessions():
    for page in sorted(PAGES.glob("page_*.json")):
        try:
            d = json.loads(page.read_text(encoding="utf-8"))
        except ValueError:
            print("bad page", page)
            continue
        for rec in d.get("data", []):
            runs = rec.get("runs")
            if isinstance(runs, str):
                try:
                    runs = json.loads(runs)
                except ValueError:
                    runs = []
            rec["runs"] = runs or []
            yield rec


def extract_turns(rec: dict) -> list:
    turns = []
    for run in rec["runs"]:
        msgs = run.get("messages") or []
        user_text = None
        ic = (run.get("input") or {}).get("input_content")
        if isinstance(ic, list) and ic and isinstance(ic[0], dict):
            user_text = ic[0].get("content")
        elif isinstance(ic, str):
            user_text = ic
        calls = []
        assistant_text = run.get("content") if run.get("content_type") == "str" else None
        for m in msgs:
            if m.get("from_history"):
                continue
            if user_text is None and m.get("role") == "user" and isinstance(m.get("content"), str):
                user_text = m["content"]
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass
                calls.append({"name": fn.get("name"), "args": args})
        if not user_text or not isinstance(user_text, str):
            continue
        turns.append({
            "turn_id": hashlib.sha1(f"{rec['id']}:{run.get('run_id')}".encode()).hexdigest()[:12],
            "session_db_id": rec["id"],
            "account": rec.get("accountId"),
            "agent": rec.get("agentId"),
            "model": run.get("model") or rec.get("model"),
            "status": run.get("status"),
            "created_at": run.get("created_at") or rec.get("createdAt"),
            "user_text": user_text.strip(),
            "tool_calls": calls,
            "routing": routing(calls),
            "assistant_text": (assistant_text or "").strip()[:2000],
        })
    return turns


def cmd_extract(args) -> None:
    n_sessions = 0
    seen = set()
    with open(TURNS, "w", encoding="utf-8") as f:
        for rec in iter_sessions():
            n_sessions += 1
            for t in extract_turns(rec):
                key = (t["account"], t["user_text"].lower())
                t["duplicate"] = key in seen
                seen.add(key)
                f.write(json.dumps(t) + "\n")
    n = sum(1 for _ in open(TURNS, encoding="utf-8"))
    print(f"{n_sessions} sessions -> {n} turns -> {TURNS}")


def load_turns() -> list:
    return [json.loads(l) for l in open(TURNS, encoding="utf-8")]


def cmd_histogram(args) -> None:
    turns = [t for t in load_turns() if not t["duplicate"]]
    print(f"{len(turns)} distinct turns")
    by_agent = collections.Counter(t["agent"] for t in turns)
    print("by agent:", by_agent.most_common())
    shape = collections.Counter()
    for t in turns:
        for s in shapes(t["user_text"]):
            shape[s] += 1
    print("shape (multi-label):", shape.most_common())
    tools = collections.Counter(c["name"] for t in turns for c in t["tool_calls"])
    print("tools called:", tools.most_common(25))
    n_calls = collections.Counter(min(len(t["tool_calls"]), 5) for t in turns)
    print("tool calls per turn (5=5+):", sorted(n_calls.items()))
    words = collections.Counter(len(t["user_text"].split()) // 5 * 5 for t in turns)
    print("request length (words, bucketed by 5):", sorted(words.items())[:12])
    print()
    print("--- routing (what the frontier agent did) ---")
    prim = collections.Counter(t["routing"]["primary"] for t in turns)
    print("primary:", prim.most_common())
    ms = sum(t["routing"]["multi_step"] for t in turns)
    cb = sum(t["routing"]["content_bearing"] for t in turns)
    print(f"multi_step: {ms} ({100*ms/len(turns):.0f}%)   content_bearing: {cb} ({100*cb/len(turns):.0f}%)")
    teach = collections.Counter(teachability(t["routing"]) for t in turns)
    print("teachability:", teach.most_common())
    print("routing x request shape:")
    cross = collections.defaultdict(collections.Counter)
    for t in turns:
        for sh in shapes(t["user_text"]):
            cross[t["routing"]["primary"]][sh] += 1
    for k in ("read", "write", "generative", "knowledge", "none"):
        print(f"  {k:10s}", cross[k].most_common(6))
    print("routing by agent:")
    by_ag = collections.defaultdict(collections.Counter)
    for t in turns:
        by_ag[t["agent"]][t["routing"]["primary"]] += 1
    for a, c in sorted(by_ag.items(), key=lambda kv: -sum(kv[1].values()))[:6]:
        print(f"  {a:28s}", dict(c))
    none = [t for t in turns if t["routing"]["primary"] == "none"]
    q = sum(1 for t in none if "?" in t["user_text"] or shapes(t["user_text"]) == ["query"])
    short = sum(1 for t in none if len(t["user_text"].split()) < 4)
    print(f"none: {len(none)} turns, {q} look like questions, {short} under 4 words")
    elig = eligible(load_turns())
    print(f"eligible (distinct, not ERROR, 2-80 words): {len(elig)}; teachability",
          collections.Counter(teachability(t["routing"]) for t in elig).most_common())


# The orchestrator's own sub-prompts are stored as user turns; they are not
# requests. Seen: "Compose the combined response to the original request…"
_NOT_A_REQUEST = re.compile(r"^Compose the combined response", re.I)


def eligible(turns: list) -> list:
    return [t for t in turns
            if not t["duplicate"] and t["status"] != "ERROR"
            and 2 <= len(t["user_text"].split()) <= 80
            and not _NOT_A_REQUEST.search(t["user_text"])]


def _arg_shape(v):
    if isinstance(v, bool):
        return "BOOL"
    if isinstance(v, int):
        return "INT"
    if isinstance(v, str):
        if _HTML.search(v):
            return "STR:html"
        if _UUID.fullmatch(v):
            return "ID"
        if v.startswith("http"):
            return "STR:url"
        return "STR:long" if len(v) > 160 else "STR"
    if isinstance(v, list):
        return "LIST"
    if isinstance(v, dict):
        return "OBJ"
    return type(v).__name__


def cmd_inventory(args) -> None:
    """Per tool: call count, turns, top-level arg keys with value shapes.
    Written to data/real_sessions/tool_inventory.json (gitignored: examples
    may carry course content). The printed summary is counts and key names only."""
    turns = [t for t in load_turns() if not t["duplicate"]]
    inv = {}
    for t in turns:
        seen = set()
        for c in t["tool_calls"]:
            n = c.get("name")
            if not n:
                continue
            e = inv.setdefault(n, {"calls": 0, "turns": 0, "agents": collections.Counter(),
                                   "args": collections.defaultdict(collections.Counter),
                                   "content_bearing": 0, "example": None})
            e["calls"] += 1
            if n not in seen:
                e["turns"] += 1
                seen.add(n)
            e["agents"][t["agent"]] += 1
            a = c.get("args")
            if isinstance(a, dict):
                for k, v in a.items():
                    e["args"][k][_arg_shape(v)] += 1
                if _content_bearing(a):
                    e["content_bearing"] += 1
                if e["example"] is None:
                    e["example"] = json.dumps(a)[:300]
    out = {n: {"calls": e["calls"], "turns": e["turns"], "agents": dict(e["agents"]),
               "content_bearing_calls": e["content_bearing"],
               "args": {k: dict(v) for k, v in e["args"].items()}, "example": e["example"]}
           for n, e in sorted(inv.items(), key=lambda kv: -kv[1]["turns"])}
    INVENTORY.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} tools -> {INVENTORY}")
    print(f"{'tool':40s} {'turns':>5s} {'calls':>5s} {'content':>7s}  args")
    for n, e in out.items():
        if e["turns"] < 3:
            continue
        argdesc = ", ".join(f"{k}:{'/'.join(sorted(v))}" for k, v in e["args"].items())
        print(f"{n[:40]:40s} {e['turns']:5d} {e['calls']:5d} {e['content_bearing_calls']:7d}  {argdesc[:90]}")


def cmd_sample(args) -> None:
    rng = random.Random(args.seed)
    turns = eligible(load_turns())
    # stratify by agent so mobi-cbiv doesn't drown everything, cap per account
    by_agent = collections.defaultdict(list)
    for t in turns:
        by_agent[t["agent"]].append(t)
    picked = []
    per_agent = {a: max(10, int(args.n * len(v) / len(turns))) for a, v in by_agent.items()}
    for a, v in by_agent.items():
        rng.shuffle(v)
        per_account = collections.Counter()
        for t in v:
            if len([p for p in picked if p["agent"] == a]) >= per_agent[a]:
                break
            if per_account[t["account"]] >= args.max_per_account:
                continue
            per_account[t["account"]] += 1
            picked.append(t)
    rng.shuffle(picked)
    picked = picked[:args.n]
    with open(CANDIDATES, "w", encoding="utf-8") as f:
        for t in picked:
            text, flags = scrub(t["user_text"])
            f.write(json.dumps({
                "turn_id": t["turn_id"], "agent": t["agent"],
                "request": text, "scrub_flags": flags,
                "shape_guess": shapes(text),
                "routing": t["routing"],
                "teachability_guess": teachability(t["routing"]),
                "frontier_tool_calls": [c["name"] for c in t["tool_calls"]],
                "review": {"keep": None, "shape": None, "expressible": None, "notes": ""},
            }) + "\n")
    sids = sorted({t["session_db_id"] for t in picked})
    EVAL_SESSIONS.write_text(json.dumps({"seed": args.seed, "n": len(picked),
                                         "session_db_ids": sids}, indent=1), encoding="utf-8")
    print(f"{len(picked)} candidates -> {CANDIDATES} (review before promoting)")
    print(f"{len(sids)} session ids frozen -> {EVAL_SESSIONS}")


def cmd_pool(args) -> None:
    """B2 mining pool: every eligible turn whose session is not in the frozen
    eval set. Scrubbed the same way; routing + teachability guess attached so
    the authoring batches can be stratified."""
    frozen = set(json.loads(EVAL_SESSIONS.read_text(encoding="utf-8"))["session_db_ids"])
    turns = [t for t in eligible(load_turns()) if t["session_db_id"] not in frozen]
    teach = collections.Counter()
    with open(POOL, "w", encoding="utf-8") as f:
        for t in turns:
            text, flags = scrub(t["user_text"])
            tg = teachability(t["routing"])
            teach[tg] += 1
            f.write(json.dumps({
                "turn_id": t["turn_id"], "session_db_id": t["session_db_id"], "agent": t["agent"],
                "request": text, "scrub_flags": flags, "shape_guess": shapes(text),
                "routing": t["routing"], "teachability_guess": tg,
                "frontier_tool_calls": [{"name": c["name"], "args": c["args"]} for c in t["tool_calls"]],
            }) + "\n")
    print(f"{len(turns)} pool turns (excluding {len(frozen)} eval sessions) -> {POOL}")
    print("teachability:", teach.most_common())


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.real_requests")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    sub.add_parser("histogram")
    sub.add_parser("inventory")
    sub.add_parser("pool")
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=250)
    s.add_argument("--seed", type=int, default=20260902)
    s.add_argument("--max-per-account", type=int, default=12)
    args = ap.parse_args()
    {"extract": cmd_extract, "histogram": cmd_histogram, "inventory": cmd_inventory,
     "sample": cmd_sample, "pool": cmd_pool}[args.cmd](args)


if __name__ == "__main__":
    main()
