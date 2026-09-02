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
  python -m harness.real_requests sample --n 250    # -> data/real_sessions/eval_candidates.jsonl
                                                    #    (scrubbed; human review before it becomes
                                                    #     data/holdout/e_real_sessions.jsonl)

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


def cmd_sample(args) -> None:
    rng = random.Random(args.seed)
    turns = [t for t in load_turns()
             if not t["duplicate"] and t["status"] != "ERROR"
             and 2 <= len(t["user_text"].split()) <= 80]
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
                "frontier_tool_calls": [c["name"] for c in t["tool_calls"]],
                "review": {"keep": None, "shape": None, "expressible": None, "notes": ""},
            }) + "\n")
    print(f"{len(picked)} candidates -> {CANDIDATES} (review before promoting)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.real_requests")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    sub.add_parser("histogram")
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=250)
    s.add_argument("--seed", type=int, default=20260902)
    s.add_argument("--max-per-account", type=int, default=12)
    args = ap.parse_args()
    {"extract": cmd_extract, "histogram": cmd_histogram, "sample": cmd_sample}[args.cmd](args)


if __name__ == "__main__":
    main()
