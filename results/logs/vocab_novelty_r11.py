"""Are U2's misses the decisions whose deciding words training never saw?

For each readable flip-slot decision: the words that separate the right
tool's description from its twins' (in the right one, in no twin). Split
the decisions by whether any such word is absent from S6's training text,
and compare miss rates, U2 vs teacher. Also pieces per deciding word under
the cache tokenizer."""
import json
import pickle
import re
from pathlib import Path

from tokenizers import Tokenizer

TINY = Path(r"C:\code\covenant-agent\models\tiny")
COV = TINY.parent.parent

cache = TINY / "data_cache_s6"
dec = json.load(open(TINY / "runs/stages_u2/miss_overlap.json"))  # from miss_overlap_r11.py
rows = pickle.load(open(cache / "rows.pkl", "rb"))["holdout"]
tk = Tokenizer.from_file(str(cache / "in_tok.json"))
W = re.compile(r"[a-z]+")

seen = set()
with open(COV / "data/s6_train.jsonl", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        seen.update(W.findall(r["request"].lower()))
        for t in r["context"]["tools"]:
            seen.update(W.findall(t["desc"].lower()))
            seen.update(W.findall((t.get("name") or "").lower().replace("_", " ")))
print(f"training vocabulary: {len(seen)} words")


def tools_of(i):
    return sorted(rows[i]["context"]["tools"], key=lambda t: int(t["sym"][1:]))


groups = {"novel": [], "seen": []}
pieces = {"hit": [], "miss": []}
for d in dec:
    ts = tools_of(d["row"])
    gold = set(W.findall(ts[d["gold"]]["desc"].lower()))
    others = set()
    for m in d["members"]:
        if m != d["gold"]:
            others |= set(W.findall(ts[m]["desc"].lower()))
    deciding = gold - others
    novel = any(w not in seen for w in deciding)
    groups["novel" if novel else "seen"].append(d)
    npc = [len(tk.encode(" " + w).ids) for w in deciding] or [0]
    pieces["miss" if d["u2"] != d["gold"] else "hit"].append(max(npc))

for g, ds in groups.items():
    n = len(ds)
    u = sum(d["u2"] != d["gold"] for d in ds) / n
    t = sum(d["teacher"] != d["gold"] for d in ds) / n
    print(f"{g:6s} deciding word unseen in training: {n:5d} decisions  "
          f"U2 misses {u:.1%}  teacher misses {t:.1%}")
for k, v in pieces.items():
    print(f"U2 {k:4s}: most-fragmented deciding word averages {sum(v) / len(v):.2f} pieces")
