"""Per-decision picks on the full exam's readable flip-slot decisions, for the
U2 and X4 stages and the teacher (descriptions), then the miss overlap."""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, r"C:\code\covenant-agent\models\tiny")
from stage_pretrain import build_head, forward, load_part  # noqa: E402
from train import called_tools  # noqa: E402

TINY = Path(r"C:\code\covenant-agent\models\tiny")
cache = TINY / "data_cache_s6"
meta = json.loads((cache / "config.json").read_text(encoding="utf-8"))
lay = meta["layout"]
n_kw, max_tool, pad = lay["n_kw"], lay["max_tool"], meta["in_pad"]
exam = load_part(cache, "holdout", 100000, ids=("+decoy", "+flip"))
hmeta = json.loads((cache / "holdout_meta.json").read_text(encoding="utf-8"))
keep = [i for i, m in enumerate(hmeta) if "+decoy" in m["task_id"] or "+flip" in m["task_id"]]
rows = pickle.load(open(cache / "rows.pkl", "rb"))["holdout"]
teacher = torch.load(cache / "teacher.pt")["table"].float()

heads = {}
for name in ("u2", "x4"):
    ck = torch.load(TINY / f"runs/stages_{name}/stages.pt", map_location="cpu")
    h = build_head(ck["cfg"], meta["in_vocab"], pad)
    h.load_state_dict(ck["head"])
    h.eval()
    heads[name] = h

dec = []   # one per readable flip-slot decision
N = exam["tgt"].size(0)
bs = 64
with torch.no_grad():
    for i in range(0, N, bs):
        b = {k: v[i:i + bs] for k, v in exam.items()}
        combs = {}
        for name, h in heads.items():
            comb, *_ , M = forward(h, b, pad)
            combs[name] = comb
        pos = called_tools(b["tgt"], n_kw, max_tool)[:, :M]
        grp = b["sig_group"][:, :M].long()
        live = torch.arange(M)[None] < b["n_tool"].long()[:, None]
        for r in range(pos.size(0)):
            for t in pos[r].nonzero().flatten().tolist():
                members = ((grp[r] == grp[r, t]) & live[r]).nonzero().flatten()
                if len(members) < 2 or not bool(b["flip_tool"][r, t]):
                    continue
                q = teacher[int(b["t_req"][r])]
                picks = {name: int(members[int(c[r, members].argmax())]) for name, c in combs.items()}
                picks["teacher"] = int(members[int((teacher[b["t_desc"][r, members].long()] @ q).argmax())])
                picks["t_name"] = int(members[int((teacher[b["t_name"][r, members].long()] @ q).argmax())])
                dec.append({"row": keep[i + r], "gold": t, "members": members.tolist(),
                            "opaque": bool(b["opaque"][r]), **picks})

n = len(dec)
print(f"readable flip-slot decisions: {n}")
for k in ("u2", "x4", "teacher", "t_name"):
    print(f"  {k:8s} {sum(d[k] == d['gold'] for d in dec) / n:.1%}")


def overlap(a, b):
    ma = [d[a] != d["gold"] for d in dec]
    mb = [d[b] != d["gold"] for d in dec]
    both = sum(x and y for x, y in zip(ma, mb))
    same_wrong = sum(x and y and d[a] == d[b] for x, y, d in zip(ma, mb, dec))
    na, nb = sum(ma), sum(mb)
    indep = na * nb / n
    print(f"{a} misses {na}, {b} misses {nb}; both {both} (independent would be {indep:.0f}); "
          f"same wrong tool {same_wrong}; only {a} {na - both}, only {b} {nb - both}")
    print(f"  of {a}'s misses, {both / na:.0%} are also {b}'s")
    return ma, mb


print()
mu, mt = overlap("u2", "teacher")
overlap("x4", "teacher")
overlap("u2", "x4")

# what the shared misses look like
both = [d for d, x, y in zip(dec, mu, mt) if x and y]
worlds = Counter(rows[d["row"]]["world"] for d in both)
levels = Counter(rows[d["row"]]["level"] for d in both)
print("\nshared misses by level", sorted(levels.items()))
print("shared misses, top worlds", worlds.most_common(8))


def tools_of(r):
    return sorted(rows[r]["context"]["tools"], key=lambda t: int(t["sym"][1:]))


def show(title, items, k=12):
    print(f"\n=== {title} ({len(items)}) ===")
    for d in items[:k]:
        r = rows[d["row"]]
        ts = tools_of(d["row"])
        print(f"- [{r['world']} L{r['level']}] {r['request']}")
        print(f"    right : {ts[d['gold']].get('name')}: {ts[d['gold']]['desc']}")
        print(f"    U2    : {ts[d['u2']].get('name')}: {ts[d['u2']]['desc']}")
        print(f"    teach : {ts[d['teacher']].get('name')}: {ts[d['teacher']]['desc']}")


import random  # noqa: E402
random.Random(0).shuffle(both)
show("both miss", both)
only_u = [d for d, x, y in zip(dec, mu, mt) if x and not y]
random.Random(1).shuffle(only_u)
show("only U2 misses", only_u, 6)
only_t = [d for d, x, y in zip(dec, mu, mt) if y and not x]
random.Random(2).shuffle(only_t)
show("only teacher misses", only_t, 6)
# per-decision picks, for follow-up
json.dump(dec, open(TINY / "runs/stages_u2/miss_overlap.json", "w"))
