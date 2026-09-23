"""How well does an off-the-shelf small sentence encoder make the choice the
tiny planner makes, from the text alone?

For every reference CALL on the decoyed holdout whose tool has signature twins
(same signature, same effect tag), embed the request and each candidate's
description with a pretrained retrieval encoder and pick the closest. No
training, no program context -- just "which description does this request
mean". Also reports how far apart the encoder keeps twins, to compare with
the tiny planner's line vectors (twin_geometry.py).

  python teacher_twins.py [--model unsloth/bge-small-en-v1.5]
      [--view desc|line|sig|name|name+desc] [--squeeze K]

--view picks what the encoder embeds per candidate (the full rendered line is
what the tiny planner reads); --squeeze K projects every embedding onto the top
K principal components of training-world text first, to ask how many
dimensions the distinction needs.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.context import TaskContext, effect_code, format_type, serialize_context  # noqa: E402

CALL = re.compile(r"CALL (T\d+)")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def sig(t):
    ps = " ".join(f"{p.sym}:{format_type(p.type)}{'' if p.required else '?'}"
                  for p in t.params)
    return f"({ps}) -> {format_type(t.returns) if t.returns else '-'} [{effect_code(t.effects)}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    # the unsloth mirror: this laptop's BAAI cache holds only a tokenizer config
    ap.add_argument("--model", default="unsloth/bge-small-en-v1.5")
    ap.add_argument("--corpus", default=str(ROOT / "data/s5_holdout_decoy.jsonl"))
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--view", default="desc",
                    choices=["desc", "line", "sig", "name", "name+desc"])
    ap.add_argument("--squeeze", type=int, default=0, metavar="K")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = AutoModel.from_pretrained(args.model).eval()
    torch.set_grad_enabled(False)
    cache: dict[str, torch.Tensor] = {}

    def embed(texts):
        todo = [t for t in dict.fromkeys(texts) if t not in cache]
        for i in range(0, len(todo), 64):
            b = tok(todo[i:i + 64], padding=True, truncation=True, max_length=128,
                    return_tensors="pt")
            v = enc(**b).last_hidden_state[:, 0]           # bge: CLS pooling
            v = torch.nn.functional.normalize(v, dim=-1)
            for t, x in zip(todo[i:i + 64], v):
                cache[t] = x
        return torch.stack([cache[t] for t in texts])

    proj = None
    if args.squeeze:
        # principal components of TRAINING-world text only; the exam never fits it
        fit = set()
        for i, line in zip(range(4000), open(ROOT / "data/s5_plain.jsonl", encoding="utf-8")):
            r = json.loads(line)
            fit.add(QUERY_PREFIX + r["request"])
            fit.update(t["desc"] for t in r["context"]["tools"])
        X = embed(sorted(fit))
        mu = X.mean(0)
        V = torch.linalg.svd(X - mu, full_matrices=False)[2][:args.squeeze].T
        proj = lambda E: torch.nn.functional.normalize((E - mu) @ V, dim=-1)

    right = total = 0
    chance = 0.0
    tw_cos, un_cos = [], []
    for i, line in zip(range(args.limit), open(args.corpus, encoding="utf-8")):
        row = json.loads(line)
        ctx = TaskContext.from_json(row["context"])
        groups = defaultdict(list)
        for t in ctx.tools.values():
            groups[sig(t)].append(t)
        lines = {ln.split(" ", 1)[0]: ln
                 for ln in serialize_context(row["request"], ctx).splitlines()
                 if re.match(r"T\d+ \(", ln)}
        text = {"desc": lambda t: t.desc,
                "line": lambda t: lines[t.sym],
                "sig": lambda t: lines[t.sym].split(" :: ", 1)[0],
                "name": lambda t: t.name.replace("_", " "),
                "name+desc": lambda t: t.name.replace("_", " ") + ": " + t.desc}[args.view]
        q = embed([QUERY_PREFIX + row["request"]])[0]
        if proj is not None:
            q = proj(q[None])[0]
        for s in dict.fromkeys(CALL.findall(row["reference"]["segments"][0])):
            t = ctx.tools.get(s)
            if t is None:
                continue
            grp = groups[sig(t)]
            if len(grp) < 2:
                continue
            D = embed([text(g) for g in grp])
            if proj is not None:
                D = proj(D)
            pick = grp[int((D @ q).argmax())]
            right += pick.sym == s
            total += 1
            chance += 1 / len(grp)
            ri = next(k for k, g in enumerate(grp) if g.sym == s)
            for k in range(len(grp)):
                if k != ri:
                    tw_cos.append(float(D[ri] @ D[k]))
        # unrelated pairs: two random tools of this task with different signatures
        ts = list(ctx.tools.values())
        for _ in range(4):
            a, b = random.sample(ts, 2)
            if sig(a) != sig(b):
                E = embed([text(a), text(b)])
                if proj is not None:
                    E = proj(E)
                un_cos.append(float(E[0] @ E[1]))

    med = lambda xs: sorted(xs)[len(xs) // 2]
    print(f"{args.model} view={args.view} squeeze={args.squeeze or 'none'}: "
          f"{total} twin decisions over {args.limit} tasks")
    print(f"  picks the real tool by request-description cosine: {right}/{total} "
          f"({right/total:.1%}); uniform guess inside the group {chance/total:.1%}")
    print(f"  cos(real desc, twin desc) median {med(tw_cos):.3f}; "
          f"cos(two unrelated tools) median {med(un_cos):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
