"""The router (.claude/plans/staged-decoder-experts.md, steps 0a and 3):
a small classifier that picks a task type from the shared encoder's output.

It reads `staged.route_features` (the mean request, tool and register
vectors) from a frozen checkpoint's encoder, trains on a type-balanced
sample of the cache's training rows, and is scored where it matters: worlds
it never saw and hand-written requests. Generated test rows give their type
away by template (an observe-act request is a long world brief, a data
request one sentence), so the test split's number is reported but is not
the gate.

  python router.py --ckpt runs/.../best.pt --cache data_cache_d3 \\
      --out runs/router_d3.pt --report ../../results/logs/staged/router.json

Exams (task JSONL, first turn only): the held-out brief worlds (observe-act),
the coursebuilder app (pages), telecom (data) and the 70 demo requests
(data). Their contexts are encoded as play.py encodes them.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate import load_model
from model import StructuralModel
from staged import TYPES, Router, route_features, task_type
from train import load_split, unpack

ROOT = Path(__file__).resolve().parents[2]
EXAMS = {
    "boatyard": "data/holdout/e_brief_boatyard.jsonl",
    "rooms_after": "data/holdout/e_brief_rooms_after.jsonl",
    "house": "data/holdout/e_brief_house.jsonl",
    "coursebuilder": "data/holdout/e_brief_coursebuilder.jsonl",
    "telecom": "data/holdout/e_service_telecom.jsonl",
    "demo": "data/holdout/e_demo_requests_typed.jsonl",
}


def encoder_of(model):
    """The shared encoder's call, whatever the checkpoint (plain, staged, or
    a bundle's first expert)."""
    m = model.experts[0] if hasattr(model, "experts") else model
    return m, (lambda inputs: StructuralModel.encode_inputs(m, inputs))


@torch.no_grad()
def cache_features(model, cache: Path, split: str, per_type: int | None, seed: int,
                   device, batch: int = 64):
    m, enc = encoder_of(model)
    meta = json.loads((cache / f"{split}_meta.json").read_text(encoding="utf-8"))
    types = [task_type(r["world"]) for r in meta]
    idx = list(range(len(meta)))
    if per_type:
        rng = random.Random(seed)
        by = defaultdict(list)
        for i in idx:
            by[types[i]].append(i)
        idx = sorted(j for t in by for j in rng.sample(by[t], min(per_type, len(by[t]))))
    ds = load_split(cache, split, "structural")
    feats = []
    for s in range(0, len(idx), batch):
        sel = torch.tensor(idx[s:s + batch])
        cols = tuple(t[sel] for t in ds.tensors)
        inputs, _ = unpack(cols, "structural", 0, ds.keys)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats.append(route_features(enc(inputs), inputs, m.c).float().cpu())
    y = torch.tensor([TYPES.index(types[i]) for i in idx])
    return torch.cat(feats), y, [meta[i]["world"] for i in idx]


@torch.no_grad()
def exam_features(model, cache: Path, path: Path, device, max_const=40, max_field=56):
    """First-turn features for a task JSONL, encoded as play.py does."""
    from canvas import Layout, context_symbols
    from harness.context import TaskContext, serialize_context
    from prep import encode_one
    from tokenizers import Tokenizer
    m, enc = encoder_of(model)
    cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    tk = Tokenizer.from_file(str(cache / "in_tok.json"))
    tk.no_truncation()
    tk.no_padding()
    layout = Layout.from_dict(cfg["layout"])
    layout = dataclasses.replace(layout, max_const=max(layout.max_const, max_const),
                                 max_field=max(layout.max_field, max_field))
    dims = {"max_line": cfg["max_line"], "max_req": cfg["max_req"],
            "max_reg": cfg.get("max_reg", 8),
            **{k: cfg[k] for k in ("names", "split", "desc_chars", "max_sig",
                                   "max_desc", "max_name", "name_words") if k in cfg}}
    feats, worlds = [], []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("symbols") != "typed":
            # as play.py --retype and the demo server do
            from harness.retype import retype_task
            from runtime.worlds import get_world
            row = retype_task(row, get_world(row["world"]), symbols="typed",
                              enums=True, kinds=True)
        ctx = TaskContext.from_json(row["context"])
        source = serialize_context(row["request"], ctx, None, names=bool(dims.get("names")))
        inputs = encode_one(source, context_symbols(row["context"]), tk, layout, dims, device)
        feats.append(route_features(enc(inputs), inputs, m.c).float().cpu())
        worlds.append(row["world"])
    return torch.cat(feats), worlds


def train_router(Xtr, ytr, Xva, yva, d: int, seed: int = 0, epochs: int = 60) -> Router:
    torch.manual_seed(seed)
    r = Router(d, list(TYPES))
    opt = torch.optim.AdamW(r.parameters(), lr=1e-3, weight_decay=0.01)
    counts = torch.bincount(ytr, minlength=len(TYPES)).float().clamp(min=1)
    w = counts.sum() / counts / len(TYPES)
    best, best_state = -1.0, None
    for _ in range(epochs):
        r.train()
        perm = torch.randperm(len(ytr))
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            loss = F.cross_entropy(r(Xtr[b]), ytr[b], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        r.eval()
        with torch.no_grad():
            pred = r(Xva).argmax(-1)
        acc = sum(float((pred[yva == k] == k).float().mean()) for k in yva.unique()) / len(yva.unique())
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in r.state_dict().items()}
    r.load_state_dict(best_state)
    r.eval()
    return r


def score(r: Router, X, want: list[str], margin: float) -> dict:
    with torch.no_grad():
        p = r(X).softmax(-1)
    top = p.argmax(-1)
    order = p.argsort(-1, descending=True)
    rows = Counter()
    conf = Counter()
    for i, t in enumerate(want):
        k = TYPES.index(t)
        rows["n"] += 1
        rows["top1"] += int(top[i]) == k
        unsure = float(p[i, top[i]]) < margin
        rows["unsure"] += unsure
        rows["in_tried"] += (k in order[i, :2].tolist()) if unsure else int(top[i]) == k
        conf[f"{t}->{TYPES[int(top[i])]}"] += 1
    n = max(rows["n"], 1)
    return {"n": rows["n"], "top1": rows["top1"] / n, "unsure": rows["unsure"] / n,
            "right_expert_tried": rows["in_tried"] / n, "confusion": dict(conf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the shared model whose encoder the experts sit on")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="router weights (.pt)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--per-type", type=int, default=2000, help="training rows per type")
    ap.add_argument("--margin", type=float, default=0.8,
                    help="below this top probability the bundle tries two experts")
    ap.add_argument("--exclude", default=None,
                    help="comma list of types to leave out of training (the bolt-on test)")
    ap.add_argument("--exam-root", default=str(ROOT),
                    help="where data/holdout/ lives (the bundle root on a pod)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    cache = Path(args.cache)
    model = load_model(Path(args.ckpt), device)
    d = encoder_of(model)[0].c.d

    Xtr, ytr, _ = cache_features(model, cache, "train", args.per_type, 0, device)
    Xva, yva, _ = cache_features(model, cache, "val", None, 0, device)
    if args.exclude:
        drop = {TYPES.index(t) for t in args.exclude.split(",")}
        keep = torch.tensor([int(v) not in drop for v in ytr])
        Xtr, ytr = Xtr[keep], ytr[keep]
        keep = torch.tensor([int(v) not in drop for v in yva])
        Xva, yva = Xva[keep], yva[keep]
    print(f"router: {len(ytr)} training rows {dict(Counter(TYPES[int(v)] for v in ytr))}", flush=True)
    r = train_router(Xtr, ytr, Xva, yva, d)

    report = {"ckpt": args.ckpt, "cache": str(cache), "margin": args.margin,
              "exclude": args.exclude, "train": dict(Counter(TYPES[int(v)] for v in ytr))}
    Xte, yte, _ = cache_features(model, cache, "test", None, 0, device)
    report["test (template give-away)"] = score(r, Xte, [TYPES[int(v)] for v in yte], args.margin)
    Xho, yho, _ = cache_features(model, cache, "holdout", 300, 0, device)
    report["holdout worlds (data)"] = score(r, Xho, [TYPES[int(v)] for v in yho], args.margin)
    for name, rel in EXAMS.items():
        X, worlds = exam_features(model, cache, Path(args.exam_root) / rel, device)
        report[name] = score(r, X, [task_type(w) for w in worlds], args.margin)
    for k, v in report.items():
        if isinstance(v, dict) and "top1" in v:
            print(f"  {k:28s} n={v['n']:5d}  top1 {v['top1']:6.1%}  unsure {v['unsure']:5.1%}  "
                  f"right expert tried {v['right_expert_tried']:6.1%}  {v['confusion']}")
    torch.save({"state": r.state_dict(), "names": r.names, "d": d,
                "hidden": r.net[1].out_features, "margin": args.margin}, args.out)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
