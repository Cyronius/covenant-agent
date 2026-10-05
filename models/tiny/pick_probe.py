"""Step 0 of plan step 2d (.claude/plans/staged-decoder-experts.md): can a
constant picker be learned from what today's encoder already gives?

About 90% of the staged decoder's wrong constants are decoys: constants in
the task's context that the program doesn't use (R28 point 11). The picker
reads each constant's vector next to the request's and says whether the
program uses it. Here it trains alone, on a FROZEN checkpoint's encoder
output, and is scored on validation rows (seen worlds), the test split, and
the plain holdout half (new worlds):

  python pick_probe.py --ckpt runs/pod_staged/runs/d3_DR_s0/best.pt \\
      --cache data_cache_d3 --report ../../results/logs/staged/pick_probe_DR_s0.json

The probe has the shape the plan's picker would have: one attention layer in
which each constant reads the request tokens and the other constants, then
a score. Reported per constant: share right, decoys let through, used
constants dropped; per row: every constant right.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evaluate import load_model
from model import StructuralModel
from train import load_split, unpack


@torch.no_grad()
def features(model, cache: Path, split: str, rows: list[int] | None, device, id_not_contains=None,
             limit=None, batch: int = 64):
    """Constant vectors, request vectors, their pads, and the labels (does the
    reference program use constant j), from the frozen encoder."""
    c = model.c
    ds = load_split(cache, split, "structural")
    meta = json.loads((cache / f"{split}_meta.json").read_text(encoding="utf-8"))
    idx = rows if rows is not None else [i for i, m in enumerate(meta)
                                          if not id_not_contains or id_not_contains not in m["task_id"]]
    if limit:
        idx = idx[:limit]
    o = c.n_kw + c.max_tool + c.max_field
    C, Cp, R, Rp, Y = [], [], [], [], []
    for b in DataLoader(Subset(ds, idx), batch_size=batch):
        inputs, tgt = unpack([t.to(device) for t in b], "structural", 0, ds.keys)
        mem = StructuralModel.encode_inputs(model, inputs)
        x, pad = mem.mem, mem.pad
        a = c.max_tool + c.max_field
        n_req = inputs["req_tok"].size(1)
        C.append(x[:, a:a + c.max_const].half().cpu())
        Cp.append(pad[:, a:a + c.max_const].cpu())
        R.append(x[:, -n_req:].half().cpu())
        Rp.append(pad[:, -n_req:].cpu())
        used = torch.zeros(tgt.size(0), c.max_const, dtype=torch.bool)
        for r in range(tgt.size(0)):
            t = tgt[r].cpu()
            u = t[(t >= o) & (t < o + c.max_const)] - o
            used[r, u] = True
        Y.append(used)
    return [torch.cat(v) for v in (C, Cp, R, Rp, Y)]


class Picker(nn.Module):
    """Each constant attends to the request tokens and the other constants,
    then gets a score: does the program use it."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.out = nn.Linear(d, 1)

    def forward(self, cv, cpad, rv, rpad):
        kv = torch.cat([rv, cv], 1)
        kpad = torch.cat([rpad, cpad], 1)
        h, _ = self.att(self.n1(cv), self.n1(kv), self.n1(kv), key_padding_mask=kpad)
        x = cv + h
        x = x + self.ff(self.n2(x))
        return self.out(x).squeeze(-1)


def score(p: Picker, data) -> dict:
    cv, cp, rv, rp, y = data
    with torch.no_grad():
        pred = p(cv.float(), cp, rv.float(), rp) > 0
    live = ~cp
    right = (pred == y) & live
    decoy, used = live & ~y, live & y
    rows_ok = ((pred == y) | cp).all(1)
    return {"constants": int(live.sum()), "decoys": int(decoy.sum()),
            "right": float(right.sum() / live.sum()),
            "decoys_let_through": float((pred & decoy).sum() / decoy.sum().clamp(min=1)),
            "used_dropped": float((~pred & used).sum() / used.sum().clamp(min=1)),
            "rows_all_right": float(rows_ok.float().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--train-rows", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--report", default=None)
    ap.add_argument("--save", default=None, help="the trained probe (.pt), for evaluate.py --pick-probe")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    cache = Path(args.cache)
    model = load_model(Path(args.ckpt), device)
    model.eval()
    n_train = len(json.loads((cache / "train_meta.json").read_text(encoding="utf-8")))
    rows = sorted(random.Random(0).sample(range(n_train), min(args.train_rows, n_train)))
    tr = features(model, cache, "train", rows, device)
    sets = {"val (seen worlds)": features(model, cache, "val", None, device),
            "test (seen worlds)": features(model, cache, "test", None, device),
            "plain (new worlds)": features(model, cache, "holdout", None, device,
                                           id_not_contains="+", limit=2000)}
    cv, cp, rv, rp, y = tr
    p = Picker(model.c.d)
    opt = torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=0.01)
    live = ~cp
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        perm = torch.randperm(cv.size(0))
        p.train()
        for s in range(0, cv.size(0), 128):
            j = perm[s:s + 128]
            logit = p(cv[j].float(), cp[j], rv[j].float(), rp[j])
            loss = F.binary_cross_entropy_with_logits(logit[live[j]], y[j][live[j]].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        p.eval()
        v = score(p, sets["val (seen worlds)"])
        if v["right"] > best:
            best, best_state = v["right"], {k: t.clone() for k, t in p.state_dict().items()}
    p.load_state_dict(best_state)
    report = {"ckpt": args.ckpt, "train": score(p, tr)}
    report.update({name: score(p, d) for name, d in sets.items()})
    for name, r in report.items():
        if name == "ckpt":
            continue
        print(f"  {name:20s} constants {r['constants']:6d} (decoys {r['decoys']:5d})  right {100*r['right']:5.1f}%"
              f"  decoys let through {100*r['decoys_let_through']:5.1f}%  used dropped {100*r['used_dropped']:5.1f}%"
              f"  rows all right {100*r['rows_all_right']:5.1f}%")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.save:
        torch.save({"state": p.state_dict(), "d": model.c.d, "ckpt": args.ckpt}, args.save)


if __name__ == "__main__":
    main()
