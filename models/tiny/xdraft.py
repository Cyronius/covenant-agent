"""Cross-fitted drafts for the staged refiner (.claude/plans/staged-decoder-
experts.md, step 2b).

The refiner learned to copy its draft because, on the rows it trains on, the
draft stage has seen the answer (R28). Here every training row gets a draft
from a draft-only model that never saw it: model k trained on fold k only
(`train.py --fold k/2`), so it drafts the rows of the other fold.

  python xdraft.py --cache data_cache_d3 \\
      --ckpts runs/d3_half0_s0/best.pt runs/d3_half1_s0/best.pt \\
      --out data_cache_d3/train_xdraft_s0.pt

Writes {"tok": (N, canvas) int16, "conf": (N, canvas) float16} in the split's
row order, for `train.py --xdraft`. Prints how good the stored drafts are
next to how good each model is on its own fold: the first should look like
the draft's validation accuracy, the second much higher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate import load_model
from staged import PAD_ID, fold_of
from train import load_split, unpack


@torch.no_grad()
def drafts(model, ds, device, batch: int = 128):
    toks, confs = [], []
    for b in DataLoader(ds, batch_size=batch):
        inputs, _ = unpack([t.to(device) for t in b], "structural", 0, ds.keys)
        p = model.encode_inputs(inputs).draft_logits.float().softmax(-1)
        conf, tok = p.max(-1)
        toks.append(tok.short().cpu())
        confs.append(conf.half().cpu())
    return torch.cat(toks), torch.cat(confs)


def accuracy(tok, tgt, kinds) -> dict:
    """Slot accuracy by kind over the program (non-PAD target slots), and
    whole-program exact match."""
    live = tgt != PAD_ID
    out = {"program": float((tok == tgt).all(1).float().mean())}
    for name, k in (("keyword", 0), ("tool", 1), ("field", 2), ("constant", 3)):
        m = live & (kinds == k)
        out[name] = float((tok[m] == tgt[m]).float().mean()) if m.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpts", nargs=2, required=True,
                    help="the draft models trained on fold 0 and fold 1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cache, device = Path(args.cache), torch.device(args.device)
    meta = json.loads((cache / f"{args.split}_meta.json").read_text(encoding="utf-8"))
    fold = torch.tensor([fold_of(m["task_id"], 2) for m in meta])
    ds = load_split(cache, args.split, "structural")
    tgt = ds.tensors[ds.keys.index("tgt")].long()

    tok = torch.zeros(tgt.shape, dtype=torch.int16)
    conf = torch.zeros(tgt.shape, dtype=torch.float16)
    report = {}
    for k, ck in enumerate(args.ckpts):
        model = load_model(Path(ck), device)
        model.eval()
        if not getattr(model.c, "stages", "") or "draft" not in model.c.stages:
            raise SystemExit(f"{ck}: not a staged model with a draft stage")
        t, c = drafts(model, ds, device)
        kinds = model._kind_of(tgt)
        other, own = fold != k, fold == k
        tok[other], conf[other] = t[other], c[other]
        report[f"model {k} on the other fold (stored)"] = accuracy(t[other].long(), tgt[other], kinds[other])
        report[f"model {k} on its own fold"] = accuracy(t[own].long(), tgt[own], kinds[own])
        print(f"model {k} ({ck}): rows {int(other.sum())} stored, {int(own.sum())} its own")
    torch.save({"tok": tok, "conf": conf, "ckpts": args.ckpts}, args.out)
    for name, r in report.items():
        print(f"  {name:36s} " + "  ".join(
            f"{k} {v:.3f}" for k, v in r.items() if v is not None))
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {tok.size(0)} rows")


if __name__ == "__main__":
    main()
