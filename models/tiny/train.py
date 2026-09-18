"""Train either arm. The arm is one flag; everything else is identical.

  python train.py --arm diffusion --cache data_cache --epochs 5
  python train.py --arm ar        --cache data_cache --epochs 5

Both read the same cache and write a checkpoint plus a JSONL log of every
evaluation, so the curves can be drawn afterwards without re-running anything.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import CanvasModel, Config, mask_canvas
from tok import OutVocab


def load_split(cache: Path, name: str, limit: int | None = None) -> TensorDataset:
    """Memory-mapped, and sliced before anything is read into RAM.

    The training tensors are about 300 MB. Holding them resident competes with
    the activations, and on a machine without much headroom the process starts
    swapping. That does not surface as an out-of-memory error -- it surfaces as
    training being mysteriously slow, which is much harder to recognise. mmap
    leaves the tensors on disk, and slicing first means a short run pays for
    only the rows it uses.
    """
    d = torch.load(cache / f"{name}.pt", mmap=True)
    # `sym` is absent from caches built before the pointer head existed; a run
    # without it simply trains the plain model.
    cols = (d["src"], d["pad"], d["tgt"],
            d.get("sym", torch.full_like(d["tgt"][:, :1], -1).expand(-1, 1)))
    if limit:
        cols = tuple(t[:limit].clone() for t in cols)
    return TensorDataset(*cols)


def shift_right(tgt: torch.Tensor, bos: int) -> torch.Tensor:
    """Teacher forcing input for the control arm."""
    return torch.cat([torch.full_like(tgt[:, :1], bos), tgt[:, :-1]], dim=1)


def batch_loss(model: CanvasModel, batch, arm: str, mask_id: int, pad_id: int,
               generator=None, pad_weight: float = 1.0):
    src, pad, tgt, sym = batch
    sym = sym if sym.size(1) == model.c.out_vocab else None
    if arm == "diffusion":
        canvas, loss_mask, _ = mask_canvas(tgt, mask_id, generator)
        logits = model(src, pad, canvas, sym=sym)
        # Loss only on the slots that were hidden. The visible ones are free
        # and scoring them would let the model earn reward for copying.
        y = tgt[loss_mask]
        per = F.cross_entropy(logits[loss_mask], y, reduction="none")
        # Programs are about half the canvas, so most slots are padding and an
        # undertrained model collapses to predicting padding everywhere -- an
        # empty program, which is the single cheapest way to be wrong. Knowing
        # where a program ends is genuinely part of the task, so padding keeps a
        # share of the loss, but the share is a knob rather than an accident.
        if pad_weight != 1.0:
            w = torch.where(y == pad_id, pad_weight, 1.0)
            loss = (per * w).sum() / w.sum().clamp(min=1e-6)
        else:
            loss = per.mean()
        n_pred = int(loss_mask.sum())
    else:
        logits = model(src, pad, shift_right(tgt, bos=mask_id), sym=sym)
        # Padding slots after the program end carry no information; scoring
        # them would reward predicting PAD forever. The diffusion arm keeps
        # them because knowing where a program stops is part of its job, so
        # the control gets one extra PAD as the stop signal and nothing more.
        keep = tgt != pad_id
        keep[:, 1:] |= (tgt[:, :-1] != pad_id) & (tgt[:, 1:] == pad_id)
        loss = F.cross_entropy(logits[keep], tgt[keep])
        n_pred = int(keep.sum())
    return loss, n_pred


@torch.no_grad()
def evaluate(model, loader, arm, mask_id, pad_id, device, seed=1234, pad_weight=1.0):
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    tot, n = 0.0, 0
    for batch in loader:
        batch = [t.to(device) for t in batch]
        loss, k = batch_loss(model, batch, arm, mask_id, pad_id, g, pad_weight)
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["diffusion", "ar"], required=True)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--out", default=None, help="run directory; default runs/<arm>_s<seed>")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=3)
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--dec-loops", type=int, default=1)
    ap.add_argument("--pointer", action="store_true",
                    help="decide symbol slots by pointing at the input")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pad-weight", type=float, default=1.0,
                    help="weight on padding slots in the diffusion loss")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="use only the first N training rows (smoke tests)")
    ap.add_argument("--limit-val", type=int, default=256,
                    help="validation rows per evaluation; the full split is "
                         "1415 rows and evaluating all of them costs more than "
                         "the training steps between evaluations")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cache = Path(args.cache)
    meta = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    ov = OutVocab.load(cache / "out_vocab.json")

    run = Path(args.out or f"runs/{args.arm}_s{args.seed}")
    run.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        in_vocab=meta["in_vocab"], out_vocab=len(ov), d=args.d,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        dec_loops=args.dec_loops, dropout=args.dropout,
        max_in=meta["max_in"], canvas=meta["canvas"],
        causal=(args.arm == "ar"), pointer=args.pointer,
    )
    device = torch.device(args.device)
    model = CanvasModel(cfg)
    if args.pointer:
        import re as _re
        model.set_symbol_ids([i for i, t in enumerate(ov.itos)
                              if _re.fullmatch(r"[TFCSNBDI]\d+", t)])
    model = model.to(device)

    train_ds = load_split(cache, "train", args.limit_train)
    val_ds = load_split(cache, "val", args.limit_val)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    steps = len(train_dl) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log = open(run / "log.jsonl", "a", encoding="utf-8")
    (run / "config.json").write_text(json.dumps(
        {**vars(args), "params": model.n_params(), **meta}, indent=1), encoding="utf-8")

    # flush: stdout is block-buffered when redirected to a file, so without this
    # a redirected run shows no sign of life until the first step print, and a
    # run that died at startup looks exactly like one that is working.
    print(f"arm={args.arm} params={model.n_params()/1e6:.2f}M device={device} "
          f"train={len(train_ds)} steps={steps}", flush=True)

    step, t0, run_loss, run_n = 0, time.time(), 0.0, 0
    best = float("inf")
    for epoch in range(args.epochs):
        for batch in train_dl:
            batch = [t.to(device) for t in batch]
            loss, k = batch_loss(model, batch, args.arm, ov.mask, ov.pad,
                                 pad_weight=args.pad_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item() * k
            run_n += k
            step += 1

            if step % 20 == 0:
                el = time.time() - t0
                print(f"  step {step:5d}/{steps} loss {run_loss/run_n:6.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {el/step:5.2f}s/step",
                      flush=True)
                run_loss, run_n = 0.0, 0

            if step % args.eval_every == 0 or step == steps:
                vl = evaluate(model, val_dl, args.arm, ov.mask, ov.pad, device,
                              pad_weight=args.pad_weight)
                rec = {"step": step, "epoch": epoch, "val_loss": vl,
                       "secs": time.time() - t0}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"  == step {step} val_loss {vl:.4f}", flush=True)
                if vl < best:
                    best = vl
                    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                                "step": step, "val_loss": vl}, run / "best.pt")

    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                "step": step, "val_loss": best}, run / "last.pt")
    log.close()
    print(f"done. best val_loss {best:.4f} -> {run}")


if __name__ == "__main__":
    main()
