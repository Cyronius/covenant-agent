"""Calibration and the early-exit threshold for staged models
(.claude/plans/staged-decoder-experts.md, "Early-exit threshold").

Two numbers per staged model (per expert, for a bundle):

  temp  rescales the draft's slot probabilities so that 0.9 means right about
        90% of the time: fit on the val split's draft slots (one trip from a
        blank canvas, the slots up to and including each program's end).
        Only needed where confidences from two experts are compared.
  thr   the early-exit threshold, on the raw program confidence (the lowest
        slot confidence, staged.program_confidence): the lowest value at which
        the programs that would exit are as accurate as the refiner's
        programs for the same tasks, measured on val with the sandbox.

Both read generations evaluate.py wrote: <gen>.jsonl carries each task's
draft confidence; its .score.json and the draft file's .draft.score.json
say which program reached the goal.

  python calibrate.py fit --ckpt runs/X/best.pt --cache data_cache_d3 \\
      --val-gen out/X_val.jsonl --out out/X_cal.json
  python calibrate.py apply --gen out/X_plain.jsonl --thr 0.93
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from evaluate import load_model
from staged import task_type
from train import load_split, unpack


def _goals(score_path: Path) -> dict:
    rows = json.loads(score_path.read_text(encoding="utf-8"))["rows"]
    return {r["task_id"]: bool(r.get("goal")) for r in rows if "goal" in r}


def joined(gen: Path) -> list[dict]:
    """[{task_id, conf, draft_goal, ref_goal}] for every task scored both ways."""
    ref = _goals(gen.with_suffix(".score.json"))
    dra = _goals(gen.with_suffix(".draft.score.json"))
    out = []
    for line in gen.open(encoding="utf-8"):
        g = json.loads(line)
        t = g["task_id"]
        if t in ref and t in dra:
            out.append({"task_id": t, "conf": g["draft_conf"], "world": g.get("world"),
                        "expert": g.get("expert"), "draft_goal": dra[t], "ref_goal": ref[t]})
    return out


def choose_threshold(rows: list[dict], min_exit: int = 20) -> float | None:
    """The lowest confidence t at which the tasks with conf >= t are solved
    at least as often by their drafts as by the refiner. None: no threshold
    keeps that promise for at least `min_exit` tasks, so never exit."""
    rows = sorted(rows, key=lambda r: -r["conf"])
    best, d, f = None, 0, 0
    for i, r in enumerate(rows, 1):
        d += r["draft_goal"]
        f += r["ref_goal"]
        last = i == len(rows) or rows[i]["conf"] < r["conf"]
        if last and i >= min_exit and d >= f:
            best = r["conf"]
    return best


def at_threshold(rows: list[dict], thr) -> dict:
    """Goal rate when tasks at or above `thr` keep their draft and the rest
    get the refiner's program, beside the two pure readouts. `thr` may be a
    dict, one threshold per expert (rows carry the expert that wrote them)."""
    n = max(len(rows), 1)

    def exits(r):
        t = thr.get(r["expert"]) if isinstance(thr, dict) else thr
        return t is not None and r["conf"] >= t

    ex = [r for r in rows if exits(r)]
    goal = sum(r["draft_goal"] if exits(r) else r["ref_goal"] for r in rows)
    return {"n": len(rows), "thr": thr, "exit_share": len(ex) / n, "goal": goal / n,
            "goal_refined": sum(r["ref_goal"] for r in rows) / n,
            "goal_draft": sum(r["draft_goal"] for r in rows) / n,
            "exited_goal_draft": (sum(r["draft_goal"] for r in ex) / len(ex)) if ex else None,
            "exited_goal_refined": (sum(r["ref_goal"] for r in ex) / len(ex)) if ex else None}


@torch.no_grad()
def fit_temperature(model, cache: Path, split: str = "val", types: set | None = None,
                    device="cpu", batch: int = 64) -> dict:
    """Temperature scaling of the draft's slots (grid over log T)."""
    ds = load_split(cache, split, "structural", types=types)
    logits, targets = [], []
    for s in range(0, len(ds), batch):
        cols = tuple(t[s:s + batch] for t in ds.tensors)
        inputs, tgt = unpack(cols, "structural", 0, ds.keys)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        mem = model._encode(inputs)
        dl = model.draft(inputs, mem, model.slot_table(mem)).float().cpu()
        pad = tgt == 0
        first = torch.where(pad.any(1), pad.float().argmax(1), torch.full_like(tgt[:, 0], tgt.size(1) - 1))
        live = torch.arange(tgt.size(1))[None] <= first[:, None]
        logits.append(dl[live])
        targets.append(tgt[live])
    L, y = torch.cat(logits), torch.cat(targets)
    best = None
    for lt in torch.linspace(math.log(0.25), math.log(4.0), 81).tolist():
        nll = float(torch.nn.functional.cross_entropy(L / math.exp(lt), y))
        if best is None or nll < best[1]:
            best = (math.exp(lt), nll)
    p = (L / best[0]).softmax(-1)
    conf, pred = p.max(-1)
    hit = (pred == y).float()
    bins = []
    for lo in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        m = conf >= lo
        bins.append({"conf>=": lo, "share": float(m.float().mean()),
                     "acc": float(hit[m].mean()) if m.any() else None})
    return {"temp": best[0], "nll": best[1], "slots": len(y), "bins": bins}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--ckpt", required=True)
    f.add_argument("--cache", required=True)
    f.add_argument("--val-gen", required=True, help="evaluate.py's val generation, scored both ways")
    f.add_argument("--types", default=None, help="an expert's types: calibrate on those rows")
    f.add_argument("--out", required=True, help="the cal JSON (temp, thr, the evidence)")
    f.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t = sub.add_parser("temp", help="the temperature alone (no sandbox needed: runs on a pod)")
    t.add_argument("--ckpt", required=True)
    t.add_argument("--cache", required=True)
    t.add_argument("--types", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    h = sub.add_parser("thr", help="thresholds alone from a scored val generation")
    h.add_argument("--val-gen", required=True)
    h.add_argument("--by-expert", action="store_true", help="one threshold per expert (bundles)")
    h.add_argument("--bundle", default=None, help="with --write: the bundle to stamp")
    h.add_argument("--write", default=None, help="save the bundle with its thresholds here")
    h.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--gen", nargs="+", required=True)
    a.add_argument("--thr", default=None, help="a number, or a cal / thr JSON")
    a.add_argument("--by-expert", action="store_true")
    args = ap.parse_args()

    if args.cmd == "fit":
        model = load_model(Path(args.ckpt), torch.device(args.device))
        types = set(args.types.split(",")) if args.types else None
        temp = fit_temperature(model, Path(args.cache), "val", types, args.device)
        rows = joined(Path(args.val_gen))
        if types:
            rows = [r for r in rows if task_type(r["world"]) in types]
        thr = choose_threshold(rows)
        out = {"temp": temp["temp"], "thr": thr, "temperature": temp,
               "val": at_threshold(rows, thr)}
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps({k: out[k] for k in ("temp", "thr", "val")}, indent=1))
    elif args.cmd == "temp":
        model = load_model(Path(args.ckpt), torch.device(args.device))
        types = set(args.types.split(",")) if args.types else None
        temp = fit_temperature(model, Path(args.cache), "val", types, args.device)
        Path(args.out).write_text(json.dumps({"temp": temp["temp"], "temperature": temp}, indent=1),
                                  encoding="utf-8")
        print(f"temp {temp['temp']:.3f} (nll {temp['nll']:.4f}, {temp['slots']} slots)")
    elif args.cmd == "thr":
        rows = joined(Path(args.val_gen))
        if args.by_expert:
            groups = sorted({r["expert"] for r in rows})
            thr = {e: choose_threshold([r for r in rows if r["expert"] == e]) for e in groups}
            val = {e: at_threshold([r for r in rows if r["expert"] == e], thr[e]) for e in groups}
            val["all"] = at_threshold(rows, thr)
        else:
            thr = choose_threshold(rows)
            val = at_threshold(rows, thr)
        Path(args.out).write_text(json.dumps({"thr": thr, "val": val}, indent=1), encoding="utf-8")
        print(json.dumps({"thr": thr}, indent=1))
        if args.write:
            ck = torch.load(args.bundle, map_location="cpu")
            for name, e in ck["experts"].items():
                e["cal"] = {**(e.get("cal") or {}),
                            "thr": thr.get(name) if isinstance(thr, dict) else thr}
            torch.save(ck, args.write)
            print(f"wrote {args.write}")
    else:
        thr = args.thr
        if thr and Path(thr).exists():
            thr = json.loads(Path(thr).read_text(encoding="utf-8"))["thr"]
        if not isinstance(thr, dict):
            thr = None if thr in (None, "None", "never") else float(thr)
        for g in args.gen:
            r = at_threshold(joined(Path(g)), thr)
            print(f"{g}: n={r['n']} exit {r['exit_share']:.1%}  goal {r['goal']:.1%}  "
                  f"(refined {r['goal_refined']:.1%}, draft {r['goal_draft']:.1%})")


if __name__ == "__main__":
    main()
