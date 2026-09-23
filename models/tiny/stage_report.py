"""Per-stage twin geometry of a planner checkpoint, on a split cache's holdout
(description-reading plan step 4's read-out; `twin_geometry.py` generalised
to the split encoder and to real decoyed twins).

For every stage the pointer's input passes through, over the holdout's
twin groups (tools of one row whose signatures match, `sig_group`):

  cos(twin, twin)            median, vs two unrelated tools of the same row
  twin dist / unrelated dist median ratio: 1.0 is "twins as far apart as
                             anything", R10's proxy line encoder sat at 0.26-0.37
  participation ratio        how many dimensions the stage actually uses

Stages: A0 (one line encoder) -- line, graph, turn, pointer key. Split --
signature, description and name stage outputs, then the same three. And for
a split model, the gate's mean weight on each stage at tool slots whose
reference tool has twins, named rows vs opaque-name rows (teacher-forced).

  python stage_report.py --ckpt runs/s6_SPt/best.pt --cache data_cache_s6 \
      [--limit 800] [--json out/s6_SPt_geometry.json]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate import Split, load_model
from stages import participation_ratio
from train import called_tools, shift_right, MASK_ID


def pair_stats(V: torch.Tensor, groups: torch.Tensor, live: torch.Tensor, rng):
    """(twin cosines, unrelated cosines, twin/unrelated distance ratios) for
    one row's vectors V (M, w)."""
    tw, un, ratio = [], [], []
    idx = live.nonzero().flatten().tolist()
    Vn = F.normalize(V.float(), dim=-1)
    for i in idx:
        mates = [j for j in idx if j != i and int(groups[j]) == int(groups[i])]
        others = [j for j in idx if int(groups[j]) != int(groups[i])]
        if not mates or not others:
            continue
        j, k = rng.choice(mates), rng.choice(others)
        tw.append(float(Vn[i] @ Vn[j]))
        un.append(float(Vn[i] @ Vn[k]))
        du = float((V[i] - V[k]).norm())
        if du > 0:
            ratio.append(float((V[i] - V[j]).norm()) / du)
    return tw, un, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--id-contains", default="+decoy")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    sp = Split(Path(args.cache), args.split, dev)
    model = load_model(Path(args.ckpt), dev).eval()
    c = model.c
    rng = random.Random(0)
    rows = [i for i, m in enumerate(sp.meta) if args.id_contains in m["task_id"]][:args.limit]
    acc = {}
    vecs = {}
    gates = {"named": [], "opaque": []}

    def add(stage, V, grp, live):
        tw, un, ra = pair_stats(V, grp, live, rng)
        a = acc.setdefault(stage, ([], [], []))
        a[0].extend(tw); a[1].extend(un); a[2].extend(ra)
        vecs.setdefault(stage, []).append(V[live][:24].float())

    for i in rows:
        inp = sp.inputs(i, model)
        grp = inp["sig_group"][0].long()
        live = grp >= 0
        n_t = int(inp["n_tool"][0])
        live[n_t:] = False
        if c.split:
            tv, st = model.encode_split_tools(inp)
            add("1 signature stage", model.sig_stage(inp["tool_sig_tok"], packed=c.pack)[0], grp, live)
            add("1 description stage", st["desc"][0], grp, live)
            add("1 name stage", st["name"][0], grp, live)
            add("2 combined (tool_proj)", tv[0], grp, live)
        else:
            tv, st = model.encode_lines(inp["tool_tok"], 0), None
            add("1 line encoder", tv[0], grp, live)
        gv, fv = model.encode_world(inp["tool_tok"], inp["field_tok"], inp["adj"],
                                    tool_vecs=tv if c.split else None)
        add("3 after graph pass", gv[0], grp, live)
        mem = model.encode_inputs(inp)
        add("4 after turn encoder", mem.pointers[0, :c.max_tool], grp, live)
        table = model.slot_table(mem)
        keys = model.k(table[:, c.n_kw:c.n_kw + c.max_tool])
        add("5 pointer key", keys[0], grp, live)
        if c.split:
            tgt = sp.target(i).to(dev)
            model.decode(inp, shift_right(tgt, bos=MASK_ID), mem=mem)
            g = model.last_gate[0]                                     # (C, 3)
            t = tgt[0].long() - c.n_kw
            opaque = bool(sp.meta[i].get("opaque"))
            for pos in ((t >= 0) & (t < c.max_tool)).nonzero().flatten().tolist():
                tool = int(t[pos])
                if int((grp == grp[tool]).sum()) >= 2:
                    gates["opaque" if opaque else "named"].append(g[pos].tolist())

    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
    out = {"ckpt": args.ckpt, "rows": len(rows), "stages": {}}
    print(f"{args.ckpt}: {len(rows)} rows of {args.split} ({args.id_contains})")
    print(f"  {'stage':28s} {'cos twin':>9s} {'cos unrel':>9s} {'twin/unrel dist':>16s} {'PR':>6s} (of width)")
    for stage in sorted(acc):
        tw, un, ra = acc[stage]
        V = torch.cat(vecs[stage])
        pr = participation_ratio(V)
        out["stages"][stage] = {"cos_twin": med(tw), "cos_unrelated": med(un),
                                "dist_ratio": med(ra), "participation_ratio": pr,
                                "width": V.size(1), "pairs": len(tw)}
        print(f"  {stage:28s} {med(tw):9.3f} {med(un):9.3f} {med(ra):16.3f} {pr:6.1f} ({V.size(1)})")
    if c.split:
        for k, gs in gates.items():
            if gs:
                m = torch.tensor(gs).mean(0).tolist()
                out[f"gate_{k}"] = {"signature": m[0], "description": m[1], "name": m[2], "n": len(gs)}
                print(f"  gate at twin tool slots, {k:6s} rows (n={len(gs)}): "
                      f"signature {m[0]:.2f}  description {m[1]:.2f}  name {m[2]:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
