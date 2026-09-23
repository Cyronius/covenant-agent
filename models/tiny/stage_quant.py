"""Step 5 of the description-reading plan, the free half: 8-bit vs 4-bit for
the pretrained description and name stages, graded on their own.

Because the stages have a task of their own (the twin benchmark), their
weight format can be settled on the laptop before any planner is trained in
it. This rounds a step-3 checkpoint's stage weights post-training -- every
attention and feed-forward matmul at the format under test, token tables at
int8, activations int8 per token, norms and the gate in float (the same split
as quant.py applies to the planner) -- and reruns the benchmark. Within a
point of full precision, the format needs no quantisation-aware training;
otherwise `stage_pretrain.py --weights <fmt>` trains in it from the start.

  python stage_quant.py --stages runs/stages_d256/stages.pt --cache data_cache_s6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

import quant
from stage_pretrain import build_head, load_part, twin_eval


def quantize_head(head, mode: str, act_bits: int = 8) -> dict:
    """Round the two stages in place: layers at `mode`, token tables int8."""
    n_block = n_end = 0
    for stage in (head.desc, head.name):
        n_block += quant.quantize_tree(stage.layers, mode, act_bits)
        P.register_parametrization(stage.emb, "weight", quant.WeightQuant("int8"),
                                   unsafe=True)
        n_end += stage.emb.weight.numel()
    return {"block_params": n_block, "table_params": n_end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--limit-eval", type=int, default=3000)
    ap.add_argument("--modes", default="fp,int8,u4")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cache = Path(args.cache)
    meta = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    lay = meta["layout"]
    ck = torch.load(args.stages, map_location="cpu")
    exam = load_part(cache, "holdout", args.limit_eval)
    teacher = None
    if (cache / "teacher.pt").exists():
        teacher = torch.load(cache / "teacher.pt")["table"].float()
    out = {"stages": args.stages, "modes": {}}
    for mode in args.modes.split(","):
        head = build_head(ck["cfg"], meta["in_vocab"], meta["in_pad"])
        head.load_state_dict(ck["head"])
        rep = {} if mode == "fp" else quantize_head(head, mode)
        res = twin_eval(head, exam, meta["in_pad"], lay["n_kw"], lay["max_tool"], teacher)
        out["modes"][mode] = {**rep, **res}
        print(f"{mode:5s}  twins {res['acc_comb']:.1%}  flip-slot twins "
              f"{res.get('acc_flip_comb') or 0:.1%}  desc {res['acc_desc']:.1%}  "
              f"name {res['acc_name']:.1%}  (n={res['n_decisions']}, flip n={res['n_flip']})",
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
