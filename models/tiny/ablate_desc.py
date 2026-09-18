"""Does the model read the tool descriptions at all?

Tool and field selection measured at chance in the first two runs. There are two
ways that happens. Either the model reads the descriptions and is bad at matching
them to the request, or it never learned to read them and is guessing from
structure alone.

This separates the two, and needs no training. Take a trained model and permute
the descriptions among the tool lines, so `T5` now carries some other tool's
description while every symbol, type signature and effect stays exactly where it
was. Then generate again.

  score unchanged -> the descriptions were never being used
  score drops     -> they were being used, just not well enough

A drop is the better news: it means the signal is there and the model is weak.
No change means the input channel that carries the whole task is being ignored,
and no amount of tuning the output side can matter.

    python ablate_desc.py --ckpt runs/ar_s0/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

import pickle

from evaluate import load_model
from prep import compact, symbol_positions
from sample import Trace, ar_sample, diffusion_sample, to_text
from tok import OutVocab


def shuffle_request(src: str, other: str) -> str:
    """Replace the REQUEST line with another task's request.

    The companion question to shuffling descriptions. If the program does not
    change when the request changes, the model is emitting a structural prior
    and reading none of the language it was given.
    """
    lines = src.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("REQUEST:"):
            lines[i] = [o for o in other.splitlines() if o.startswith("REQUEST:")][0]
            break
    return "\n".join(lines) + "\n"


def shuffle_descriptions(src: str, rng: random.Random) -> str:
    """Permute the text after ' :: ' among the TOOLS lines only.

    Fields and constants keep their descriptions, so this isolates the one thing
    the model must read to choose a tool. Everything structural -- symbol names,
    parameter types, effects, ordering -- is untouched, so a model relying on
    structure is unaffected by construction.
    """
    lines = src.splitlines()
    idx = [i for i, l in enumerate(lines)
           if l.startswith("T") and " :: " in l and l[1:2].isdigit()]
    if len(idx) < 2:
        return src + "\n"
    descs = [lines[i].split(" :: ", 1)[1] for i in idx]
    shifted = descs[1:] + descs[:1]        # derangement: nothing keeps its own
    rng.shuffle(shifted)
    for i, dsc in zip(idx, shifted):
        lines[i] = lines[i].split(" :: ", 1)[0] + " :: " + dsc
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/ablate.jsonl")
    ap.add_argument("--mode", choices=["desc", "request"], default="desc")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache = Path(args.cache)
    cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    ov = OutVocab.load(cache / "out_vocab.json")
    tk = Tokenizer.from_file(str(cache / "in_tok.json"))
    meta = json.loads((cache / f"{args.split}_meta.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if cfg.get("binding", "flat") != "flat":
        # The structural cache holds per-line tensors, not one token stream, so
        # a permuted context has to be re-split and re-encoded per line. Not
        # built yet; refuse rather than silently ablate the wrong thing.
        raise SystemExit("ablate_desc.py supports the flat binding only; "
                         "this cache is structural")
    model = load_model(Path(args.ckpt), device, ov)
    if model.c.binding != "flat":
        raise SystemExit("ablate_desc.py supports flat checkpoints only")
    arm = "ar" if model.c.causal else "diffusion"

    # Rebuild the inputs from the cached raw rows, so the descriptions can be
    # permuted before tokenizing. rows.pkl ships with the cache, which means this
    # runs on a pod that has never seen the corpus.
    from harness.context import TaskContext, serialize_context
    rows_raw = pickle.load(open(cache / "rows.pkl", "rb"))[args.split]
    by_id = {r["id"]: serialize_context(r["request"], TaskContext.from_json(r["context"]))
             for r in rows_raw}
    rng = random.Random(args.seed)

    rows = []
    for m in meta[:args.limit]:
        raw = by_id.get(m["task_id"])
        if raw is None:
            continue
        if args.mode == "desc":
            alt = shuffle_descriptions(compact(raw), rng)
        else:
            other = by_id[rng.choice([k for k in by_id if k != m["task_id"]])]
            alt = shuffle_request(compact(raw), compact(other))
        for tag, text in (("intact", compact(raw)), ("shuffled", alt)):
            enc = tk.encode(text)
            ids = enc.ids[:cfg["max_in"]]
            ids += [tk.token_to_id("<pad>")] * (cfg["max_in"] - len(ids))
            src = torch.tensor([ids], device=device)
            pad = (src == tk.token_to_id("<pad>"))
            inputs = {"src": src, "pad": pad}
            if model.c.pointer:
                inputs["sym"] = torch.tensor([symbol_positions(text, enc, ov, cfg["max_in"])],
                                             device=device)
            tr = Trace()
            if arm == "diffusion":
                canvas, tr = diffusion_sample(model, inputs, ov, steps=args.steps, trace=tr)
            else:
                canvas, tr = ar_sample(model, inputs, ov, trace=tr)
            rows.append({**m, "condition": tag, "program": to_text(canvas, ov),
                         "passes": tr.passes, "canvas": canvas[0].tolist(),
                         "unmask_step": tr.unmask_step, "repairs": 0,
                         "steps": tr.steps, "compiled_inline": False,
                         "reference": ""})

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # How often does permuting the descriptions change the program at all?
    pairs = {}
    for r in rows:
        pairs.setdefault(r["task_id"], {})[r["condition"]] = r["program"]
    both = [v for v in pairs.values() if len(v) == 2]
    same = sum(1 for v in both if v["intact"] == v["shuffled"])
    print(f"{len(both)} tasks generated twice ({arm})")
    print(f"  identical program despite shuffled {args.mode}: {same} ({same/max(len(both),1):.1%})")
    print("\n  A high number here means the descriptions are not being read.")
    print(f"\nwrote {dest}  -- score each condition separately to compare goal success")


if __name__ == "__main__":
    main()
