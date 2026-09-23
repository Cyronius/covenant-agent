"""Are signature twins' line vectors too close together for the pointer to split?

The owner's hypothesis: the encoder has enough width, but twin tools' vectors
cluster so tightly that one dot product cannot separate them. This measures
it on a trained checkpoint, at every reference tool slot, teacher-forced.

For each held-out task in the checkpoint's own cache, every called tool gets
signature twins beside it -- built with the exam's own description bank
(harness/decoys.py), or with a real description borrowed from another tool of
the same effect class -- so the twin's line differs from the real one by the
description alone. Then, per slot:

  geometry   cosine(real, twin) of the line vector (what the pointer reads)
             and of the pointer key; the twin distance as a fraction of the
             ordinary distance between two unrelated tools in the same task
  pointer    logit(real) - logit(twin), split into |q| * |dk| * cos(q, dk):
             is the gap small because the keys coincide (dk small), or
             because the query does not look along the direction that
             separates them (cos small)?
  spectrum   how many directions the keys actually use (participation
             ratio), and how much of the twin difference lives in the
             directions the keys vary least

  python twin_geometry.py --ckpt runs/X/best.pt --cache data_cache_X
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from canvas import Layout, TaskCodec, context_symbols, load_keywords  # noqa: E402
from corpus import COVENANT  # noqa: E402
from evaluate import load_model  # noqa: E402
from model import TAG_CANVAS, TAG_TOOL  # noqa: E402
from prep import encode_one  # noqa: E402

sys.path.insert(0, str(COVENANT))
from harness.context import TaskContext, serialize_context  # noqa: E402
from harness.decoys import DECOY_OPS, _article, _bank_key, _noun_for  # noqa: E402

MASK_ID = 1
CALL = re.compile(r"CALL (T\d+)")


def pointer_parts(model, inputs, canvas):
    """decode() up to the pointer, keeping what it throws away."""
    c = model.c
    mem = model.encode_inputs(inputs)
    table = model.slot_table(mem)
    x = table.gather(1, canvas.long().unsqueeze(-1).expand(-1, -1, c.d))
    x = x + model.out_pos + model.tag_emb.weight[TAG_CANVAS]
    for i in range(c.dec_loops):
        if model.loop_emb is not None:
            x = x + model.loop_emb.weight[min(i, model.loop_emb.num_embeddings - 1)]
        for layer in model.dec:
            x = layer(x, mem.mem, mem.pad, model.causal_mask)
    h = model.dec_norm(x)
    keys = model.k(table[:, c.n_kw:])
    q = model.q(h)
    logits = torch.einsum("bcd,bjd->bcj", q, keys) / math.sqrt(c.d)
    full = model.decode(inputs, canvas)              # the model's own answer
    raw = model.encode_lines(inputs["tool_tok"], TAG_TOOL)[0]
    world = model.encode_world(inputs["tool_tok"], inputs["field_tok"], inputs["adj"])[0][0]
    stages = {"1 line encoder (CLS pool)": raw,
              "2 after the schema graph pass": world,
              "3 after the turn encoder (request mixed in)": mem.pointers[0],
              "4 pointer key": keys[0]}
    return table[0, c.n_kw:], keys[0], q[0], logits[0], full[0], stages


def add_twins(ctx: dict, called: list[str], per: int, cap: int,
              rng: random.Random, borrow: dict | None) -> dict[str, list[str]]:
    """Give each called tool twins -- same params, same effects, a different
    description. Returns {real sym: [twin syms]}.

    A twin takes the place of a tool the reference does not call (its symbol
    and its line position), so the tool count never grows: the held-out
    worlds already fill every pointer row the layout has."""
    by_sym = {t["sym"]: t for t in ctx["tools"]}
    spare = [t["sym"] for t in ctx["tools"] if t["sym"] not in called]
    rng.shuffle(spare)
    out: dict[str, list[str]] = {}
    for s in called:
        t = by_sym[s]
        key = _bank_key(t)
        if key is None:
            continue
        if borrow is not None:
            pool = [d for d in borrow.get(key, []) if d != t["desc"]]
            descs = rng.sample(pool, min(per, len(pool)))
        else:
            noun, _ = _noun_for(t, {})
            verbs = rng.sample(sorted(DECOY_OPS[key]), min(per, len(DECOY_OPS[key])))
            descs = [DECOY_OPS[key][v].format(a_noun=_article(noun), noun=noun)
                     for v in verbs]
        for d in descs:
            if not spare:
                return out
            slot = spare.pop()
            tw = copy.deepcopy(t)
            tw["sym"], tw["name"], tw["desc"] = slot, f"twin_{slot}", d
            i = next(i for i, x in enumerate(ctx["tools"]) if x["sym"] == slot)
            ctx["tools"][i] = tw
            out.setdefault(s, []).append(slot)
    return out


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--n", type=int, default=400, help="tasks to probe")
    ap.add_argument("--twins", choices=["bank", "borrowed"], default="bank",
                    help="bank: the exam's decoy descriptions; borrowed: a real "
                         "description of another tool with the same effect class")
    ap.add_argument("--per", type=int, default=1, help="twins per called tool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sandbox import register_themes
    register_themes()
    torch.set_grad_enabled(False)
    cache = Path(args.cache)
    cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    layout = Layout.from_dict(cfg["layout"])
    dims = {"max_line": cfg["max_line"], "max_req": cfg["max_req"],
            "max_reg": cfg.get("max_reg", 8)}
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(cache / "in_tok.json"))
    tk.no_truncation(); tk.no_padding()
    keywords = load_keywords(cache / "keywords.json")
    model = load_model(Path(args.ckpt), torch.device("cpu"))
    model.eval()
    d = model.c.d
    rows = pickle.load(open(cache / "rows.pkl", "rb"))[args.split]
    rows = [r for r in rows if len(r["reference"]["segments"]) == 1]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    borrow = None
    if args.twins == "borrowed":
        # real descriptions from the training worlds, keyed by effect class.
        # rows.pkl keeps no train split, but val and test are rows of the
        # same worlds, so their tools are ones the model trained on.
        borrow = defaultdict(list)
        splits = pickle.load(open(cache / "rows.pkl", "rb"))
        for r in splits["val"] + splits["test"]:
            for t in r["context"]["tools"]:
                k = _bank_key(t)
                if k and t["desc"] not in borrow[k]:
                    borrow[k].append(t["desc"])

    S = defaultdict(list)          # per-slot records
    all_keys, all_dk = [], []
    stage_vecs: dict = {}
    clipped = total_lines = 0
    tasks = 0
    skipped = defaultdict(int)
    for row in rows:
        if tasks >= args.n:
            break
        ref = row["reference"]["segments"][0]
        called = list(dict.fromkeys(CALL.findall(ref)))
        ctx = copy.deepcopy(row["context"])
        twins = add_twins(ctx, called, args.per, layout.max_tool, rng, borrow)
        if not twins:
            continue
        syms = context_symbols(ctx)
        source = serialize_context(row["request"], TaskContext.from_json(ctx))
        for line in source.splitlines():
            if re.match(r"T\d+ ", line):
                total_lines += 1
                clipped += len(tk.encode(line.split(" ", 1)[1]).ids) > dims["max_line"]
        try:
            inputs = encode_one(source, syms, tk, layout, dims)
            codec = TaskCodec(keywords, layout, **syms)
            tgt = codec.encode(ref, length=cfg["canvas"])
        except (SystemExit, Exception) as exc:  # noqa: BLE001
            skipped[type(exc).__name__] += 1       # counted, never swallowed
            continue
        tasks += 1
        tgt_t = torch.tensor([tgt])
        canvas = torch.cat([torch.full_like(tgt_t[:, :1], MASK_ID), tgt_t[:, :-1]], 1)
        vec, keys, q, logits, full, stages = pointer_parts(model, inputs, canvas)
        n_tool = len(syms["tools"])
        kv = keys[:n_tool]
        all_keys.append(kv)
        # ordinary distance: two tools of this task that are not twins of anything
        twin_set = {x for v in twins.values() for x in v} | set(twins)
        plain = [codec.tool_id[s] for s in syms["tools"] if s not in twin_set]
        ord_d = [float((kv[i] - kv[j]).norm()) for a, i in enumerate(plain)
                 for j in plain[a + 1:]]
        ord_c = [cos(vec[i], vec[j]) for a, i in enumerate(plain) for j in plain[a + 1:]]
        mean_ord = sum(ord_d) / len(ord_d) if ord_d else float("nan")
        for p, jid in enumerate(tgt):
            if not (layout.tool0 <= jid < layout.field0):
                continue
            ri = jid - layout.tool0
            real = syms["tools"][ri]
            if real not in twins:
                continue
            for tw in twins[real]:
                ti = codec.tool_id[tw]
                dk = keys[ri] - keys[ti]
                all_dk.append(dk)
                gap = float(logits[p, ri] - logits[p, ti])
                pred = int(full[p].argmax())
                S["gap"].append(gap)
                S["picked_real"].append(pred == jid)
                S["picked_twin"].append(pred == layout.tool0 + ti)
                S["cos_vec"].append(cos(vec[ri], vec[ti]))
                S["cos_key"].append(cos(keys[ri], keys[ti]))
                S["ord_cos_vec"].append(sum(ord_c) / len(ord_c) if ord_c else float("nan"))
                S["twin_over_ord"].append(float(dk.norm()) / mean_ord)
                S["dk_rel"].append(float(dk.norm() / keys[ri].norm()))
                S["cos_q_dk"].append(cos(q[p], dk))
                S["qnorm_dknorm"].append(float(q[p].norm() * dk.norm()) / math.sqrt(d))
                for name, V in stages.items():
                    V = V[:n_tool]
                    od = [float((V[i] - V[j]).norm()) for a, i in enumerate(plain)
                          for j in plain[a + 1:]]
                    S["st_cos|" + name].append(cos(V[ri], V[ti]))
                    S["st_ratio|" + name].append(float((V[ri] - V[ti]).norm())
                                                 / (sum(od) / len(od)))
                    stage_vecs.setdefault(name, []).append(V)

    n = len(S["gap"])
    if skipped:
        print("tasks the layout could not encode:", dict(skipped))
    if not n:
        print("no probe slots"); return 1
    fail = [i for i in range(n) if S["picked_twin"][i]]
    ok = [i for i in range(n) if S["picked_real"][i]]

    def row(name, key, fmt="{:.3f}"):
        def m(ix):
            xs = [S[key][i] for i in ix]
            return (fmt.format(pct(xs, .5)) + " [" + fmt.format(pct(xs, .1)) + ", "
                    + fmt.format(pct(xs, .9)) + "]") if xs else "-"
        print(f"  {name:44s} {m(range(n)):>28s} {m(ok):>28s} {m(fail):>28s}")

    print(f"checkpoint d={d}  cache {cache.name}  twins={args.twins} x{args.per}")
    print(f"{tasks} tasks, {n} (real, twin) slot pairs; teacher-forced")
    print(f"tool lines longer than max_line={dims['max_line']} (description clipped): "
          f"{clipped}/{total_lines}")
    print(f"picked the real tool {len(ok)}/{n} ({len(ok)/n:.1%}); "
          f"picked the twin {len(fail)}/{n} ({len(fail)/n:.1%})")
    print(f"\n  {'median [p10, p90]':44s} {'all':>28s} {'real picked':>28s} {'twin picked':>28s}")
    row("cos(line vector: real, twin)", "cos_vec")
    row("cos(line vector: two unrelated tools)", "ord_cos_vec")
    row("cos(pointer key: real, twin)", "cos_key")
    row("|k_real - k_twin| / |k_real|", "dk_rel")
    row("twin distance / unrelated-pair distance", "twin_over_ord")
    row("logit gap real - twin", "gap")
    row("  = |q||dk|/sqrt(d)  (max possible gap)", "qnorm_dknorm")
    row("  x cos(q, dk)       (how much is used)", "cos_q_dk")

    print()
    print("where do twins collapse? (median; real picked | twin picked)")
    print(f"  {'stage':46s} {'cos(real, twin)':>22s} {'twin / unrelated dist':>24s} {'used dims':>10s}")
    for name in stage_vecs:
        V = torch.cat(stage_vecs[name]); V = V - V.mean(0)
        e = torch.linalg.eigvalsh(V.T @ V / len(V))
        pr_ = float(e.sum() ** 2 / (e ** 2).sum())
        cc = [S["st_cos|" + name][i] for i in ok], [S["st_cos|" + name][i] for i in fail]
        rr = [S["st_ratio|" + name][i] for i in ok], [S["st_ratio|" + name][i] for i in fail]
        print(f"  {name:46s} {pct(cc[0], .5):9.3f} | {pct(cc[1], .5):6.3f}      "
              f"{pct(rr[0], .5):9.2f} | {pct(rr[1], .5):6.2f}   {pr_:8.1f}")

    K = torch.cat(all_keys)
    K = K - K.mean(0)
    ev, evec = torch.linalg.eigh(K.T @ K / len(K))
    ev, evec = ev.flip(0), evec.flip(1)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    DK = torch.stack(all_dk)
    energy = (DK @ evec) ** 2
    share = energy.sum(0) / energy.sum()
    top = int(max(1, round(pr)))
    print(f"\nkey spectrum over {len(K)} tool keys, d={d}: participation ratio "
          f"{pr:.1f} directions")
    print(f"  top {top} key directions hold {float(ev[:top].sum()/ev.sum()):.1%} "
          f"of key variance and {float(share[:top].sum()):.1%} of twin-difference energy")
    print(f"  the {d - top} low-variance directions hold "
          f"{float(share[top:].sum()):.1%} of twin-difference energy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
