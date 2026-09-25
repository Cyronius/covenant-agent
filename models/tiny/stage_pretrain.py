"""Train the description and name stages on their own (description-reading
plan step 3), before any planner sees them.

Nothing in these stages' objective needs the planner. "Which tool does this
request mean" comes straight from the corpus: each row gives a request, the
tools its reference calls, and every other tool of the task as a negative --
the twins and authored decoys first, because they are already there. The
teacher's similarities are the second target and need no labels at all.

  python stage_pretrain.py --cache data_cache_s6 --out runs/stages_d256 \
      [--desc-w 256 --desc-layers 4 --name-w 128 --name-layers 2]
      [--pool cls|mean] [--open-pairs] [--epochs 3] [--lam-rel 1.0]

Graded standalone on the cache's holdout split (the unseen exam worlds) by
the twin benchmark: for every reference call whose tool has signature twins,
pick among the twins by score. The bar is the teacher's own zero-shot pick on
the same decisions from descriptions alone (65.0% on the R10 exam); the
teacher numbers are recomputed here from teacher.pt, so the bar moves with
the exam. Also reported: participation ratio of the description vectors
(the proxy's line encoder used ~4 of 128 dimensions) and, on the opaque-name
rows, where the gate puts its weight.

Writes <out>/stages.pt (loadable by `train.py --stages-from`) and
<out>/log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from stages import (ReadHead, TextStage, multi_positive_nce, participation_ratio,
                    relational_loss, twin_nce)
from train import called_tools

COLS = ("tool_desc_tok", "tool_name_tok", "req_tok", "n_tool", "tgt",
        "sig_group", "flip_tool", "t_desc", "t_name", "t_req")


def load_part(cache: Path, name: str, limit: int | None = None,
              ids: tuple[str, ...] | None = None) -> dict:
    """A cached split's columns. `ids`: keep only rows whose task id contains
    one of these (the exam's decoyed and flip halves -- the plain half has
    no twins to decide between), then the first `limit`."""
    d = torch.load(cache / f"{name}.pt", mmap=True)
    meta = json.loads((cache / f"{name}_meta.json").read_text(encoding="utf-8"))
    keep = [i for i, m in enumerate(meta)
            if ids is None or any(x in m["task_id"] for x in ids)]
    keep = keep[:limit] if limit else keep
    idx = torch.tensor(keep, dtype=torch.long)
    out = {k: d[k][idx] for k in COLS if k in d}
    out["opaque"] = torch.tensor([bool(meta[i].get("opaque")) for i in keep])
    if "flip_tool" in out and ids is not None:
        out["flip_tool"] = out["flip_tool"] & readable(cache, name, keep,
                                                       out["flip_tool"].shape[1])
    return out


def readable(cache: Path, split: str, keep: list[int], max_tool: int) -> torch.Tensor:
    """(len(keep), max_tool): flip-slot tools whose own request templates the
    row's request carries (`domains.request_fits`). A flip-slot call the
    request never names -- "send it to {name}" beside notify / invite /
    email -- has three right answers, and S6 has 14% of them (R11 §3); a
    grounding number counts only the decisions a reader could get right.
    Matched by description, which opaque-name rows keep."""
    import pickle
    from corpus import COVENANT  # also puts the checkout on sys.path
    from data.gen import domains
    rows = pickle.load(open(cache / "rows.pkl", "rb")).get(split)
    if rows is None:
        # prep keeps no raw train rows; every segment of a paused task carries
        # its task's row (corpus.load), so look them up in the corpus by id
        meta = json.loads((cache / f"{split}_meta.json").read_text(encoding="utf-8"))
        cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
        want = {meta[i]["task_id"].split("#")[0] for i in keep}
        by_id = {}
        with open(COVENANT / "data" / cfg["corpus"], encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] in want:
                    by_id[r["id"]] = r
        rows = {i: by_id[meta[i]["task_id"].split("#")[0]] for i in keep}
    specs = {}
    for path in sorted((COVENANT / "data/gen/themes").glob("*.json")):
        th = json.loads(path.read_text(encoding="utf-8"))
        for slot in domains.FLIP_VERBS:
            real = th["tools"][slot]
            for sp in [real] + list(real.get("decoys") or []):
                specs[(th["domain"], sp["desc"])] = (slot, sp)
    out = torch.zeros(len(keep), max_tool, dtype=torch.bool)
    for n, i in enumerate(keep):
        r = rows[i]
        tools = sorted(r["context"]["tools"], key=lambda t: int(t["sym"][1:]))
        for j, t in enumerate(tools[:max_tool]):
            hit = specs.get((r["world"], t["desc"]))
            if hit and domains.request_fits(r["request"], hit[0], hit[1]):
                out[n, j] = True
    return out


def build_head(cfg: dict, in_vocab: int, pad: int) -> ReadHead:
    L = max(cfg["max_desc"], cfg["max_req"])
    desc = TextStage(in_vocab, cfg["desc_w"], cfg["desc_layers"], L, pad,
                     dropout=cfg.get("dropout", 0.1), pool=cfg["pool"])
    name = TextStage(in_vocab, cfg["name_w"], cfg["name_layers"],
                     max(cfg["max_name"], cfg["max_req"]), pad,
                     dropout=cfg.get("dropout", 0.1), pool=cfg["pool"])
    return ReadHead(desc, name)


def load_stages(model, path) -> None:
    """Put pretrained stage weights into a split StructuralModel."""
    ck = torch.load(path, map_location="cpu")
    c = model.c
    for key, want in (("desc_w", c.desc_w), ("desc_layers", c.desc_layers),
                      ("name_w", c.name_w), ("name_layers", c.name_layers),
                      ("pool", c.stage_pool)):
        if ck["cfg"][key] != want:
            raise SystemExit(f"--stages-from {path}: {key}={ck['cfg'][key]}, "
                             f"the planner is built with {want}")
    sd = ck["head"]
    for stage, prefix in ((model.desc_stage, "desc."), (model.name_stage, "name.")):
        stage.load_state_dict({k[len(prefix):]: v for k, v in sd.items()
                               if k.startswith(prefix)})
    with torch.no_grad():
        model.stage_scale.copy_(sd["scale"])


def teacher_token_init(tk, model: str, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Model2Vec-style static embeddings for our BPE vocabulary, from the
    teacher's input table: each token's text, re-tokenized by the teacher,
    averaged over its word-piece rows, projected onto the top `width`
    principal components and scaled to the stage's init. Returns (table,
    covered mask). Unseen-world words fragment more under our BPE (84% vs
    92% one-token, R10 §8), so the pieces arriving with some meaning
    already is the point."""
    from transformers import AutoModel, AutoTokenizer
    btok = AutoTokenizer.from_pretrained(model)
    E = AutoModel.from_pretrained(model).embeddings.word_embeddings.weight.detach()
    vocab = tk.get_vocab()
    V = torch.zeros(len(vocab), E.size(1))
    ok = torch.zeros(len(vocab), dtype=torch.bool)
    for piece, i in vocab.items():
        text = piece.replace("Ġ", " ").replace("Ċ", " ").strip()
        if not text or text.startswith("<"):
            continue
        ids = btok(text, add_special_tokens=False)["input_ids"]
        if ids:
            V[i] = E[ids].mean(0)
            ok[i] = True
    X = V[ok] - V[ok].mean(0)
    comps = torch.linalg.svd(X, full_matrices=False)[2][:width].T     # (384, w)
    P = torch.zeros(len(vocab), width)
    P[ok] = X @ comps
    P[ok] = P[ok] / P[ok].std() * 0.02
    return P, ok


def encode_unique(stage: TextStage, tok: torch.Tensor, query: bool = False) -> torch.Tensor:
    """(B, M, T) -> (B, M, w), encoding each distinct token row once: one
    world's tools recur across a batch, so this is most of the saving."""
    B, M, T = tok.shape
    flat = tok.reshape(B * M, T)
    uniq, inv = torch.unique(flat, dim=0, return_inverse=True)
    v = stage(uniq.unsqueeze(0), query=query).squeeze(0)
    return v[inv].reshape(B, M, -1)


def forward(head: ReadHead, b: dict, pad: int):
    """Scores and vectors for a batch: (combined, desc-only, name-only,
    gate, vectors)."""
    req = b["req_tok"].long()
    # trim the columns to what the batch uses
    def trim(t):
        live = (t != pad)
        n = int(live.any(0).nonzero().max()) + 1 if live.any() else 1
        return t[..., :n]
    desc_tok = trim(b["tool_desc_tok"].long())
    name_tok = trim(b["tool_name_tok"].long())
    M = int(b["n_tool"].max())
    desc_tok, name_tok = desc_tok[:, :M], name_tok[:, :M]
    req = trim(req)
    q_d = head.desc(req, query=True)
    q_n = head.name(req, query=True)
    d = encode_unique(head.desc, desc_tok)
    n = encode_unique(head.name, name_tok)
    enc = {"q_desc": q_d, "desc": d, "q_name": q_n, "name": n}
    comb, g = head.scores(enc)
    s = head.scale.exp().clamp(max=100)
    sd = s * torch.einsum("bw,bmw->bm", F.normalize(q_d, dim=-1), F.normalize(d, dim=-1))
    sn = s * torch.einsum("bw,bmw->bm", F.normalize(q_n, dim=-1), F.normalize(n, dim=-1))
    return comb, sd, sn, g, enc, M


def batch_losses(head, b, pad, n_kw, max_tool, teacher, lam_rel, flip_weight=0.0):
    comb, sd, sn, g, enc, M = forward(head, b, pad)
    live = torch.arange(M)[None] < b["n_tool"].long()[:, None]
    pos = called_tools(b["tgt"], n_kw, max_tool)[:, :M]
    grp = b["sig_group"][:, :M].long()
    clive = live
    if "unread" in b:
        # a flip-slot call its request never names is neither right nor wrong
        # to a reader: out of the positives AND the candidates, or it would
        # teach "the authored tool loses" instead of "the authored tool wins"
        u = b["unread"][:, :M]
        pos, clive = pos & ~u, live & ~u
    out = {"nce": multi_positive_nce(comb, pos, clive),
           "nce_desc": multi_positive_nce(sd, pos, clive),
           "nce_name": multi_positive_nce(sn, pos, clive),
           "twin": twin_nce(comb, pos, clive, grp),
           "twin_desc": twin_nce(sd, pos, clive, grp)}
    if flip_weight and "flip_tool" in b:
        # the flip slots' twins on their own: the decisions the rest of the
        # twin loss is dominated away from (every other twin group is won by
        # recognising the tool requests always ask for, R11 §3)
        fpos = pos & b["flip_tool"][:, :M]
        out["twin_flip"] = twin_nce(comb, fpos, clive, grp)
    if lam_rel and teacher is not None and "t_desc" in b:
        ridx = b["t_req"].long()
        for part, key in (("desc", "t_desc"), ("name", "t_name")):
            idx = b[key].long()[:, :M]
            vec = torch.cat([enc[f"q_{part}"].unsqueeze(1), enc[part]], 1)
            tt = torch.cat([teacher[ridx.clamp(min=0)].unsqueeze(1),
                            teacher[idx.clamp(min=0)]], 1)
            lv = torch.cat([(ridx >= 0).unsqueeze(1), live & (idx >= 0)], 1)
            out[f"rel_{part}"] = relational_loss(vec, tt, lv)
    total = (out["nce"] + out["nce_desc"] + out["nce_name"] + out["twin"]
             + out["twin_desc"] + flip_weight * out.get("twin_flip", 0.0)) + lam_rel * sum(
        v for k, v in out.items() if k.startswith("rel_"))
    return total, out


# ---------------------------------------------------------------- open pairs
def open_part(tk, cfg, desc_chars, pad, teacher_model=None, limit=None) -> dict:
    """data/open_pairs as extra (request, tools, called) rows, in the cache's
    tokenizer. Only rows whose reference calls a tool. Tokens past a cap are
    cut: this is pretraining text, not a program the planner must bind."""
    from corpus import COVENANT
    from prep import compact_desc, teacher_name, QUERY_PREFIX
    rows = []
    for f in ("glaive_12k", "hermes_full", "toolace_4k"):
        for line in open(COVENANT / "data/open_pairs" / f"{f}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            segs = (r.get("reference") or {}).get("segments") or []
            calls = set(re.findall(r"CALL (T\d+)", " ".join(segs)))
            tools = sorted(r["context"]["tools"], key=lambda t: int(t["sym"][1:]))
            if not calls or not tools:
                continue
            rows.append((r["request"], tools, calls))
    random.Random(0).shuffle(rows)
    rows = rows[:limit] if limit else rows
    MT = max(len(t) for _, t, _ in rows)
    DL, NL, RL = cfg["max_desc"], cfg["max_name"], cfg["max_req"]
    N = len(rows)
    out = {"tool_desc_tok": torch.full((N, MT, DL), pad, dtype=torch.int16),
           "tool_name_tok": torch.full((N, MT, NL), pad, dtype=torch.int16),
           "req_tok": torch.full((N, RL), pad, dtype=torch.int16),
           "n_tool": torch.zeros(N, dtype=torch.int16),
           "called": torch.zeros(N, MT, dtype=torch.bool)}
    texts = []
    for n, (req, tools, calls) in enumerate(rows):
        ids = tk.encode(req).ids[:RL]
        out["req_tok"][n, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
        descs = [compact_desc(t["desc"], desc_chars) for t in tools]
        for i, (t, ds) in enumerate(zip(tools, descs)):
            di = tk.encode(ds).ids[:DL]
            ni = tk.encode(t["name"]).ids[:NL]
            out["tool_desc_tok"][n, i, :len(di)] = torch.tensor(di, dtype=torch.int16)
            out["tool_name_tok"][n, i, :len(ni)] = torch.tensor(ni, dtype=torch.int16)
            out["called"][n, i] = t["sym"] in calls
        out["n_tool"][n] = len(tools)
        texts.append((QUERY_PREFIX + req, descs, [teacher_name(t["name"]) for t in tools]))
    print(f"  open pairs: {N} rows with a call, up to {MT} tools each", flush=True)
    return out


def open_losses(head, b, pad):
    """In-batch contrastive for open rows: every tool of every row in the
    batch is a candidate for each request, its own calls the positives. Most
    open rows declare one tool, so the batch supplies the negatives."""
    B = b["req_tok"].size(0)
    M = int(b["n_tool"].max())
    req = b["req_tok"].long()
    q_d, q_n = head.desc(req, query=True), head.name(req, query=True)
    d = encode_unique(head.desc, b["tool_desc_tok"][:, :M].long())
    n = encode_unique(head.name, b["tool_name_tok"][:, :M].long())
    live = torch.arange(M)[None] < b["n_tool"].long()[:, None]
    flat_live = live.reshape(-1)
    D = F.normalize(d.reshape(B * M, -1)[flat_live], dim=-1)
    Nv = F.normalize(n.reshape(B * M, -1)[flat_live], dim=-1)
    pos = torch.zeros(B, B * M, dtype=torch.bool)
    for i in range(B):
        pos[i, i * M:(i + 1) * M] = b["called"][i, :M]
    pos = pos[:, flat_live]
    s = head.scale.exp().clamp(max=100)
    allv = torch.ones_like(pos)
    return (multi_positive_nce(s * F.normalize(q_d, dim=-1) @ D.T, pos, allv)
            + multi_positive_nce(s * F.normalize(q_n, dim=-1) @ Nv.T, pos, allv))


# ---------------------------------------------------------------- the benchmark
@torch.no_grad()
def twin_eval(head, part, pad, n_kw, max_tool, teacher=None, bs=64) -> dict:
    """The twin benchmark on a cached split: per reference call whose tool
    has signature twins, the candidate with the highest score among the
    twins. Also the teacher's picks on the same decisions."""
    head.eval()
    keys = ("comb", "desc", "name", "t_desc", "t_name")
    tally = {k: [0, 0] for k in keys + ("comb_opaque", "comb_named", "desc_opaque")}
    # the same picks on flip-slot decisions only: the population the audit
    # passes, and the one a grounding number is quoted on
    tally.update({f"flip_{k}": [0, 0] for k in keys})
    chance = chance_flip = 0.0
    gates = {"opaque": [], "named": []}
    vecs = []
    N = part["tgt"].size(0)
    for i in range(0, N, bs):
        b = {k: v[i:i + bs] for k, v in part.items()}
        comb, sd, sn, g, enc, M = forward(head, b, pad)
        pos = called_tools(b["tgt"], n_kw, max_tool)[:, :M]
        grp = b["sig_group"][:, :M].long()
        live = torch.arange(M)[None] < b["n_tool"].long()[:, None]
        vecs.append(enc["desc"][live][:64])
        for r in range(comb.size(0)):
            opaque = bool(b["opaque"][r])
            for t in pos[r].nonzero().flatten().tolist():
                members = ((grp[r] == grp[r, t]) & live[r]).nonzero().flatten()
                if len(members) < 2:
                    continue
                chance += 1 / len(members)
                flip = "flip_tool" in b and bool(b["flip_tool"][r, t])
                chance_flip += (1 / len(members)) if flip else 0.0
                picks = {"comb": comb[r, members], "desc": sd[r, members],
                         "name": sn[r, members]}
                if teacher is not None and "t_desc" in b:
                    q = teacher[int(b["t_req"][r])]
                    picks["t_desc"] = teacher[b["t_desc"][r, members].long()] @ q
                    picks["t_name"] = teacher[b["t_name"][r, members].long()] @ q
                for k, sc in picks.items():
                    hit = int(members[int(sc.argmax())]) == t
                    tally[k][0] += hit
                    tally[k][1] += 1
                    if flip:
                        tally[f"flip_{k}"][0] += hit
                        tally[f"flip_{k}"][1] += 1
                    if k == "comb":
                        tally["comb_opaque" if opaque else "comb_named"][0] += hit
                        tally["comb_opaque" if opaque else "comb_named"][1] += 1
                    if k == "desc" and opaque:
                        tally["desc_opaque"][0] += hit
                        tally["desc_opaque"][1] += 1
                if g is not None:
                    gates["opaque" if opaque else "named"].append(float(g[r, members].mean()))
    head.train()
    n = tally["comb"][1]
    res = {f"acc_{k}": (v[0] / v[1] if v[1] else None) for k, v in tally.items()}
    res["n_decisions"] = n
    res["chance"] = chance / max(n, 1)
    res["n_flip"] = tally["flip_comb"][1]
    res["chance_flip"] = chance_flip / max(res["n_flip"], 1)
    res["gate_desc_opaque"] = sum(gates["opaque"]) / len(gates["opaque"]) if gates["opaque"] else None
    res["gate_desc_named"] = sum(gates["named"]) / len(gates["named"]) if gates["named"] else None
    V = torch.cat(vecs)[:5000]
    res["participation_ratio_desc"] = participation_ratio(V.float())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--desc-w", type=int, default=256)
    ap.add_argument("--desc-layers", type=int, default=4)
    ap.add_argument("--name-w", type=int, default=128)
    ap.add_argument("--name-layers", type=int, default=2)
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam-rel", type=float, default=1.0)
    ap.add_argument("--flip-weight", type=float, default=0.0,
                    help="weight of a twin loss on flip-slot decisions alone")
    ap.add_argument("--open-pairs", action="store_true",
                    help="add data/open_pairs as extra request/tool text, one "
                         "open batch every --open-every corpus batches")
    ap.add_argument("--open-every", type=int, default=4)
    ap.add_argument("--drop-unreadable", action="store_true",
                    help="leave training flip-slot calls whose request never "
                         "names the tool (S6's L9 'ambiguous' rows, 14%%) out "
                         "of every contrastive term; the exam already skips them")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--init-emb", choices=["random", "teacher"], default="random",
                    help="initialise both stages' token tables from the "
                         "teacher's (the plan's optional step-3 arm)")
    ap.add_argument("--teacher-model", default="unsloth/bge-small-en-v1.5")
    ap.add_argument("--weights", choices=["fp", "int8", "u4"], default="fp",
                    help="train the stages in this format from the start "
                         "(step 5, when post-training rounding costs more "
                         "than a point; stage_quant.py measures that)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cache = Path(args.cache)
    meta = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    if not meta.get("split"):
        raise SystemExit("needs a cache built with prep.py --split")
    lay = meta["layout"]
    n_kw, max_tool, pad = lay["n_kw"], lay["max_tool"], meta["in_pad"]
    cfg = {"desc_w": args.desc_w, "desc_layers": args.desc_layers,
           "name_w": args.name_w, "name_layers": args.name_layers,
           "pool": args.pool, "max_desc": meta["max_desc"],
           "max_name": meta["max_name"], "max_req": meta["max_req"]}
    head = build_head(cfg, meta["in_vocab"], pad)
    n_params = sum(p.numel() for p in head.parameters())
    if args.init_emb == "teacher":
        from tokenizers import Tokenizer
        tk0 = Tokenizer.from_file(str(cache / "in_tok.json"))
        for stage in (head.desc, head.name):
            P, ok = teacher_token_init(tk0, args.teacher_model, stage.width)
            with torch.no_grad():
                stage.emb.weight[ok] = P[ok]
        print(f"  token tables initialised from {args.teacher_model}: "
              f"{int(ok.sum())} of {len(ok)} pieces", flush=True)
        cfg["init_emb"] = "teacher"
    if args.weights != "fp":
        from stage_quant import quantize_head
        quantize_head(head, args.weights)
        cfg["weights"] = args.weights
    teacher = None
    if (cache / "teacher.pt").exists():
        teacher = torch.load(cache / "teacher.pt")["table"].float()

    train = load_part(cache, "train", args.limit_train)
    if args.drop_unreadable:
        keep = list(range(train["tgt"].size(0)))
        flip = train["flip_tool"] & called_tools(train["tgt"], n_kw, max_tool)[:, :max_tool]
        # called ones only: an uncalled decoy never fits the request, and
        # masking it would take the twin decision itself out of the loss
        train["unread"] = flip & ~readable(cache, "train", keep, flip.shape[1])
        print(f"  unreadable flip-slot calls dropped from training: "
              f"{int(train['unread'].sum())} of {int(flip.sum())}", flush=True)
    exam = load_part(cache, "holdout", args.limit_eval, ids=("+decoy", "+flip"))
    opn = None
    if args.open_pairs:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(cache / "in_tok.json"))
        tk.no_truncation(); tk.no_padding()
        opn = open_part(tk, cfg, meta.get("desc_chars", 60), pad)

    N = train["tgt"].size(0)
    steps = int(math.ceil(N / args.batch) * args.epochs)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm)))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    print(f"stages: desc {args.desc_w}x{args.desc_layers}, name {args.name_w}x"
          f"{args.name_layers}, pool {args.pool}: {n_params/1e6:.2f}M params; "
          f"{N} train rows, {steps} steps; open pairs {'on' if opn else 'off'}",
          flush=True)

    def evaluate(step):
        res = twin_eval(head, exam, pad, n_kw, max_tool, teacher)
        rec = {"step": step, **res, "secs": time.time() - t0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        f = lambda k: f"{res[k]:.1%}" if res.get(k) is not None else "-"  # noqa: E731
        print(f"  == step {step}: twins {f('acc_comb')} (desc {f('acc_desc')}, name "
              f"{f('acc_name')}; opaque-name rows {f('acc_comb_opaque')}) "
              f"teacher desc {f('acc_t_desc')} name {f('acc_t_name')}  chance "
              f"{res['chance']:.1%}  n={res['n_decisions']}  PR {res['participation_ratio_desc']:.1f}  "
              f"gate->desc opaque {res['gate_desc_opaque']} named {res['gate_desc_named']}",
              flush=True)
        print(f"     flip slots: twins {f('acc_flip_comb')} (desc {f('acc_flip_desc')}, name "
              f"{f('acc_flip_name')}) teacher desc {f('acc_flip_t_desc')} name "
              f"{f('acc_flip_t_name')}  chance {res['chance_flip']:.1%}  n={res['n_flip']}",
              flush=True)
        return res

    t0 = time.time()
    step = 0
    best = -1.0
    order = torch.randperm(N)
    cursor = 0
    ocursor = 0
    run = {}
    while step < steps:
        if cursor + args.batch > N:
            order, cursor = torch.randperm(N), 0
        idx = order[cursor:cursor + args.batch]
        cursor += args.batch
        b = {k: v[idx] for k, v in train.items()}
        loss, parts = batch_losses(head, b, pad, n_kw, max_tool, teacher, args.lam_rel,
                                   args.flip_weight)
        if opn is not None and step % args.open_every == 0:
            on = opn["n_tool"].size(0)
            oidx = torch.arange(ocursor, ocursor + args.batch) % on
            ocursor = (ocursor + args.batch) % on
            ob = {k: v[oidx] for k, v in opn.items()}
            ol = open_losses(head, ob, pad)
            loss = loss + ol
            parts["open"] = ol
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        for k, v in parts.items():
            v = float(v.detach()) if torch.is_tensor(v) else float(v)
            run[k] = 0.9 * run.get(k, v) + 0.1 * v
        if step % 20 == 0:
            el = time.time() - t0
            print(f"  step {step}/{steps} " + " ".join(f"{k} {v:.3f}" for k, v in run.items())
                  + f"  {el/step:.2f}s/step", flush=True)
        if step % args.eval_every == 0 or step == steps:
            res = evaluate(step)
            score = res.get("acc_flip_comb") or res["acc_comb"] or 0
            if score > best:
                best = score
                torch.save({"cfg": cfg, "head": head.state_dict(), "step": step,
                            "eval": res, "n_params": n_params}, out / "stages.pt")
    torch.save({"cfg": cfg, "head": head.state_dict(), "step": step,
                "n_params": n_params}, out / "stages_last.pt")
    print(f"done; best twin accuracy {best:.1%} -> {out / 'stages.pt'}", flush=True)


if __name__ == "__main__":
    main()
