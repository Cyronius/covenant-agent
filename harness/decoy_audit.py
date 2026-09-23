"""Can a decoyed exam be passed without reading the request?

The audit that found the problem (results/R10.md section 8), promoted to a
gate (.claude/plans/description-reading.md step 1). A decoy exam is honest
only if nothing about a tool *by itself* says whether it is the real one or a
sibling: not its description's voice, not its name, not where its text sits
in a pretrained encoder's space. So a classifier is given exactly that --
one tool's text, no request -- and asked "real or decoy?". If it can answer,
a planner can too, and a decoyed score measures that instead of reading.

  python -m harness.decoy_audit data/s5_holdout_decoy.jsonl \
      [--train-corpus data/s5_crowded.jsonl] [--tokenizer in_tok.json] \
      [--teacher unsloth/bge-small-en-v1.5] [--splits 5]

Population: every tool in a signature group that holds a decoy (the only
tools a twin decision is ever made between), deduplicated per world. Rows
tagged `opaque-names` are left out of the name audit -- both sides of it are
nonsense there by design.

Protocol: worlds split in half, fit on one half, AUC on the other, averaged
over `--splits` random halvings. With `--train-corpus`, also fit on that
corpus's worlds and test on this one's: the threat a planner trained there
actually poses.

Gates (all must hold):
  description bag-of-words  AUC <= 0.60
  name bag-of-words         AUC <= 0.60
  teacher embeddings        AUC <= 0.60   (skipped without the model)
  one-token word share      decoys within 2 points of real descriptions
                            (needs --tokenizer; a vocabulary fitted on real
                            training text fragments unfamiliar words, and
                            that is a style signal too)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GATE_AUC = 0.60
GATE_ONE_TOKEN = 0.02


# ---------------------------------------------------------------- population
def _sig(tool: dict) -> tuple:
    return (tuple((p["sym"], p["type"], p.get("required", True))
                  for p in tool["params"]),
            tool.get("returns"), tuple(sorted(tool["effects"])))


def collect(path: str, limit: int = 0, flip_only: bool = False) -> List[dict]:
    """[{world, name, desc, decoy, opaque}] -- every tool in a signature
    group that holds a decoy, once per (world, name, desc). `flip_only`
    keeps the groups of the flip slots (`provenance.flip_slot_tools`): the
    decisions where every sibling can be the working tool
    (`data.gen --twin-roles`), which are the only ones a grounding claim
    rests on (results/R11.md)."""
    seen, items = set(), []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            r = json.loads(line)
            decoys = set((r.get("provenance") or {}).get("decoys") or [])
            if not decoys:
                continue
            opaque = "opaque-names" in (r.get("tags") or [])
            groups = defaultdict(list)
            for t in r["context"]["tools"]:
                groups[_sig(t)].append(t)
            flip = set((r.get("provenance") or {}).get("flip_slot_tools") or [])
            for grp in groups.values():
                names = {t["name"] for t in grp}
                if not (names & decoys) or not (names - decoys):
                    continue
                if flip_only and not (names & flip):
                    continue
                for t in grp:
                    key = (r["world"], t["name"], t["desc"], opaque)
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append({"world": r["world"], "name": t["name"],
                                  "desc": t["desc"], "opaque": opaque,
                                  "decoy": t["name"] in decoys})
    return items


# ---------------------------------------------------------------- features
_WORD = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def desc_feats(s: str) -> List[str]:
    """A deliberately strong style reader: case-kept words, lowercased
    bigrams, punctuation, and length and shape tokens."""
    toks = _WORD.findall(s)
    low = [t.lower() for t in toks]
    f = [f"w:{t}" for t in toks]
    f += [f"b:{a}_{b}" for a, b in zip(low, low[1:])]
    n = len([t for t in toks if t.isalpha()])
    f.append(f"len:{min(n // 3, 10)}")
    f.append(f"first:{'U' if s[:1].isupper() else 'l'}")
    f.append(f"end:{s.rstrip()[-1:] or '-'}")
    return f


def _name_parts(name: str) -> List[str]:
    return [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+",
                                          name)]


def name_feats(name: str) -> List[str]:
    parts = _name_parts(name)
    f = [f"p:{p}" for p in parts]
    f += [f"pp:{a}_{b}" for a, b in zip(parts, parts[1:])]
    f.append(f"np:{len(parts)}")
    f.append(f"style:{'snake' if '_' in name else 'camel' if re.search('[a-z][A-Z]', name) else 'flat'}")
    padded = f"^{name.lower()}$"
    f += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
    return f


def _matrix(docs: List[List[str]], vocab: Dict[str, int]) -> torch.Tensor:
    X = torch.zeros(len(docs), len(vocab))
    for i, d in enumerate(docs):
        for t in d:
            j = vocab.get(t)
            if j is not None:
                X[i, j] = 1.0
    return X


def _vocab(docs: List[List[str]], min_df: int = 2) -> Dict[str, int]:
    df = defaultdict(int)
    for d in docs:
        for t in set(d):
            df[t] += 1
    keep = sorted(t for t, n in df.items() if n >= min_df)
    return {t: i for i, t in enumerate(keep)}


# ---------------------------------------------------------------- classifier
def fit_predict(Xtr: torch.Tensor, ytr: torch.Tensor, Xte: torch.Tensor,
                l2: float = 1e-2) -> torch.Tensor:
    """L2 logistic regression by LBFGS; class-balanced loss."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = ytr.mean().clamp(1e-3, 1 - 1e-3)
    weight = torch.where(ytr > 0, 0.5 / pos, 0.5 / (1 - pos))
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xtr @ w + b
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            z, ytr, weight=weight) + l2 * (w * w).sum())
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return Xte @ w + b


def auc(scores: torch.Tensor, y: torch.Tensor) -> float:
    """Mann-Whitney, ties counted half."""
    s, y = scores.numpy(), y.numpy()
    pos, neg = s[y > 0], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _run(tr: List[dict], te: List[dict], feats, key: str) -> float:
    dtr = [feats(it[key]) for it in tr]
    vocab = _vocab(dtr)
    if not vocab:
        return float("nan")
    Xtr, Xte = _matrix(dtr, vocab), _matrix([feats(it[key]) for it in te], vocab)
    ytr = torch.tensor([float(it["decoy"]) for it in tr])
    yte = torch.tensor([float(it["decoy"]) for it in te])
    return auc(fit_predict(Xtr, ytr, Xte), yte)


def _run_dense(tr, te, emb) -> float:
    Xtr = torch.stack([emb[it["desc"]] for it in tr])
    Xte = torch.stack([emb[it["desc"]] for it in te])
    ytr = torch.tensor([float(it["decoy"]) for it in tr])
    yte = torch.tensor([float(it["decoy"]) for it in te])
    return auc(fit_predict(Xtr, ytr, Xte), yte)


def _halves(items: List[dict], seed: int) -> Tuple[List[dict], List[dict]]:
    worlds = sorted({it["world"] for it in items})
    random.Random(seed).shuffle(worlds)
    a = set(worlds[: len(worlds) // 2])
    return ([it for it in items if it["world"] in a],
            [it for it in items if it["world"] not in a])


def teacher_embed(texts: List[str], model: str) -> Dict[str, torch.Tensor]:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    enc = AutoModel.from_pretrained(model).eval()
    out = {}
    uniq = sorted(set(texts))
    with torch.no_grad():
        for i in range(0, len(uniq), 64):
            b = tok(uniq[i:i + 64], padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
            v = torch.nn.functional.normalize(
                enc(**b).last_hidden_state[:, 0], dim=-1)
            for t, x in zip(uniq[i:i + 64], v):
                out[t] = x
    return out


def one_token_share(items: List[dict], tokenizer_path: str) -> Tuple[float, float]:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tokenizer_path)
    tk.no_truncation()
    tk.no_padding()
    tally = {True: [0, 0], False: [0, 0]}
    for it in items:
        for w in re.findall(r"[A-Za-z]+", it["desc"]):
            tally[it["decoy"]][0] += len(tk.encode(w).tokens) == 1
            tally[it["decoy"]][1] += 1
    share = lambda c: c[0] / max(c[1], 1)
    return share(tally[False]), share(tally[True])


# ---------------------------------------------------------------- report
def audit(corpus: str, train_corpus: str | None = None,
          tokenizer: str | None = None, teacher: str | None = None,
          splits: int = 5, limit: int = 0, flip_only: bool = False) -> dict:
    items = collect(corpus, limit, flip_only)
    named = [it for it in items if not it["opaque"]]
    n_dec = sum(it["decoy"] for it in items)
    res = {"corpus": corpus, "tools": len(items), "decoys": n_dec,
           "worlds": len({it["world"] for it in items})}
    emb = None
    if teacher:
        try:
            texts = [it["desc"] for it in items]
            if train_corpus:
                texts += [it["desc"] for it in collect(train_corpus, limit, flip_only)]
            emb = teacher_embed(texts, teacher)
        except Exception as e:  # noqa: BLE001 -- the model may be absent
            res["teacher_error"] = str(e)[:200]

    for label, run in (("desc", lambda a, b: _run(a, b, desc_feats, "desc")),
                       ("name", lambda a, b: _run(
                           [x for x in a if not x["opaque"]],
                           [x for x in b if not x["opaque"]],
                           name_feats, "name")),
                       ("teacher", (lambda a, b: _run_dense(a, b, emb))
                        if emb is not None else None)):
        if run is None:
            continue
        pool = named if label == "name" else items
        aucs = [run(*_halves(pool, s)) for s in range(splits)]
        aucs = [a for a in aucs if a == a]
        res[f"auc_{label}"] = sum(aucs) / len(aucs) if aucs else float("nan")
        res[f"auc_{label}_max"] = max(aucs) if aucs else float("nan")
    if train_corpus:
        tr = collect(train_corpus, limit, flip_only)
        overlap = {it["world"] for it in tr} & {it["world"] for it in items}
        res["transfer_world_overlap"] = len(overlap)
        res["transfer_auc_desc"] = _run(tr, items, desc_feats, "desc")
        res["transfer_auc_name"] = _run(
            [x for x in tr if not x["opaque"]], named, name_feats, "name")
        if emb is not None:
            res["transfer_auc_teacher"] = _run_dense(tr, items, emb)
    if tokenizer:
        real, dec = one_token_share(items, tokenizer)
        res["one_token_real"], res["one_token_decoy"] = real, dec
    gates = {}
    for k in ("auc_desc", "auc_name", "auc_teacher", "transfer_auc_desc",
              "transfer_auc_name", "transfer_auc_teacher"):
        if k in res and res[k] == res[k]:
            gates[k] = res[k] <= GATE_AUC
    if "one_token_real" in res:
        gates["one_token"] = abs(res["one_token_real"]
                                 - res["one_token_decoy"]) <= GATE_ONE_TOKEN
    res["gates"] = gates
    res["pass"] = bool(gates) and all(gates.values())
    return res


def slot_map(themes_dir: str) -> Dict[Tuple[str, str], str]:
    """(world, tool name) -> the theme slot it belongs to, real or decoy."""
    out = {}
    for path in sorted(Path(themes_dir).glob("*.json")):
        th = json.loads(path.read_text(encoding="utf-8"))
        w = th["domain"]
        specs = [(f"tools.{k}", v) for k, v in th["tools"].items()]
        specs += [(f"v2.{k}", v) for k, v in (th.get("v2") or {}).items()]
        for slot, v in specs:
            out[(w, v["name"])] = slot
            for d in v.get("decoys") or []:
                out[(w, d["name"])] = slot
    return out


def by_slot(items: List[dict], themes_dir: str, splits: int = 3) -> Dict[str, float]:
    """Description AUC per theme slot: fitted on half the worlds (all slots),
    scored on the other half's items of that slot only. Where to aim a
    revision pass."""
    sm = slot_map(themes_dir)
    scored = defaultdict(lambda: ([], []))
    for seed in range(splits):
        tr, te = _halves(items, seed)
        dtr = [desc_feats(it["desc"]) for it in tr]
        vocab = _vocab(dtr)
        Xtr, Xte = _matrix(dtr, vocab), _matrix([desc_feats(it["desc"]) for it in te], vocab)
        ytr = torch.tensor([float(it["decoy"]) for it in tr])
        sc = fit_predict(Xtr, ytr, Xte)
        for it, v in zip(te, sc.tolist()):
            slot = sm.get((it["world"], it["name"]), "other")
            scored[slot][0].append(v)
            scored[slot][1].append(float(it["decoy"]))
    return {slot: auc(torch.tensor(s), torch.tensor(y))
            for slot, (s, y) in sorted(scored.items())}


def top_features(corpus: str, k: int = 12, limit: int = 0,
                 flip_only: bool = False) -> List[Tuple[str, float]]:
    """The description features that most say "decoy", fitted on the whole
    corpus: what to rewrite when the gate fails."""
    items = collect(corpus, limit, flip_only)
    docs = [desc_feats(it["desc"]) for it in items]
    vocab = _vocab(docs, min_df=3)
    X = _matrix(docs, vocab)
    y = torch.tensor([float(it["decoy"]) for it in items])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            Xs @ w + b, y) + 1e-2 * (w * w).sum())
        loss.backward()
        return loss

    opt.step(closure)
    inv = {i: t for t, i in vocab.items()}
    order = torch.argsort(w.detach(), descending=True)
    return [(inv[int(i)], float(w.detach()[i])) for i in order[:k]]


def main() -> int:
    ap = argparse.ArgumentParser(prog="harness.decoy_audit")
    ap.add_argument("corpus")
    ap.add_argument("--train-corpus")
    ap.add_argument("--tokenizer")
    ap.add_argument("--teacher", default="unsloth/bge-small-en-v1.5",
                    help="'' to skip the embedding audit")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", help="also write the result here")
    ap.add_argument("--flip-only", action="store_true",
                    help="audit only the flip slots' twin groups: the "
                         "grounding population under --twin-roles")
    ap.add_argument("--themes", help="theme dir: also report description AUC "
                                     "per theme slot")
    args = ap.parse_args()
    res = audit(args.corpus, args.train_corpus, args.tokenizer,
                args.teacher or None, args.splits, args.limit, args.flip_only)
    print(f"{res['corpus']}: {res['tools']} tools in decoyed groups "
          f"({res['decoys']} decoys) over {res['worlds']} worlds")
    for k, v in res.items():
        if k.startswith(("auc_", "transfer_", "one_token")):
            print(f"  {k:26s} {v:.3f}" if isinstance(v, float) else f"  {k:26s} {v}")
    if "teacher_error" in res:
        print(f"  teacher skipped: {res['teacher_error']}")
    print("  gates:", ", ".join(f"{k} {'ok' if v else 'FAIL'}"
                                for k, v in res["gates"].items()))
    print("  strongest decoy features:",
          ", ".join(f"{t} ({w:+.2f})" for t, w in
                    top_features(args.corpus, limit=args.limit, flip_only=args.flip_only)))
    if args.themes:
        per = by_slot(collect(args.corpus, args.limit, args.flip_only), args.themes)
        res["auc_desc_by_slot"] = per
        print("  description AUC by slot:",
              ", ".join(f"{k} {v:.2f}" for k, v in per.items()))
    print("PASS" if res["pass"] else "FAIL")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
