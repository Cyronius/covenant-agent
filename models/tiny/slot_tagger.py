"""Slots by tagging: ELECTRA marks which words fill each role
(.claude/plans/electra-slot-reader.md, step 2, as changed 2026-10-01).

The regressing slot head (slot_reader.py) learned roles but approximated its
target vectors too loosely to match constants (probe 57.9 plain). Reading
the tagged words as a short text of their own does match them, so here the
head only tags: per role, a start word and an end word, or "absent" (the
[CLS] position), as an extractive question-answering head does. Each slot's
vector is the reader's vector of the tagged words, pooled out of one pass
over the text (electra_reader.py, R27); the whole slot is the whole text's.
The planner gets 8 vectors per text.

  python slot_tagger.py train --body electra --out reader/slots/t_electra [--limit 120000]
  python slot_tagger.py check --tagger reader/slots/t_electra/tagger.pt [--body tern-electra:...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from slot_reader import ELECTRA, ROLES, ROOT, load_roles, make_body, roles_of, span_texts, swap_pairs, tofrom_pairs

TAGGED = ROLES[1:]          # every role but the whole text
MAX_SPAN = 16               # word pieces


class SpanTagger(nn.Module):
    """A learned mix of the body's layers, one self-attention layer for
    span-level context, then per role a start and an end score for every
    word piece. Position 0 ([CLS]) means "absent"."""

    def __init__(self, n_states: int, d: int = 256, heads: int = 4):
        super().__init__()
        self.cfg = dict(n_states=n_states, d=d, heads=heads)
        self.mix = nn.Parameter(torch.zeros(n_states))
        self.ln = nn.LayerNorm(d)
        self.att = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, 2 * len(TAGGED))

    def forward(self, states: torch.Tensor, mask: torch.Tensor):
        """-> start, end logits (B, R, T); padding at -inf."""
        h = self.ln((F.softmax(self.mix, 0)[:, None, None, None] * states).sum(0))
        h = self.att(h, src_key_padding_mask=~mask)
        lg = self.out(h).view(h.size(0), h.size(1), 2, len(TAGGED)).permute(2, 0, 3, 1)
        lg = lg.masked_fill(~mask[None, :, None, :], float("-inf"))
        return lg[0], lg[1]


class Tok:
    """ELECTRA's word pieces with character offsets (the same pieces every
    body here reads: bert-base-uncased)."""

    def __init__(self):
        from tern_electra import _hf
        self.tk = _hf()[1].from_pretrained(ELECTRA)

    def __call__(self, texts: list[str]):
        enc = self.tk(texts, padding=True, truncation=True, max_length=128,
                      return_offsets_mapping=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"].bool(), enc["offset_mapping"]


def labels(rows: list[dict], offsets: torch.Tensor) -> torch.Tensor:
    """(B, R, 2) start and end piece of each role's span; 0 when absent or
    when the span falls past the 128-piece cut."""
    out = torch.zeros(len(rows), len(TAGGED), 2, dtype=torch.long)
    for b, r in enumerate(rows):
        off = offsets[b].tolist()
        real = [i for i, (a, z) in enumerate(off) if z > a]
        for k, role in enumerate(TAGGED):
            if role not in r["roles"]:
                continue
            c0, c1 = r["roles"][role]
            inside = [i for i in real if off[i][0] >= c0 and off[i][1] <= c1]
            if inside:
                out[b, k, 0], out[b, k, 1] = inside[0], inside[-1]
    return out


def decode(start: torch.Tensor, end: torch.Tensor, offsets: torch.Tensor):
    """Per text and role: (char start, char end) of the best span, or None
    when [CLS] (absent) scores higher."""
    B, R, T = start.shape
    s, e = start.float(), end.float()
    pair = s.unsqueeze(-1) + e.unsqueeze(-2)                       # (B, R, T, T): start i, end j
    ok = torch.ones(T, T, dtype=torch.bool).triu() & ~torch.ones(T, T, dtype=torch.bool).triu(MAX_SPAN)
    ok[0, :] = False
    ok[:, 0] = False
    word = offsets[:, :, 1] > offsets[:, :, 0]                      # not [CLS] / [SEP] / padding
    ok = ok[None, None] & word[:, None, :, None] & word[:, None, None, :]
    pair = pair.masked_fill(~ok, float("-inf"))
    best, idx = pair.view(B, R, -1).max(-1)
    null = s[:, :, 0] + e[:, :, 0]
    out = []
    for b in range(B):
        row = []
        for k in range(R):
            if not torch.isfinite(best[b, k]) or best[b, k] <= null[b, k]:
                row.append(None)
                continue
            i, j = divmod(int(idx[b, k]), T)
            row.append((int(offsets[b, i, 0]), int(offsets[b, j, 1])))
        out.append(row)
    return out


def cmd_train(args) -> int:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = load_roles("train")[:args.limit]
    held = load_roles("held")[:1000]
    body = make_body(args.body, None)
    tok = Tok()
    tagger = SpanTagger(body.n_states, body.d)
    opt = torch.optim.AdamW(tagger.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.batch)
    warm = max(1, steps // 30)
    log = (out / "log.jsonl").open("w", encoding="utf-8")
    step, t0, acc = 0, time.time(), defaultdict(float)
    for ep in range(args.epochs):
        rng.shuffle(train)
        for b in range(0, len(train) - args.batch + 1, args.batch):
            rows = train[b:b + args.batch]
            lr = args.lr * (step / warm if step < warm else
                            0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
            for g in opt.param_groups:
                g["lr"] = lr
            texts = [r["text"] for r in rows]
            ids, mask, offsets = tok(texts)
            states, _ = body.states(texts)
            st, en = tagger(states, mask)
            y = labels(rows, offsets)
            loss = (F.cross_entropy(st.reshape(-1, st.size(-1)), y[..., 0].reshape(-1))
                    + F.cross_entropy(en.reshape(-1, en.size(-1)), y[..., 1].reshape(-1))) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
            opt.step()
            step += 1
            acc["loss"] += float(loss.detach())
            if step % args.log_every == 0 or step == steps:
                row = {"step": step, "loss": round(acc.pop("loss") / args.log_every, 4),
                       "sec": round(time.time() - t0)}
                if step % args.eval_every == 0 or step == steps:
                    row.update(span_scores(tagger, body, tok, held))
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
    torch.save({"cfg": tagger.cfg, "state": tagger.state_dict(), "body": args.body,
                "mix": F.softmax(tagger.mix, 0).tolist()}, out / "tagger.pt")
    print(f"layer mix: {[round(x, 2) for x in F.softmax(tagger.mix, 0).tolist()]}")
    print(f"wrote {out / 'tagger.pt'}")
    return 0


@torch.no_grad()
def tag(tagger, body, tok, texts: list[str], bs: int = 256) -> list[list]:
    """Per text, per role: (char start, char end) or None."""
    tagger.eval()
    out = []
    for s in range(0, len(texts), bs):
        chunk = texts[s:s + bs]
        _, mask, offsets = tok(chunk)
        states, _ = body.states(chunk)
        st, en = tagger(states, mask)
        out += decode(st, en, offsets)
    tagger.train()
    return out


def _overlap(a, b) -> float:
    """Character overlap of two spans as a share of their union."""
    if a is None or b is None:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1, max(a[1], b[1]) - min(a[0], b[0]))


def span_scores(tagger, body, tok, rows: list[dict]) -> dict:
    """Against the parser's spans: presence accuracy, and for roles present in
    both, exact span match and mean overlap."""
    pred = tag(tagger, body, tok, [r["text"] for r in rows])
    pres_ok = pres_n = exact = both = 0
    ov = 0.0
    for r, p in zip(rows, pred):
        for k, role in enumerate(TAGGED):
            gold = tuple(r["roles"][role]) if role in r["roles"] else None
            pres_ok += int((gold is None) == (p[k] is None))
            pres_n += 1
            if gold is not None and p[k] is not None:
                both += 1
                exact += int(tuple(p[k]) == gold)
                ov += _overlap(p[k], gold)
    return {"present_acc": round(100 * pres_ok / pres_n, 1), "exact": round(100 * exact / max(1, both), 1),
            "overlap": round(ov / max(1, both), 3), "n_both": both}


def cmd_check(args) -> int:
    """Tagging against the parser, swaps and to/from flips, and our requests.
    The constant probe over the tagged spans is electra_reader.py check."""
    ck = torch.load(args.tagger)
    body = make_body(args.body or ck["body"], None)
    tagger = SpanTagger(**ck["cfg"])
    tagger.load_state_dict(ck["state"])
    tok = Tok()
    held = load_roles("held")
    D, S, O = TAGGED.index("destination"), TAGGED.index("source"), TAGGED.index("object")
    rep = {"tagger": args.tagger, "body": body.name, "mix": [round(x, 3) for x in ck["mix"]]}

    rep["held"] = span_scores(tagger, body, tok, held)
    print(f"{body.name}: held-out spans vs the parser: presence {rep['held']['present_acc']}%, "
          f"exact {rep['held']['exact']}%, overlap {rep['held']['overlap']} (n={rep['held']['n_both']})")

    def hits(spans, want, text):
        return spans is not None and text[spans[0]:spans[1]].strip().lower() == want.strip().lower()

    def where(text, words):
        i = text.find(words)
        return (i, i + len(words)) if i >= 0 else None

    def points(spans, text, right, wrong):
        r, w = where(text, right), where(text, wrong)
        return spans is not None and r is not None and _overlap(spans, r) > (_overlap(spans, w) if w else 0.0)

    for a_role, b_role in (("object", "destination"), ("object", "source")):
        pairs = swap_pairs(held, a_role, b_role)
        k = TAGGED.index(b_role)
        p0 = tag(tagger, body, tok, [p[0] for p in pairs])
        p1 = tag(tagger, body, tok, [p[1] for p in pairs])
        before = [hits(x[k], p[3], p[0]) for x, p in zip(p0, pairs)]
        after = [hits(x[k], p[2], p[1]) for x, p in zip(p1, pairs)]
        # looser, as step 0 measured it: the slot covers more of the right
        # words than of the wrong ones
        near0 = [points(x[k], p[0], p[3], p[2]) for x, p in zip(p0, pairs)]
        near1 = [points(x[k], p[1], p[2], p[3]) for x, p in zip(p1, pairs)]
        rep[f"swap_{b_role}"] = {"n": len(pairs), "before": round(100 * sum(before) / len(pairs), 1),
                                 "after": round(100 * sum(after) / len(pairs), 1),
                                 "both": round(100 * sum(a and b for a, b in zip(before, after)) / len(pairs), 1),
                                 "points_both": round(100 * sum(a and b for a, b in zip(near0, near1)) / len(pairs), 1)}
        print(f"swap {a_role}<->{b_role} (n={len(pairs)}): the {b_role} span is exactly the right words "
              f"{rep[f'swap_{b_role}']['before']}% before, {rep[f'swap_{b_role}']['after']}% after, "
              f"both {rep[f'swap_{b_role}']['both']}%; points at the right words both times "
              f"{rep[f'swap_{b_role}']['points_both']}%")

    pairs = tofrom_pairs(held)
    p0 = tag(tagger, body, tok, [p[0] for p in pairs])
    p1 = tag(tagger, body, tok, [p[1] for p in pairs])
    before = [hits(x[D], p[2], p[0]) for x, p in zip(p0, pairs)]
    after = [hits(x[S], p[2], p[1]) and x[D] is None for x, p in zip(p1, pairs)]
    near0 = [x[D] is not None and _overlap(x[D], where(p[0], p[2]) or (0, 0)) > 0.5 for x, p in zip(p0, pairs)]
    near1 = [x[S] is not None and _overlap(x[S], where(p[1], p[2]) or (0, 0)) > 0.5 for x, p in zip(p1, pairs)]
    rep["to_from"] = {"n": len(pairs), "dest_before": round(100 * sum(before) / len(pairs), 1),
                      "source_after_dest_gone": round(100 * sum(after) / len(pairs), 1),
                      "both": round(100 * sum(a and b for a, b in zip(before, after)) / len(pairs), 1),
                      "points_both": round(100 * sum(a and b for a, b in zip(near0, near1)) / len(pairs), 1)}
    print(f"to->from (n={len(pairs)}): destination exact before {rep['to_from']['dest_before']}%, "
          f"source exact and no destination after {rep['to_from']['source_after_dest_gone']}%, "
          f"both {rep['to_from']['both']}%; destination then source mostly on the right words "
          f"{rep['to_from']['points_both']}%")

    hand = ["move it to bob", "move it from bob", "assign card 6 to Bob", "assign Bob to card 6",
            "copy the report from the archive to the inbox", "copy the report from the inbox to the archive",
            "send priya the list", "delete the cards that are not done", "list orders placed before March"]
    rep["hand"] = {}
    for t, p in zip(hand, tag(tagger, body, tok, hand)):
        got = {role: t[s[0]:s[1]] for role, s in zip(TAGGED, p) if s is not None}
        rep["hand"][t] = got
        print(f"  {t!r}: {got}")

    import spacy
    nlp = spacy.load("en_core_web_sm")
    reqs = []
    for name in ("s6_holdout_s500.jsonl", "clt_holdout_s500.jsonl"):
        for line in (ROOT / "data" / name).open(encoding="utf-8"):
            reqs.append(json.loads(line)["request"])
    reqs = list(dict.fromkeys(reqs))[:600]
    rrows = [{"text": t, "roles": roles_of(doc)} for t, doc in zip(reqs, nlp.pipe(reqs))]
    rep["requests"] = span_scores(tagger, body, tok, rrows)
    print(f"our requests (never trained on) vs the parser: presence {rep['requests']['present_acc']}%, "
          f"exact {rep['requests']['exact']}%, overlap {rep['requests']['overlap']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--body", required=True, help="electra | tern-electra:<model.pt>")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("check")
    p.add_argument("--tagger", required=True)
    p.add_argument("--body", default=None)
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    return {"train": cmd_train, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
