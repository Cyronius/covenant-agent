"""A reader that hands the planner a few role vectors per text
(.claude/plans/electra-slot-reader.md, steps 0 and 2).

Slots: one 384-number vector per role, plus a "present" score, whatever the
text's length. A slot head (one learned query per role attending over a
body's word states) makes them. Targets come from general English only:
spaCy's parse of data/general marks each role's words, and the teacher
(all-MiniLM-L6-v2) embeds them.

  python slot_reader.py parse                 # -> reader/slots/roles.jsonl
  python slot_reader.py teacher               # -> reader/slots/teacher.pt
  python slot_reader.py train --body electra --out reader/slots/h_electra
  python slot_reader.py check --head reader/slots/h_electra/head.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import zlib
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
GENERAL = ROOT / "data" / "general" / "texts.jsonl"
SLOTS = HERE / "reader" / "slots"
ROLES = ("whole", "action", "object", "destination", "source", "condition", "time", "amount")
R = {r: i for i, r in enumerate(ROLES)}
MAX_WORDS = 30
ELECTRA = "google/electra-small-discriminator"

DEST = {"to", "into", "onto", "toward", "towards"}
SRC = {"from"}
WHEN = {"before", "after", "since", "until", "during"}
COND = {"with", "without", "where"}


# ── role spans from a parse ───────────────────────────────────────────────────

def _span(tokens) -> tuple[int, int]:
    tokens = list(tokens)
    return min(t.idx for t in tokens), max(t.idx + len(t.text) for t in tokens)


def _subtree_minus(tok, skip) -> list:
    """tok's subtree without the subtrees of its descendants in `skip`."""
    drop = {d.i for s in skip for d in s.subtree}
    return [t for t in tok.subtree if t.i not in drop]


def roles_of(doc) -> dict[str, tuple[int, int]]:
    """Each role's character span in the text, from spaCy's dependency parse.
    The whole text is always present; the rest only when the parse shows them."""
    out = {"whole": (0, len(doc.text))}
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return out
    # a verb, or a word the tagger called a noun but that takes an object
    # ("Copies the object ..."); a copula ("she was able") is not an action
    verb = root.pos_ == "VERB" or any(c.dep_ in ("dobj", "dative") for c in root.children)
    if verb:
        act = [root] + [c for c in root.children if c.dep_ in ("neg", "prt")]
        out["action"] = _span(act)
    obj = None
    for c in root.children:
        if c.dep_ in ("dobj", "nsubjpass") or (c.dep_ == "attr" and verb):
            obj = c
            break

    def role_preps(tok):
        return [c for c in tok.children if c.dep_ == "prep"
                and c.lower_ in DEST | SRC | WHEN | COND]

    def first_pobj(prep):
        return next((c for c in prep.children if c.dep_ == "pobj"), None)

    if obj is not None:
        # clauses on the object, however deep ("the number of records created
        # after ..."), are its condition, not part of it
        clauses = [t for t in obj.subtree if t.dep_ in ("relcl", "acl") and t is not obj]
        clauses = [c for c in clauses if not any(a in clauses for a in c.ancestors)]
        skip = clauses + role_preps(obj)
        out["object"] = _span(_subtree_minus(obj, skip))
        cond = clauses + [c for c in obj.children if c.dep_ == "prep" and c.lower_ in COND]
        if cond:
            out["condition"] = _span([t for c in cond for t in c.subtree])
        for c in obj.children:
            if c.dep_ == "nummod" and obj.tag_ in ("NNS", "NNPS"):
                out["amount"] = _span([c])
    heads = [root] + ([obj] if obj is not None else [])
    for h in heads:
        for p in role_preps(h):
            po = first_pobj(p)
            if po is None:
                continue
            if p.lower_ in DEST and "destination" not in out:
                out["destination"] = _span(po.subtree)
            elif p.lower_ in SRC and "source" not in out:
                out["source"] = _span(po.subtree)
            elif p.lower_ in WHEN and "time" not in out:
                out["time"] = _span(p.subtree)
            elif p.lower_ in COND and h is root and "condition" not in out:
                out["condition"] = _span(p.subtree)
        for c in h.children:
            if c.dep_ == "dative" and "destination" not in out:
                out["destination"] = _span(c.subtree)
    # "move it out of the folder": out/of as a source
    for t in doc:
        if t.lower_ == "out" and t.i + 1 < len(doc) and doc[t.i + 1].lower_ == "of" and "source" not in out:
            po = first_pobj(doc[t.i + 1])
            if po is not None:
                out["source"] = _span(po.subtree)
    for t in doc:
        if t.dep_ == "prep" and t.lower_ in WHEN and "time" not in out and first_pobj(t) is not None:
            out["time"] = _span(t.subtree)
    if "time" not in out:
        for e in doc.ents:
            if e.label_ in ("DATE", "TIME") and re.search(r"[A-Za-z]", e.text):
                out["time"] = (e.start_char, e.end_char)
                break
    return {r: s for r, s in out.items() if s[1] > s[0]}


def span_texts(text: str, roles: dict) -> dict[str, str]:
    return {r: text[a:b] for r, (a, b) in roles.items()}


def cmd_parse(args) -> int:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    rng = random.Random(0)
    texts = [json.loads(line)["text"] for line in GENERAL.open(encoding="utf-8")]
    texts = [t for t in texts if 3 <= len(t.split()) <= MAX_WORDS]
    held = [t for t in texts if held_out(t)]
    train = [t for t in texts if not held_out(t)]
    rng.shuffle(held)
    rng.shuffle(train)
    if args.cue:
        # a second pass, appended: texts with a direction or time word, which
        # random texts (mostly noun-phrase descriptions) rarely have
        done = {r["text"] for r in load_roles()}
        cue = re.compile(r"\b(to|from|into|onto|out of|before|after|since|until)\b", re.I)
        held = [t for t in held if t not in done and cue.search(t)]
        train = [t for t in train if t not in done and cue.search(t)]
    todo = [(t, "held") for t in held[:args.n_held]] + [(t, "train") for t in train[:args.n_train]]
    SLOTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    count = defaultdict(int)
    with (SLOTS / "roles.jsonl").open("a" if args.cue else "w", encoding="utf-8") as fh:
        for (text, split), doc in zip(todo, nlp.pipe((t for t, _ in todo), batch_size=512,
                                                     n_process=args.procs)):
            roles = roles_of(doc)
            for r in roles:
                count[(split, r)] += 1
            fh.write(json.dumps({"text": text, "split": split, "roles": roles}) + "\n")
    print(f"{len(todo)} texts parsed in {time.time() - t0:.0f}s -> {SLOTS / 'roles.jsonl'}")
    for split in ("train", "held"):
        n = sum(1 for _, s in todo if s == split)
        print(f"  {split} ({n}): " + "  ".join(f"{r} {100 * count[(split, r)] / n:.0f}%" for r in ROLES))
    return 0


def load_roles(split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in (SLOTS / "roles.jsonl").open(encoding="utf-8")]
    return [r for r in rows if split is None or r["split"] == split]


def held_out(text: str) -> bool:
    """The general corpus's held-out tenth (crc32), the same split since R25."""
    return zlib.crc32(text.encode("utf-8")) % 10 == 0


def teacher():
    """all-MiniLM-L6-v2, whose vectors the slots copy (electra_reader.MiniLM)."""
    from electra_reader import MiniLM      # electra_reader imports this module
    return MiniLM()


def cmd_teacher(args) -> int:
    """The teacher's vector for every text and span."""
    rows = load_roles()
    texts = list(dict.fromkeys(t for r in rows for t in span_texts(r["text"], r["roles"]).values()))
    t0 = time.time()
    table = teacher().vectors(texts, log_every=200)
    path = SLOTS / "teacher.pt"
    torch.save({"texts": texts, "table": table.half()}, path)
    print(f"{len(texts)} texts in {time.time() - t0:.0f}s -> {path}")
    return 0


def with_span_rows(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """rows plus up to n of their role spans as texts of their own (only the
    whole slot present): the reader also reads 1-4 word texts, constants
    like "Bob Alvarez" and fields like "user name"."""
    spans = list(dict.fromkeys(s for r in rows for role, s in span_texts(r["text"], r["roles"]).items()
                               if role != "whole"))
    rng.shuffle(spans)
    return rows + [{"text": s, "roles": {"whole": [0, len(s)]}} for s in spans[:n]]


# ── bodies ────────────────────────────────────────────────────────────────────

class ElectraBody:
    """Frozen ELECTRA-small: every layer's word states (embeddings + 12)."""

    def __init__(self, layers: int | None = None):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.tk = AutoTokenizer.from_pretrained(ELECTRA)
        self.m = AutoModel.from_pretrained(ELECTRA, output_hidden_states=True).eval()
        self.n_states = (layers if layers is not None else self.m.config.num_hidden_layers) + 1
        self.d = self.m.config.hidden_size
        self.name = f"electra-small[:{self.n_states - 1}]"

    @torch.no_grad()
    def states(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tk(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        hs = self.m(**enc).hidden_states[:self.n_states]
        return torch.stack(hs), enc["attention_mask"].bool()


class TernElectraBody:
    """Frozen ternary ELECTRA (tern_electra.py): the same list of states as
    ElectraBody, from the ternary student."""

    def __init__(self, path: str):
        from tern_electra import TernElectra, _hf
        self.m = TernElectra.load(Path(path)).eval()
        self.tk = _hf()[1].from_pretrained(ELECTRA)
        self.n_states, self.d = self.m.dims["layers"] + 1, self.m.dims["d"]
        self.name = f"tern-electra:{Path(path).parent.name}"

    @torch.no_grad()
    def states(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tk(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        mask = enc["attention_mask"].bool()
        st, _ = self.m(enc["input_ids"], mask)
        return torch.stack(st), mask


def make_body(name: str, layers: int | None = None):
    """electra | tern-electra:<path to tern_electra.py's model.pt>"""
    if name.startswith("tern-electra:"):
        return TernElectraBody(name.split(":", 1)[1])
    if name != "electra":
        raise SystemExit(f"unknown body {name!r}: electra | tern-electra:<model.pt>")
    return ElectraBody(layers)


# ── the slot head ─────────────────────────────────────────────────────────────

class SlotHead(nn.Module):
    """A learned mix of the body's layers, then one query per role attending
    over the words, a small feed-forward, and per slot a 384-number unit
    vector and a present score."""

    def __init__(self, n_states: int, d: int = 256, out: int = 384, heads: int = 4, rounds: int = 2):
        super().__init__()
        self.cfg = dict(n_states=n_states, d=d, out=out, heads=heads, rounds=rounds)
        self.mix = nn.Parameter(torch.zeros(n_states))
        self.ln = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(len(ROLES), d) * 0.02)
        self.att = nn.ModuleList(nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(rounds))
        self.ff = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                                              nn.Linear(4 * d, d)) for _ in range(rounds))
        self.lnq = nn.ModuleList(nn.LayerNorm(d) for _ in range(rounds))
        self.proj = nn.Linear(d, out)
        self.present = nn.Linear(d, 1)

    def forward(self, states: torch.Tensor, mask: torch.Tensor):
        """states (L, B, T, d), mask (B, T) -> slots (B, K, out) unit, present logits (B, K)."""
        h = self.ln((F.softmax(self.mix, 0)[:, None, None, None] * states).sum(0))
        q = self.q.unsqueeze(0).expand(h.size(0), -1, -1)
        for att, ff, lnq in zip(self.att, self.ff, self.lnq):
            q = q + att(lnq(q), h, h, key_padding_mask=~mask, need_weights=False)[0]
            q = q + ff(q)
        return F.normalize(self.proj(q), dim=-1), self.present(q).squeeze(-1)


def targets(rows: list[dict], index: dict, T: torch.Tensor):
    """(B, K, 384) teacher vectors (zero where absent) and (B, K) presence."""
    tv = torch.zeros(len(rows), len(ROLES), T.size(1))
    pres = torch.zeros(len(rows), len(ROLES))
    for b, r in enumerate(rows):
        for role, s in span_texts(r["text"], r["roles"]).items():
            tv[b, R[role]] = T[index[s]]
            pres[b, R[role]] = 1
    return tv, pres


def slot_loss(slots, logits, tv, pres, lam_present: float = 0.5):
    cos = (slots * tv).sum(-1)
    vec = ((1 - cos) * pres).sum() / pres.sum().clamp(min=1)
    bce = F.binary_cross_entropy_with_logits(logits[:, 1:], pres[:, 1:])
    return vec + lam_present * bce, vec, bce


def cmd_train(args) -> int:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tt = torch.load(SLOTS / "teacher.pt")
    index = {t: i for i, t in enumerate(tt["texts"])}
    T = tt["table"].float()
    train = load_roles("train")[:args.limit]
    if args.span_rows:
        train = with_span_rows(train, args.span_rows, rng)
    held = load_roles("held")[:1000]
    body = make_body(args.body, args.layers)
    head = SlotHead(body.n_states, body.d)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
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
            states, mask = body.states([r["text"] for r in rows])
            slots, logits = head(states, mask)
            tv, pres = targets(rows, index, T)
            loss, vec, bce = slot_loss(slots, logits, tv, pres)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            step += 1
            acc["loss"] += float(loss.detach()); acc["vec"] += float(vec.detach()); acc["bce"] += float(bce.detach())
            if step % args.log_every == 0 or step == steps:
                row = {"step": step, **{k: round(v / args.log_every, 4) for k, v in acc.items()},
                       "sec": round(time.time() - t0)}
                acc.clear()
                if step % args.eval_every == 0 or step == steps:
                    row.update(routing(head, body, held, index, T))
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
    torch.save({"cfg": head.cfg, "state": head.state_dict(), "body": args.body, "target": "teacher",
                "layers": args.layers, "mix": F.softmax(head.mix, 0).tolist()}, out / "head.pt")
    print(f"layer mix: {[round(x, 2) for x in F.softmax(head.mix, 0).tolist()]}")
    print(f"wrote {out / 'head.pt'}")
    return 0


# ── checks (step 0) ───────────────────────────────────────────────────────────

@torch.no_grad()
def read(head, body, texts: list[str], bs: int = 256):
    out_s, out_p = [], []
    head.eval()
    for s in range(0, len(texts), bs):
        st, mask = body.states(texts[s:s + bs])
        sl, lg = head(st, mask)
        out_s.append(sl)
        out_p.append(torch.sigmoid(lg))
    head.train()
    return torch.cat(out_s), torch.cat(out_p)


def routing(head, body, rows: list[dict], index: dict, T: torch.Tensor) -> dict:
    """Role routing: each present role's slot is closer to its own span's
    teacher vector than to the text's other spans'; presence accuracy; and
    each slot's cosine to its target."""
    slots, pres = read(head, body, [r["text"] for r in rows])
    hit = tot = 0
    cos, pres_ok, pres_n = defaultdict(list), 0, 0
    for b, r in enumerate(rows):
        sp = {role: T[index[s]] for role, s in span_texts(r["text"], r["roles"]).items() if role != "whole"}
        for role, v in sp.items():
            cos[role].append(float(slots[b, R[role]] @ v))
        for k, role in enumerate(ROLES[1:], 1):
            pres_ok += int((pres[b, k] > 0.5) == (role in sp))
            pres_n += 1
        if len(sp) < 2:
            continue
        for role, v in sp.items():
            mine = float(slots[b, R[role]] @ v)
            others = [float(slots[b, R[role]] @ w) for o, w in sp.items() if o != role]
            hit += int(all(mine > x for x in others))
            tot += 1
    return {"route": round(100 * hit / max(1, tot), 1), "route_n": tot,
            "present_acc": round(100 * pres_ok / max(1, pres_n), 1),
            "cos": {k: round(sum(v) / len(v), 3) for k, v in cos.items()}}


def swap_pairs(rows: list[dict], a_role: str, b_role: str) -> list[tuple[str, str, str, str]]:
    """Texts with both roles as separate spans, and the same text with the two
    spans' words exchanged: (text, swapped, span a, span b)."""
    out = []
    for r in rows:
        ro = r["roles"]
        if a_role in ro and b_role in ro:
            (a0, a1), (b0, b1) = sorted([tuple(ro[a_role]), tuple(ro[b_role])])
            if a1 <= b0:
                t = r["text"]
                sa, sb = t[a0:a1], t[b0:b1]
                if sa.lower() != sb.lower():
                    sw = t[:a0] + sb + t[a1:b0] + sa + t[b1:]
                    first = a_role if tuple(ro[a_role]) == (a0, a1) else b_role
                    span_a, span_b = (sa, sb) if first == a_role else (sb, sa)
                    out.append((t, sw, span_a, span_b))
    return out


def tofrom_pairs(rows: list[dict]) -> list[tuple[str, str, str]]:
    """Texts whose destination follows "to", and the same with "from": (text, flipped, the span)."""
    out = []
    for r in rows:
        if "destination" not in r["roles"]:
            continue
        a, b = r["roles"]["destination"]
        t = r["text"]
        m = re.search(r"\bto\s+$", t[:a])
        if m:
            out.append((t, t[:m.start()] + "from " + t[a:], t[a:b]))
    return out


def cmd_check(args) -> int:
    from chunk_probe import SUITES, rank_stats, tasks as probe_tasks
    from prep import reader_text
    ck = torch.load(args.head)
    body = make_body(args.body or ck["body"], ck["layers"])
    head = SlotHead(**ck["cfg"])
    head.load_state_dict(ck["state"])
    tt = torch.load(SLOTS / "teacher.pt")
    index = {t: i for i, t in enumerate(tt["texts"])}
    T = tt["table"].float()
    tch = teacher()
    held = load_roles("held")
    rep = {"head": args.head, "body": body.name, "target": "teacher", "mix": [round(x, 3) for x in ck["mix"]]}

    rep["held"] = routing(head, body, held, index, T)
    print(f"{body.name}: held-out role routing {rep['held']['route']}% (n={rep['held']['route_n']}), "
          f"presence {rep['held']['present_acc']}%, slot cosine {rep['held']['cos']}")

    # swaps: the destination (or source) slot follows its words
    for a_role, b_role in (("object", "destination"), ("object", "source")):
        pairs = swap_pairs(held, a_role, b_role)
        if not pairs:
            continue
        texts = [p[0] for p in pairs] + [p[1] for p in pairs]
        sl, _ = read(head, body, texts)
        ta, tb = tch.vectors([p[2] for p in pairs]), tch.vectors([p[3] for p in pairs])
        n = len(pairs)
        k = R[b_role]
        orig = ((sl[:n, k] * tb).sum(1) > (sl[:n, k] * ta).sum(1)).float().mean()
        swp = ((sl[n:, k] * ta).sum(1) > (sl[n:, k] * tb).sum(1)).float().mean()
        both = (((sl[:n, k] * tb).sum(1) > (sl[:n, k] * ta).sum(1))
                & ((sl[n:, k] * ta).sum(1) > (sl[n:, k] * tb).sum(1))).float().mean()
        rep[f"swap_{b_role}"] = {"n": n, "orig": round(100 * float(orig), 1),
                                 "swapped": round(100 * float(swp), 1), "both": round(100 * float(both), 1)}
        print(f"swap {a_role}<->{b_role} (n={n}): the {b_role} slot follows its words "
              f"{rep[f'swap_{b_role}']['orig']}% before, {rep[f'swap_{b_role}']['swapped']}% after, "
              f"both {rep[f'swap_{b_role}']['both']}%  (chance 25% for both)")

    # to -> from: the words move from the destination slot to the source slot
    pairs = tofrom_pairs(held)
    texts = [p[0] for p in pairs] + [p[1] for p in pairs]
    sl, pr = read(head, body, texts)
    tx = tch.vectors([p[2] for p in pairs])
    n = len(pairs)
    d, s = R["destination"], R["source"]
    before = ((sl[:n, d] * tx).sum(1) > (sl[:n, s] * tx).sum(1)).float()
    after = ((sl[n:, s] * tx).sum(1) > (sl[n:, d] * tx).sum(1)).float()
    rep["to_from"] = {"n": n, "dest_before": round(100 * float(before.mean()), 1),
                      "source_after": round(100 * float(after.mean()), 1),
                      "both": round(100 * float((before * after).mean()), 1),
                      "present_dest_before": round(100 * float((pr[:n, d] > 0.5).float().mean()), 1),
                      "present_source_after": round(100 * float((pr[n:, s] > 0.5).float().mean()), 1)}
    print(f"to->from (n={n}): words in the destination slot before {rep['to_from']['dest_before']}%, "
          f"in the source slot after {rep['to_from']['source_after']}%, both {rep['to_from']['both']}%; "
          f"present: destination before {rep['to_from']['present_dest_before']}%, "
          f"source after {rep['to_from']['present_source_after']}%")

    # hand pairs from C0 / R25
    hand = [("move it to bob", "move it from bob", "bob"),
            ("assign card 6 to Bob", "assign Bob to card 6", "Bob"),
            ("copy the report from the archive to the inbox", "copy the report from the inbox to the archive", "the inbox")]
    rep["hand"] = []
    for a, b, w in hand:
        sl, pr = read(head, body, [a, b])
        tw = tch.vectors([w])[0]
        row = {"text": a, "flip": b, "word": w,
               "dest": [round(float(sl[i, d] @ tw), 3) for i in (0, 1)],
               "source": [round(float(sl[i, s] @ tw), 3) for i in (0, 1)],
               "object": [round(float(sl[i, R["object"]] @ tw), 3) for i in (0, 1)]}
        rep["hand"].append(row)
        print(f"  {a!r} / {b!r}: {w!r} vs destination {row['dest']}, source {row['source']}, object {row['object']}")

    # our requests: parsed the same way, never trained on (information only)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    reqs = []
    for name in ("s6_holdout_s500.jsonl", "clt_holdout_s500.jsonl"):
        for line in (ROOT / "data" / name).open(encoding="utf-8"):
            reqs.append(json.loads(line)["request"])
    reqs = list(dict.fromkeys(reqs))[:600]
    rrows = [{"text": t, "roles": roles_of(doc)} for t, doc in zip(reqs, nlp.pipe(reqs))]
    spans = list(dict.fromkeys(s for r in rrows for s in span_texts(r["text"], r["roles"]).values()))
    tv = tch.vectors(spans)
    idx2 = {t: i for i, t in enumerate(spans)}
    rep["requests"] = routing(head, body, rrows, idx2, tv)
    print(f"our requests (parsed, never trained on): routing {rep['requests']['route']}% "
          f"(n={rep['requests']['route_n']}), presence {rep['requests']['present_acc']}%")

    # constant probe: a constant's whole slot against the request's slots
    rep["probe"] = {}
    for suite, path in SUITES.items():
        agg = defaultdict(lambda: [0.0, 0, 0.0, 0])
        for tid, request, consts, used in probe_tasks(path, dedupe=(suite == "demo")):
            syms = [x for x, _, _ in consts]
            if not (set(syms) & used) or not (set(syms) - used):
                continue
            ctexts = [reader_text(v) for _, _, v in consts]
            cs, _ = read(head, body, ctexts)
            rs, rp = read(head, body, [request])
            cw = cs[:, 0]
            present = [0] + [k for k in range(1, len(ROLES)) if rp[0, k] > 0.5]
            methods = {"whole": (cw @ rs[0, 0]).tolist(),
                       "slots": (cw @ rs[0, present].T).max(1).values.tolist()}
            for mname, sc in methods.items():
                w, p, t1 = rank_stats(sc, used, syms)
                a = agg[mname]
                a[0] += w; a[1] += p; a[2] += t1; a[3] += 1
        rep["probe"][suite] = {m: round(100 * a[0] / a[1], 1) for m, a in agg.items()}
    print("constant probe (pair): " + "  ".join(f"{s} whole {v['whole']} slots {v['slots']}"
                                               for s, v in rep["probe"].items())
          + "   (the teacher's 4-word chunks: clut 84.8, plain 62.8, demo 87.8)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--n-train", type=int, default=200_000)
    p.add_argument("--n-held", type=int, default=5_000)
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--cue", action="store_true",
                   help="append texts with a direction or time word not parsed yet")
    p = sub.add_parser("teacher")
    p = sub.add_parser("train")
    p.add_argument("--body", required=True,
                   help="electra | tern-electra:<tern_electra.py model.pt>")
    p.add_argument("--span-rows", type=int, default=0,
                   help="also train on this many role spans as texts of their own")
    p.add_argument("--layers", type=int, default=None, help="ELECTRA layers kept (default all 12)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="first N training texts (smoke)")
    p = sub.add_parser("check")
    p.add_argument("--head", required=True)
    p.add_argument("--body", default=None,
                   help="run the head on another body than it trained on, e.g. "
                        "tern-electra:reader/tern_electra/e1/model.pt (step 1's gate)")
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    return {"parse": cmd_parse, "teacher": cmd_teacher, "train": cmd_train, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
