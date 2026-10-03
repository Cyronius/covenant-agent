"""ELECTRA as the whole reader (.claude/plans/electra-only-reader.md, step 0).

The ternary ELECTRA (tern_electra.py, reader/tern_electra/e1, frozen) reads a
text, and an embedding head turns a stretch of its words into one 384-number
unit vector in the teacher's space (all-MiniLM-L6-v2). A piece of a request
(a tagged role's words, a 4-word chunk) can be read two ways:

  alone    ELECTRA reads the piece by itself, one pass per piece
  context  ELECTRA reads the whole request once, and the head pools the
           piece's words out of that one pass

Constants, fields and tool descriptions are texts of their own, read alone
either way. Two heads of the same shape, trained on the same pieces and
targets:

  alone  every piece read alone
  mixed  half its pieces read alone, half in context

The head: a learned mix of ELECTRA's layers, one attention layer that sees
only the piece's own words, a feed-forward per word, then the mean over the
piece's words. [CLS] and [SEP] are never part of a piece, so a text read alone
and the same words pooled out of a longer text differ only by what ELECTRA
saw around them.

  python electra_reader.py targets                    # -> reader/embed/targets.pt
  python electra_reader.py train --mode alone --out reader/embed/a
  python electra_reader.py train --mode mixed --out reader/embed/b
  python electra_reader.py check --head a=reader/embed/a/head.pt --head b=reader/embed/b/head.pt \\
      --out ../../results/logs/embed/check.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from slot_reader import ROOT, held_out, load_roles
from slot_tagger import SpanTagger, Tok, decode

HERE = Path(__file__).parent
EMBED = HERE / "reader" / "embed"
BODY = HERE / "reader" / "tern_electra" / "e1" / "model.pt"
TAGGER = HERE / "reader" / "slots" / "t_tern_e1" / "tagger.pt"
NAMES = ROOT / "data" / "general" / "names.jsonl"
TEACHER = "sentence-transformers/all-MiniLM-L6-v2"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOWS = (2, 3, 4, 6)
torch.backends.cuda.matmul.allow_tf32 = True


class MiniLM:
    """The teacher as sentence-transformers runs it: mean over its word
    pieces ([CLS] and [SEP] included), unit length."""

    def __init__(self):
        from tern_electra import _hf
        AutoModel, AutoTokenizer = _hf()
        self.tk = AutoTokenizer.from_pretrained(TEACHER)
        self.m = AutoModel.from_pretrained(TEACHER).eval().to(DEV)
        self.name = "all-MiniLM-L6-v2"

    @torch.no_grad()
    def vectors(self, texts: list[str], bs: int = 1024, log_every: int = 0) -> torch.Tensor:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), 384)
        t0 = time.time()
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = self.tk([texts[i] for i in idx], padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to(DEV)
            h = self.m(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[idx] = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1).float().cpu()
            if log_every and (s // bs) % log_every == 0:
                print(f"  teacher {s + len(idx)}/{len(texts)}  {(s + len(idx)) / (time.time() - t0):.0f}/s",
                      flush=True)
        return out


class Body:
    """The frozen ternary ELECTRA: word pieces with character offsets, and
    every state (embeddings, then each layer)."""

    def __init__(self, path: str | Path = BODY):
        from tern_electra import TernElectra
        self.m = TernElectra.load(Path(path)).eval().to(DEV)
        self.tok = Tok()
        self.n_states, self.d = self.m.dims["layers"] + 1, self.m.dims["d"]

    @torch.no_grad()
    def __call__(self, texts: list[str]):
        """-> states (L, B, T, d) and mask (B, T) on DEV, offsets (B, T, 2) on the CPU."""
        ids, mask, offsets = self.tok(texts)
        states, _ = self.m(ids.to(DEV), mask.to(DEV))
        return torch.stack(states), mask.to(DEV), offsets


class EmbedHead(nn.Module):
    def __init__(self, n_states: int = 13, d: int = 256, heads: int = 4, ffn: int = 1024, out: int = 384):
        super().__init__()
        self.cfg = dict(n_states=n_states, d=d, heads=heads, ffn=ffn, out=out)
        self.mix = nn.Parameter(torch.zeros(n_states))
        self.ln = nn.LayerNorm(d)
        self.att = nn.TransformerEncoderLayer(d, heads, ffn, dropout=0.0, batch_first=True, norm_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, out))

    def forward(self, states: torch.Tensor, rows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """states (L, B, T, d); rows (P,) the text each piece is in; mask (P, T)
        the piece's word pieces -> (P, out) unit vectors."""
        h = self.ln((F.softmax(self.mix, 0)[:, None, None, None] * states).sum(0))[rows]
        h = self.att(h, src_key_padding_mask=~mask)
        z = self.out(h)
        m = mask.unsqueeze(-1).float()
        return F.normalize((z * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


def piece_masks(offsets: torch.Tensor, rows: list[int], spans: list[tuple[int, int]]) -> torch.Tensor:
    """(P, T) bool: the word pieces of text rows[p] inside characters spans[p]."""
    off = offsets[rows]
    c = torch.tensor(spans, dtype=off.dtype)
    return (off[..., 0] >= c[:, :1]) & (off[..., 1] <= c[:, 1:]) & (off[..., 1] > off[..., 0])


def read_alone(head: EmbedHead, body: Body, texts: list[str]):
    """Each text read by itself -> (vectors (P', out), indices of the texts kept:
    the ones with at least one word piece)."""
    states, _, offsets = body(texts)
    word = offsets[..., 1] > offsets[..., 0]
    keep = [i for i in range(len(texts)) if word[i].any()]
    rows = torch.tensor(keep, dtype=torch.long, device=DEV)
    return head(states, rows, word[keep].to(DEV)), keep


def read_context(head: EmbedHead, body: Body, texts: list[str], pieces: list[tuple[int, int, int]]):
    """Each text read once; pieces (text index, char start, char end) pooled
    out of its pass -> (vectors, indices of the pieces kept)."""
    states, _, offsets = body(texts)
    mask = piece_masks(offsets, [p[0] for p in pieces], [(p[1], p[2]) for p in pieces])
    keep = [i for i in range(len(pieces)) if mask[i].any()]
    rows = torch.tensor([pieces[i][0] for i in keep], dtype=torch.long, device=DEV)
    return head(states, rows, mask[keep].to(DEV)), keep


def chunk_spans(text: str, size: int) -> list[tuple[int, int, str]]:
    """prep.chunk_request's windows, with each window's character span."""
    from prep import chunk_request
    w = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if len(w) <= size:
        starts, n = [0], len(w)
    else:
        step = max(1, size // 2)
        starts = list(range(0, len(w) - size + 1, step))
        if starts[-1] + size < len(w):
            starts.append(len(w) - size)
        n = size
    out = [(w[s][0], w[min(s + n, len(w)) - 1][1]) for s in starts]
    out = [(a, b, " ".join(text[a:b].split())) for a, b in out]
    assert [t for _, _, t in out] == chunk_request(text, size), text
    return out


def with_pieces(row: dict, rng: random.Random) -> dict:
    """A parsed general text and its pieces (char start, char end, text): the
    whole text, each role's words, and windows of 2, 3, 4 or 6 words."""
    t = row["text"]
    pieces = {(0, len(t)): t}
    for role, (a, b) in row["roles"].items():
        if role != "whole" and t[a:b].strip():
            pieces.setdefault((a, b), t[a:b])
    for a, b, w in chunk_spans(t, rng.choice(WINDOWS)):
        pieces.setdefault((a, b), w)
    return {"text": t, "pieces": [[a, b, s] for (a, b), s in pieces.items()]}


# ── names: people, companies, references, titles (R27: ID constants) ─────────

NAME_LOCALES = ("en_US", "en_GB", "en_IE", "en_AU", "en_IN", "de_DE", "fr_FR", "es_ES", "es_MX", "it_IT",
                "nl_NL", "pt_BR", "pl_PL", "sv_SE", "da_DK", "fi_FI", "tr_TR", "cs_CZ", "hu_HU", "ro_RO",
                "id_ID", "vi_VN")
SUFFIXES = ("Group", "Systems", "Partners", "Works", "Industries", "Holdings", "Studio", "Ventures",
            "Logistics", "Foods", "Health", "Media", "Supply", "Freight", "Clinic", "Farms")
STOP_CAPS = {"the", "a", "an", "of", "and", "for", "to", "in", "on", "at", "by", "with", "from"}
NAME_KINDS = {"person": 0.30, "first": 0.05, "last": 0.05, "company": 0.15, "ref": 0.25, "title": 0.15,
              "numbered": 0.05}


def _words(t: str) -> set[str]:
    return {re.sub(r"'s$", "", w.lower()) for w in re.findall(r"[A-Za-z][A-Za-z'-]*", t)}


def exam_words() -> tuple[set[str], set[str]]:
    """What a generated name must not contain: any capitalized word of an ID
    constant in the probe's suites or of the generator's own lists
    (data/gen/worldgen.py); and any such constant whole (lower case)."""
    import importlib.util
    from chunk_probe import SUITES, kind, tasks
    from prep import reader_text
    words, whole = set(), set()

    def add(t: str) -> None:
        whole.add(t.lower())
        words.update(re.sub(r"'s$", "", w.lower()) for w in re.findall(r"[A-Za-z][A-Za-z'-]*", t)
                     if w[0].isupper() and w.lower() not in STOP_CAPS)

    for path in SUITES.values():
        for _, _, consts, _ in tasks(path, dedupe=False):
            for _, typ, v in consts:
                if kind(typ) == "id":
                    add(reader_text(v))
    spec = importlib.util.spec_from_file_location("worldgen", ROOT / "data" / "gen" / "worldgen.py")
    wg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wg)
    for t in wg.FIRST_NAMES + wg.LAST_NAMES + wg.COMPANY_NAMES + wg.CARD_TITLES + wg.PROJECT_NAMES:
        add(t)
    return words, whole


def make_names(n: int, rng: random.Random, seed: int, banned: set[str], banned_whole: set[str],
               lemmas: list[str]) -> dict[str, list[str]]:
    """About n short name-like texts, by kind (NAME_KINDS): people from Faker's
    locales, companies (Faker's, or a WordNet word with an optional suffix),
    "<person or company>'s <thing> id", titles (Faker catch phrases, "The
    <word>", 2-4 WordNet words) and numbered records. None shares a word with
    exam_words()."""
    from faker import Faker
    Faker.seed(seed)
    fakes = [Faker(loc) for loc in NAME_LOCALES]
    en = Faker("en_US")

    def person():
        f = rng.choice(fakes)
        return f"{f.first_name()} {f.last_name()}"

    def company():
        if rng.random() < 0.5:
            return en.company()
        w = rng.choice(lemmas).capitalize()
        return w if rng.random() < 0.5 else f"{w} {rng.choice(SUFFIXES)}"

    def title():
        r = rng.random()
        if r < 0.3:
            return en.catch_phrase()
        if r < 0.45:
            return "The " + rng.choice(lemmas).capitalize()
        t = " ".join(rng.choice(lemmas) for _ in range(rng.randint(2, 4)))
        return t.capitalize() if rng.random() < 0.5 else t.title()

    makers = {
        "person": person,
        "first": lambda: rng.choice(fakes).first_name(),
        "last": lambda: rng.choice(fakes).last_name(),
        "company": company,
        "ref": lambda: f"{rng.choice([person, company])()}'s "
                       + ("id" if rng.random() < 0.2 else f"{rng.choice(lemmas)} id"),
        "title": title,
        "numbered": lambda: f"{rng.choice(lemmas)} {rng.randint(1, 9999)}"
                            + ("" if rng.random() < 0.5 else f" - {title()}"),
    }
    out = {}
    for kind, make in makers.items():
        want, got, tries = int(n * NAME_KINDS[kind]), set(), 0
        while len(got) < want and tries < 20 * want:
            tries += 1
            t = " ".join(make().split())
            if t and t.lower() not in banned_whole and not (_words(t) & banned):
                got.add(t)
        out[kind] = sorted(got)
        rng.shuffle(out[kind])
    return out


def substituted(rows: list[dict], names: list[str], n: int, rng: random.Random) -> list[dict]:
    """General texts with one object / destination / source span replaced by a
    generated name, as pieces: the whole text, the name's span, windows."""
    slots = ("object", "destination", "source")
    cands = [r for r in rows if any(k in r["roles"] for k in slots)]
    out = []
    for r in rng.sample(cands, min(n, len(cands))):
        a, b = r["roles"][rng.choice([k for k in slots if k in r["roles"]])]
        name = rng.choice(names)
        t = r["text"][:a] + name + r["text"][b:]
        pieces = {(0, len(t)): t, (a, a + len(name)): name}
        for c0, c1, w in chunk_spans(t, rng.choice(WINDOWS)):
            pieces.setdefault((c0, c1), w)
        out.append({"text": t, "pieces": [[c0, c1, s] for (c0, c1), s in pieces.items()]})
    return out


def cmd_targets(args) -> int:
    from prep import reader_text
    rng = random.Random(args.seed)
    train = load_roles("train")
    rng.shuffle(train)
    rows = {"train": [with_pieces(r, rng) for r in train[:args.n_texts]],
            "held": [with_pieces(r, rng) for r in load_roles("held")[:args.n_held]]}
    names = list(dict.fromkeys(n for n in (reader_text(json.loads(line)["text"])
                                           for line in NAMES.open(encoding="utf-8")) if n))
    rng.shuffle(names)
    names = {"train": [n for n in names if not held_out(n)][:args.n_names],
             "held": [n for n in names if held_out(n)][:args.n_held]}
    texts = list(dict.fromkeys([s for split in rows.values() for r in split for _, _, s in r["pieces"]]
                               + names["train"] + names["held"]))
    extra = None
    if args.names_extra:
        banned, banned_whole = exam_words()
        lemmas = []
        for line in NAMES.open(encoding="utf-8"):
            d = json.loads(line)
            x = d["text"]
            if d["source"] == "wordnet" and x.isalpha() and x.islower() and 4 <= len(x) <= 10 and x not in banned:
                lemmas.append(x)
        lemmas = list(dict.fromkeys(lemmas))
        made = make_names(args.names_extra, rng, args.seed, banned, banned_whole, lemmas)
        flat = [t for ts in made.values() for t in ts]
        rng.shuffle(flat)
        ntr, nho = [t for t in flat if not held_out(t)], [t for t in flat if held_out(t)]
        extra = {"kinds": {k: v[:5] + [len(v)] for k, v in made.items()},
                 "names": {"train": ntr, "held": nho[:args.n_held]},
                 "rows": {"train": substituted(train[:args.n_texts], ntr, args.sub_rows, rng),
                          "held": substituted(load_roles("held")[:args.n_held], nho, args.n_held // 4, rng)}}
        for k, v in extra["kinds"].items():
            print(f"  {k} ({v[-1]}): {v[:-1]}")
        print(f"  {len(banned)} banned words, e.g. {sorted(banned)[:12]}")
        texts = list(dict.fromkeys(texts + ntr + extra["names"]["held"]
                                   + [s for split in extra["rows"].values() for r in split for _, _, s in r["pieces"]]))
    t0 = time.time()
    table = MiniLM().vectors(texts, log_every=100)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "names": names, "extra": extra, "texts": texts, "table": table.half(),
                "args": vars(args)}, out)
    n_pieces = sum(len(r["pieces"]) for r in rows["train"])
    print(f"{len(texts)} texts ({n_pieces} train pieces in {len(rows['train'])} texts, "
          f"{len(names['train'])} names"
          + (f", {len(extra['names']['train'])} generated names, {len(extra['rows']['train'])} texts "
             f"with one" if extra else "") + f") in {time.time() - t0:.0f}s -> {out}")
    return 0


def embed_loss(v: torch.Tensor, t: torch.Tensor, lam_rel: float):
    """1 - cosine to the teacher's vector, plus how far the batch's pairwise
    cosines are from the teacher's (squared, off the diagonal)."""
    cos = 1 - (v * t).sum(-1).mean()
    off = ~torch.eye(len(v), dtype=torch.bool, device=v.device)
    rel = ((v @ v.T - t @ t.T)[off] ** 2).mean()
    return cos + lam_rel * rel, cos, rel


class Targets:
    def __init__(self, path: str | Path):
        d = torch.load(path)
        self.rows, self.names, self.texts = d["rows"], d["names"], d["texts"]
        self.extra = d.get("extra")
        self.index = {t: i for i, t in enumerate(self.texts)}
        self.table = d["table"].to(DEV)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return self.table[torch.tensor([self.index[t] for t in texts], device=DEV)].float()


def held_cos(head, body, tg: Targets, rows: list[dict], names: list[str]) -> dict:
    """Mean cosine to the teacher on held-out pieces, read alone and in context."""
    head.eval()
    with torch.no_grad():
        pieces = [(i, a, b, s) for i, r in enumerate(rows) for a, b, s in r["pieces"]]
        alone = [s for *_, s in pieces] + names
        va, ka = [], []
        for s in range(0, len(alone), 512):
            v, k = read_alone(head, body, alone[s:s + 512])
            va.append(v)
            ka += [alone[s + i] for i in k]
        ca = float((torch.cat(va) * tg(ka)).sum(-1).mean())
        vc, kc = [], []
        for s in range(0, len(rows), 128):
            sub = [(i - s, a, b, t) for i, a, b, t in pieces if s <= i < s + 128]
            v, k = read_context(head, body, [r["text"] for r in rows[s:s + 128]], [p[:3] for p in sub])
            vc.append(v)
            kc += [sub[i][3] for i in k]
        cc = float((torch.cat(vc) * tg(kc)).sum(-1).mean())
    head.train()
    return {"cos_alone": round(ca, 4), "cos_context": round(cc, 4)}


def held_report(head, body, tg: Targets, n: int) -> dict:
    """held_cos on general text, and on the generated names (alone) and the
    texts carrying one (in context) when the targets have them."""
    r = held_cos(head, body, tg, tg.rows["held"][:n], tg.names["held"][:n])
    if tg.extra:
        x = held_cos(head, body, tg, tg.extra["rows"]["held"][:n], tg.extra["names"]["held"][:n])
        r.update({f"names_{k}": v for k, v in x.items()})
    return r


def cmd_train(args) -> int:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tg = Targets(args.targets)
    rows = tg.rows["train"]
    pool = [s for r in rows for _, _, s in r["pieces"]] + tg.names["train"]
    if args.name_frac and not tg.extra:
        raise SystemExit(f"--name-frac needs targets built with --names-extra ({args.targets})")
    n_ax = round(args.name_frac * args.alone_batch)
    n_cx = round(args.name_frac * args.ctx_batch)
    body = Body(args.body)
    head = EmbedHead(body.n_states, body.d).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, args.steps // 30)
    log = (out / "log.jsonl").open("w", encoding="utf-8")
    t0, acc = time.time(), defaultdict(float)
    for step in range(1, args.steps + 1):
        mult = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm)))
        for g in opt.param_groups:
            g["lr"] = args.lr * mult
        alone = rng.sample(pool, args.alone_batch - n_ax)
        ctx = rng.sample(rows, args.ctx_batch - n_cx)
        if n_ax:
            alone += rng.sample(tg.extra["names"]["train"], n_ax)
            ctx += rng.sample(tg.extra["rows"]["train"], n_cx)
        pieces = [(i, a, b, s) for i, r in enumerate(ctx) for a, b, s in r["pieces"]]
        v1, k1 = read_alone(head, body, alone)
        l1, c1, r1 = embed_loss(v1, tg([alone[i] for i in k1]), args.lam_rel)
        if args.mode == "alone":
            v2, k2 = read_alone(head, body, [s for *_, s in pieces])
        else:
            v2, k2 = read_context(head, body, [r["text"] for r in ctx], [p[:3] for p in pieces])
        l2, c2, r2 = embed_loss(v2, tg([pieces[i][3] for i in k2]), args.lam_rel)
        loss = (l1 + l2) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        for k, v in (("loss", loss), ("cos_a", c1), ("rel_a", r1), ("cos_b", c2), ("rel_b", r2)):
            acc[k] += float(v.detach())
        if step % args.log_every == 0 or step == args.steps:
            row = {"step": step, **{k: round(v / args.log_every, 5) for k, v in acc.items()},
                   "sec": round(time.time() - t0)}
            acc.clear()
            if step % args.eval_every == 0 or step == args.steps:
                row.update(held_report(head, body, tg, args.n_eval))
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
    torch.save({"cfg": head.cfg, "state": head.state_dict(), "mode": args.mode, "body": str(args.body),
                "mix": F.softmax(head.mix, 0).tolist(), "args": vars(args)}, out / "head.pt")
    print(f"layer mix: {[round(x, 2) for x in F.softmax(head.mix, 0).tolist()]}")
    print(f"wrote {out / 'head.pt'}")
    return 0


# ── checks ────────────────────────────────────────────────────────────────────

def load_head(path: str | Path) -> EmbedHead:
    ck = torch.load(path, map_location="cpu")
    h = EmbedHead(**ck["cfg"])
    h.load_state_dict(ck["state"])
    return h.to(DEV).eval()


class Reading:
    """One way of turning a request's pieces and a task's constants into
    vectors: the teacher alone, or a head alone or in context."""

    def __init__(self, name: str, body: Body | None, head: EmbedHead | None, mode: str, teacher: MiniLM | None):
        self.name, self.body, self.head, self.mode, self.teacher = name, body, head, mode, teacher

    @torch.no_grad()
    def alone(self, texts: list[str]) -> torch.Tensor:
        if self.teacher is not None:
            return self.teacher.vectors(texts)
        out = torch.zeros(len(texts), self.head.cfg["out"])
        for s in range(0, len(texts), 512):
            v, k = read_alone(self.head, self.body, texts[s:s + 512])
            out[[s + i for i in k]] = v.float().cpu()
        return out

    @torch.no_grad()
    def pieces(self, text: str, spans: list[tuple[int, int, str]]) -> torch.Tensor:
        if self.mode == "alone":
            return self.alone([s for _, _, s in spans])
        out = torch.zeros(len(spans), self.head.cfg["out"])
        v, k = read_context(self.head, self.body, [text], [(0, a, b) for a, b, _ in spans])
        out[k] = v.float().cpu()
        return out


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def probe(readings: list[Reading], spans_of: dict, limit: int | None) -> dict:
    """The constant probe (chunk_probe.py's tasks and pair accuracy): each
    constant read alone, scored by its best cosine over the request's pieces.
    Methods: whole (the request), chunk-4, slots (the request + the tagger's
    role spans, as R26) and slots+chunks; overlap is chunk_probe's no-model
    word overlap."""
    from chunk_probe import SUITES, kind, rank_stats, tasks, words
    from prep import reader_text
    rep = {}
    for suite, path in SUITES.items():
        agg = defaultdict(lambda: [0.0, 0])
        by_kind = defaultdict(lambda: [0.0, 0])

        def add_kinds(method, sc, used, syms, kinds):
            for i, s in enumerate(syms):
                if s not in used:
                    continue
                for j, s2 in enumerate(syms):
                    if s2 not in used:
                        bk = by_kind[(method, kinds[i])]
                        bk[0] += 1.0 if sc[i] > sc[j] else 0.5 if sc[i] == sc[j] else 0.0
                        bk[1] += 1
        n = 0
        for tid, request, consts, used in tasks(path, dedupe=(suite == "demo")):
            syms = [x for x, _, _ in consts]
            if not (set(syms) & used) or not (set(syms) - used):
                continue
            n += 1
            if limit and n > limit:
                break
            ctexts = [reader_text(v) for _, _, v in consts]
            kinds = [id_kind(typ, t) for (_, typ, _), t in zip(consts, ctexts)]
            ov = [len(words(t) & words(request)) / max(1, len(words(t))) for t in ctexts]
            w, p, _ = rank_stats(ov, used, syms)
            agg[("overlap", "-")][0] += w
            agg[("overlap", "-")][1] += p
            add_kinds("chunk-4 | overlap", ov, used, syms, kinds)
            whole = [(0, len(request), request)]
            tagged = [(a, b, request[a:b]) for a, b in spans_of[request]]
            chunks = chunk_spans(request, 4)
            for rd in readings:
                cv = rd.alone(ctexts)
                pv = rd.pieces(request, whole + tagged + chunks)
                nw, nt = 1, 1 + len(tagged)
                for m, M in (("whole", pv[:nw]), ("chunk-4", pv[nt:]), ("slots", pv[:nt]),
                             ("slots+chunks", pv)):
                    sc = (cv @ M.T).max(1).values.tolist()
                    w, p, _ = rank_stats(sc, used, syms)
                    agg[(m, rd.name)][0] += w
                    agg[(m, rd.name)][1] += p
                    if m in ("chunk-4", "slots+chunks"):
                        add_kinds(f"{m} | {rd.name}", sc, used, syms, kinds)
        rep[suite] = {"tasks": min(n, limit) if limit else n,
                      **{f"{m} | {r}": round(100 * a / b, 1) for (m, r), (a, b) in agg.items()},
                      "by_kind": {f"{m} | {k}": [round(100 * a / b, 1), b] for (m, k), (a, b) in by_kind.items()}}
    return rep


def id_kind(typ: str, text: str) -> str:
    """chunk_probe.kind, with IDs split three ways: a reference ("Acme's
    vineyard id"), a proper name (every word capitalized: "Bob Alvarez",
    "Acme") or a title ("Alfalfa hay")."""
    from chunk_probe import kind
    k = kind(typ)
    if k != "id":
        return k
    if "'s" in text:
        return "id-ref"
    ws = re.findall(r"[A-Za-z0-9][\w'-]*", text)
    return "id-name" if ws and all(w[0].isupper() or w[0].isdigit() for w in ws) else "id-title"


def cmd_check(args) -> int:
    from chunk_probe import CALIBRATION, ROLE_PAIRS, SUITES, tasks
    body = Body(args.body)
    ck = torch.load(args.tagger, map_location="cpu")
    tagger = SpanTagger(**ck["cfg"])
    tagger.load_state_dict(ck["state"])
    tagger = tagger.to(DEV).eval()
    teacher = MiniLM()
    heads = {}
    for spec in args.head:
        name, path = spec.split("=", 1)
        heads[name] = load_head(path)
    readings = [Reading("teacher", None, None, "alone", teacher)]
    for name, h in heads.items():
        readings.append(Reading(f"{name} alone", body, h, "alone", None))
        readings.append(Reading(f"{name} context", body, h, "context", None))
    rep = {"body": str(args.body), "tagger": str(args.tagger), "heads": {k: str(v) for k, v in
                                                                         (s.split("=", 1) for s in args.head)}}

    # held-out pieces: cosine to the teacher, and rank agreement over random pairs
    tg = Targets(args.targets)
    rows, names = tg.rows["held"], tg.names["held"]
    rep["held"] = {}
    rng = random.Random(0)
    alone_texts = [s for r in rows for _, _, s in r["pieces"]] + names
    pa = [rng.randrange(len(alone_texts)) for _ in range(args.n_pairs)]
    pb = [rng.randrange(len(alone_texts)) for _ in range(args.n_pairs)]
    t_all = tg(alone_texts).cpu()
    t_pair = (t_all[pa] * t_all[pb]).sum(-1)
    for name, h in heads.items():
        r = held_report(h, body, tg, len(rows))
        v = Reading(name, body, h, "alone", None).alone(alone_texts)
        r["rank_alone"] = round(spearman((v[pa] * v[pb]).sum(-1), t_pair), 4)
        rep["held"][name] = r
        print(f"held-out, head {name}: cosine to the teacher alone {r['cos_alone']}, in context "
              f"{r['cos_context']}; pair-rank agreement alone {r['rank_alone']}"
              + (f"; generated names alone {r['names_cos_alone']}, in context {r['names_cos_context']}"
                 if "names_cos_alone" in r else ""), flush=True)

    # the tagger's role spans for every probe request
    reqs = list(dict.fromkeys(req for path in SUITES.values() for _, req, _, _ in tasks(path, dedupe=False)))
    spans_of = {}
    for s in range(0, len(reqs), 256):
        st, mask, offsets = body(reqs[s:s + 256])
        with torch.no_grad():
            a, b = tagger(st, mask)
        for t, row in zip(reqs[s:s + 256], decode(a.cpu(), b.cpu(), offsets)):
            spans_of[t] = [x for x in row if x is not None]

    rep["probe"] = probe(readings, spans_of, args.limit_tasks)
    print("\nconstant probe, pair accuracy (cluttered / plain / demo):")
    methods = ["whole", "chunk-4", "slots", "slots+chunks"]
    order = ("clut", "plain", "demo")
    print(f"  {'overlap (no model)':34s} " + " / ".join(f"{rep['probe'][s]['overlap | -']:5.1f}" for s in order))
    for rd in readings:
        for m in methods:
            print(f"  {m + ' | ' + rd.name:34s} " + " / ".join(f"{rep['probe'][s][f'{m} | {rd.name}']:5.1f}"
                                                            for s in order))

    print("\nchunk-4 by the used constant's kind, plain / cluttered (pairs on plain):")
    kinds = sorted({k.rsplit(" | ", 1)[1] for k in rep["probe"]["plain"]["by_kind"]})
    names_m = ["chunk-4 | overlap"] + [f"chunk-4 | {rd.name}" for rd in readings]
    print(f"  {'':24s} " + "".join(f"{k:>16s}" for k in kinds))
    for m in names_m:
        cells = []
        for k in kinds:
            pl = rep["probe"]["plain"]["by_kind"].get(f"{m} | {k}")
            cl = rep["probe"]["clut"]["by_kind"].get(f"{m} | {k}")
            cells.append(f"{(pl or ['-'])[0]:>6} / {(cl or ['-'])[0]:>6}" if pl or cl else f"{'-':>15}")
        print(f"  {m[len('chunk-4 | '):]:24s} " + "".join(f"{c:>16s}" for c in cells))
    print("  pairs (plain): " + ", ".join(f"{k} {rep['probe']['plain']['by_kind'].get(f'chunk-4 | teacher | {k}', [0, 0])[1]}"
                                        for k in kinds))

    rep["pairs"] = {}
    for rd in readings:
        if rd.mode != "alone":
            continue
        v = rd.alone([x for p in ROLE_PAIRS + CALIBRATION for x in p])
        rep["pairs"][rd.name] = {f"{a} | {b}": round(float(v[2 * i] @ v[2 * i + 1]), 3)
                                 for i, (a, b) in enumerate(ROLE_PAIRS + CALIBRATION)}

    # step 0's gates
    def score(m, r, s):
        return rep["probe"][s][f"{m} | {r}"]
    gates = {}
    for name in heads:
        for mode in ("alone", "context"):
            gates[f"{name} {mode} chunk-4 within 2 of the teacher"] = all(
                score("chunk-4", f"{name} {mode}", s) >= score("chunk-4", "teacher", s) - 2 for s in order)
    if {"a", "b"} <= set(heads):
        for m in ("chunk-4", "slots+chunks"):
            gates[f"b context {m} within 1 of a alone"] = all(
                score(m, "b context", s) >= score(m, "a alone", s) - 1 for s in order)
    rep["gates"] = gates
    print("\ngates:")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'miss'}  {k}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("targets")
    p.add_argument("--out", default=str(EMBED / "targets.pt"))
    p.add_argument("--n-texts", type=int, default=200_000)
    p.add_argument("--n-names", type=int, default=100_000)
    p.add_argument("--n-held", type=int, default=2000)
    p.add_argument("--names-extra", type=int, default=0,
                   help="also generate about this many name-like texts (make_names), held out by crc32")
    p.add_argument("--sub-rows", type=int, default=40_000,
                   help="with --names-extra: general texts with one span replaced by a generated name")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("train")
    p.add_argument("--mode", choices=["alone", "mixed"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--targets", default=str(EMBED / "targets.pt"))
    p.add_argument("--body", default=str(BODY))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--alone-batch", type=int, default=512)
    p.add_argument("--ctx-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam-rel", type=float, default=1.0)
    p.add_argument("--name-frac", type=float, default=0.0,
                   help="share of each batch drawn from the generated names and the texts carrying one")
    p.add_argument("--n-eval", type=int, default=300)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("check")
    p.add_argument("--head", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--targets", default=str(EMBED / "targets.pt"))
    p.add_argument("--body", default=str(BODY))
    p.add_argument("--tagger", default=str(TAGGER))
    p.add_argument("--n-pairs", type=int, default=5000)
    p.add_argument("--limit-tasks", type=int, default=None)
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    return {"targets": cmd_targets, "train": cmd_train, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
