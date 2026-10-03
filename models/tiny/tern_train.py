"""Teach Ternlight word order (.claude/plans/role-aware-reader.md, steps 0,
2, 3 and 4).

A role flip changes who-does-what and keeps the words: "move it to bob" ->
"move it from bob", "copies A to B" -> "copies B to A", "cards that are
done" -> "cards that are not done". The shipped reader gives a sentence and
its flip nearly the same vector (C0's probe: 0.92-1.00). Here the reader
(tern_reader.py, with a position table) learns to copy the teacher
(all-MiniLM-L6-v2) on general English and on its flips.

Flips come from data/general/texts.jsonl only. A tenth of the source texts
(crc32 % 10 == 0) are held out for the checks. Our own requests are never
trained on; they are only checked.

  python tern_train.py flips                # -> reader/role/flips.jsonl
  python tern_train.py teacher-probe        # step 0: does the teacher keep flips apart?
  python tern_train.py teacher              # teacher vectors for every training text
  python tern_train.py train --out reader/role/r1 [--push 0.85 --lam-push 1]
  python tern_train.py check --model reader/role/r1/reader.pt   # step 4's gates
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from tern_reader import TERN_DIR, Tokenizer, TorchReader, TernReader, shuffled

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
GENERAL = ROOT / "data" / "general" / "texts.jsonl"
ROLE = HERE / "reader" / "role"
TEACHER = "sentence-transformers/all-MiniLM-L6-v2"

# ── flips (step 2) ────────────────────────────────────────────────────────────

SWAP = {"to": ["from"], "from": ["to"], "into": ["out", "of"], "before": ["after"],
        "after": ["before"], "above": ["below"], "below": ["above"], "with": ["without"],
        "without": ["with"], "more": ["less"], "less": ["more"], "first": ["last"],
        "last": ["first"]}
AUX = {"is", "are", "was", "were", "do", "does", "can", "should"}
CONTRACTED = {"isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were",
              "don't": "do", "doesn't": "does", "can't": "can", "cannot": "can",
              "shouldn't": "should"}
# "to" as part of a fixed phrase, not a direction: "due to", "prior to", ...
TO_FIXED = {"due", "according", "prior", "respect", "regard", "addition", "order",
            "relation", "able", "unable", "have", "has", "had", "going", "want", "wants",
            "need", "needs", "used", "how", "what", "where", "whether", "ought"}
REL = {"to", "from", "into", "onto", "than", "before", "after", "with", "without", "for", "by"}
NP_TAGS = {"DT", "PRP$", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "CD", "PRP"}
HEAD_TAGS = {"NN", "NNS", "NNP", "NNPS", "CD", "PRP"}
VERB_TAGS = {"VBZ", "VBD", "VBP"}
SURE_START = {"DT", "PRP$", "PRP", "CD", "NNP", "NNPS"}
MAX_NP = 4
MAX_WORDS = 24
ROLE_FAMILIES = ("word", "not", "swap", "ant")    # "shuf" rows train word order only
ANT_POS = {"JJ": ("a", ""), "JJR": ("a", "er"), "JJS": ("a", "est"),
           "VB": ("v", ""), "VBP": ("v", ""), "VBZ": ("v", "s")}


def _wn_antonym(base: str, pos: str) -> str | None:
    """The first one-word antonym of `base` in WordNet, most common sense first."""
    from nltk.corpus import wordnet as wn
    for syn in wn.synsets(base, pos):
        for lem in syn.lemmas():
            if lem.name() == base:
                for a in lem.antonyms():
                    if "_" not in a.name():
                        return a.name()
    return None


def _inflect(word: str, suffix: str) -> str:
    if not suffix:
        return word
    if suffix == "s":
        return word + ("es" if word.endswith(("s", "x", "z", "ch", "sh")) else "s")
    if word.endswith("e"):
        return word + suffix[1:]
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        return word[:-1] + "i" + suffix
    if (len(word) <= 4 and word[-1] in "bdgmnpt" and word[-2] in "aeiou"
            and (len(word) < 3 or word[-3] not in "aeiou")):
        return word + word[-1] + suffix
    return word + suffix


_VOCAB: set[str] | None = None


def corpus_vocab() -> set[str]:
    """Every lower-case word of the general corpus: an inflected antonym has
    to be one ("biggest" -> "littlest" is, "earliest" -> "middlest" is not)."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = set()
        for line in GENERAL.open(encoding="utf-8"):
            _VOCAB.update(re.findall(r"[a-z]+", json.loads(line)["text"].lower()))
    return _VOCAB


def antonym(word: str, tag: str) -> str | None:
    """A word's WordNet antonym in the same form: an adjective as is, -er or
    -est (oldest -> youngest), a verb as is or -s (increases -> decreases).
    An inflected form must occur in the general corpus."""
    if tag not in ANT_POS:
        return None
    pos, suffix = ANT_POS[tag]
    w = word.lower()
    if not suffix:
        bases = [w]
    elif suffix == "s":
        bases = [w[:-1], w[:-2]] if w.endswith("s") else []
    else:
        n = len(suffix)
        bases = [w[:-n], w[:-n] + "e", w[:-n - 1] + "y", w[:-n - 1]] if w.endswith(suffix) else []
    for base in bases:
        if len(base) > 1 and (a := _wn_antonym(base, pos)) and _inflect(base, suffix) == w:
            out = _inflect(a, suffix)
            return out if not suffix or out in corpus_vocab() else None
    return None


def held_out(text: str) -> bool:
    return zlib.crc32(text.encode("utf-8")) % 10 == 0


def _parts(w: str) -> tuple[str, str, str]:
    m = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$", w)
    return m.group(1), m.group(2), m.group(3)


class Flipper:
    """Every role flip of a text, as (kind, start, end, replacement words):
    the flip is words[:start] + replacement + words[end:]."""

    def __init__(self):
        from nltk.tag import PerceptronTagger
        self.tagger = PerceptronTagger()

    def sites(self, text: str) -> tuple[list[str], list]:
        words = text.split()
        parts = [_parts(w) for w in words]
        core = [c.lower() for _, c, _ in parts]
        tags = ["SYM"] * len(words)
        idx = [i for i, c in enumerate(core) if c]
        if idx:
            for i, (_, t) in zip(idx, self.tagger.tag([parts[i][1] for i in idx])):
                tags[i] = t
        for i in idx[1:]:
            if parts[i][1][0].isupper() and tags[i].startswith("VB"):
                tags[i] = "NNP"         # "assign card 6 to Bob": a name, not a verb
        clean = [not p and not s for p, _, s in parts]
        out = []
        n = len(words)

        def swap_word(i: int, new: list[str]) -> list[str]:
            p, _, s = parts[i]
            return new[:-1] + [p + new[-1] + s] if len(new) > 1 else [p + new[0] + s]

        # a role word, swapped
        for i, c in enumerate(core):
            if c in SWAP and i + 1 < n:
                # "to" + a verb is an infinitive ("able to swim"), unless an
                # object comes first ("move it to bob": the tagger reads bob as a verb)
                infinitive = tags[i + 1] == "VB" and not (i and tags[i - 1] in ("PRP", "CD", "NNP"))
                if c == "to" and (infinitive or (i and core[i - 1] in TO_FIXED)):
                    continue
                out.append((f"word:{c}", i, i + 1, swap_word(i, SWAP[c])))
            if c == "out" and i + 2 < n and core[i + 1] == "of" and clean[i]:
                out.append(("word:out of", i, i + 2, swap_word(i + 1, ["into"])))

        # a word swapped for its antonym
        for i, c in enumerate(core):
            if c and c not in SWAP and (a := antonym(c, tags[i])):
                out.append((f"ant:{ANT_POS[tags[i]][0]}", i, i + 1, swap_word(i, [a])))

        # not added or dropped
        for i, c in enumerate(core):
            if c in AUX and i + 1 < n and clean[i] and core[i + 1] not in ("not", "n't"):
                out.append(("not:add", i, i + 1, [words[i], "not"]))
            if c in AUX and i + 1 < n and core[i + 1] == "not" and clean[i]:
                out.append(("not:drop", i, i + 2, swap_word(i + 1, [words[i]])))
            if c in CONTRACTED:
                out.append(("not:drop", i, i + 1, swap_word(i, [CONTRACTED[c]])))

        # two phrases swapped around a relation word or a verb
        def np_back(end: int) -> int | None:
            """Start of the noun phrase ending at `end` (exclusive), or None."""
            s = end
            while s > 0 and end - s < MAX_NP and tags[s - 1] in NP_TAGS and clean[s - 1]:
                if s == 1 and tags[0] not in SURE_START:
                    break           # "Copies ..." / "Moves ...": a description's verb, tagged a noun
                s -= 1
            return s if s < end and any(tags[k] in HEAD_TAGS for k in range(s, end)) else None

        def np_fwd(start: int) -> int | None:
            """End (exclusive) of the noun phrase starting at `start`, or None;
            only its last word may carry punctuation after it."""
            e = start
            while e < n and e - start < MAX_NP and tags[e] in NP_TAGS and not parts[e][0]:
                e += 1
                if parts[e - 1][2]:
                    break
            return e if e > start and any(tags[k] in HEAD_TAGS for k in range(start, e)) else None

        def swapped(a0, a1, mid, b0, b1) -> list[str]:
            a, b = words[a0:a1], [_parts(w)[1] if k == b1 - 1 else w for k, w in enumerate(words[b0:b1], b0)]
            tail = parts[b1 - 1][2]
            new = b + mid + a
            return new[:-1] + [new[-1] + tail]

        for i, c in enumerate(core):
            if c == "from" and clean[i]:
                a1 = np_fwd(i + 1)
                if a1 and a1 < n - 1 and core[a1] == "to" and clean[a1 - 1] and clean[a1]:
                    b1 = np_fwd(a1 + 1)
                    if b1:
                        out.append(("swap:from-to", i + 1, b1, swapped(i + 1, a1, [words[a1]], a1 + 1, b1)))
                        continue
            if (c in REL or tags[i] in VERB_TAGS) and clean[i] and 0 < i < n - 1:
                a0, b1 = np_back(i), np_fwd(i + 1)
                if a0 is not None and b1 and words[a0:i] != words[i + 1:b1]:
                    kind = "swap:verb" if tags[i] in VERB_TAGS and c not in REL else f"swap:{c}"
                    out.append((kind, a0, b1, swapped(a0, i, [words[i]], i + 1, b1)))
        return words, out


def family(kind: str) -> str:
    return kind.split(":")[0]


def make_flips(texts: list[str], seed: int, window_p: float, shuf_p: float = 0.0) -> list[dict]:
    """One flip per text that has one: a family (word / not / swap) chosen
    evenly among those the text offers, then a kind within it (to, for,
    verb, ...), then a site. With probability window_p, and always for a
    text over MAX_WORDS words (where one flipped word barely moves a mean
    over every word), both sides are cut to the same few words around the
    site, as short as the planner's 4-word request chunks. Each text's
    choices are seeded by the text, so a rule change moves only the texts it
    touches (and the teacher's cache keeps the rest)."""
    fl = Flipper()
    out = []
    for text in texts:
        words, sites = fl.sites(text)
        if not sites:
            continue
        rng = random.Random(seed * 1_000_003 + zlib.crc32(text.encode("utf-8")))
        by = defaultdict(lambda: defaultdict(list))
        for s in sites:
            by[family(s[0])][s[0]].append(s)
        fam = by[rng.choice(sorted(by))]
        kind, a, b, new = rng.choice(fam[rng.choice(sorted(fam))])
        src, flip = words, words[:a] + new + words[b:]
        window = False
        if (len(words) > MAX_WORDS or rng.random() < window_p) and len(words) > 4:
            kl, kr = rng.randint(0, 4), rng.randint(0, 4)
            src = words[max(0, a - kl):b + kr]
            flip = words[max(0, a - kl):a] + new + words[b:b + kr]
            window = True
        src, flip = " ".join(src), " ".join(flip)
        if src.lower() != flip.lower():
            out.append({"src": src, "flip": flip, "kind": kind, "window": window})
    if shuf_p:
        # word order alone: a share of texts and a shuffle of their words (a
        # window of 6-12 words of a long one), copied from the teacher
        for text in texts:
            rng = random.Random(seed * 7_000_003 + zlib.crc32(text.encode("utf-8")))
            if rng.random() >= shuf_p:
                continue
            words = text.split()
            if len(words) > MAX_WORDS:
                k = rng.randint(6, 12)
                s0 = rng.randrange(len(words) - k + 1)
                words = words[s0:s0 + k]
            src = " ".join(words)
            if len(words) >= 4 and (flip := shuffled(src, rng)):
                out.append({"src": src, "flip": flip, "kind": "shuf:", "window": len(words) < len(text.split())})
    return out


def cmd_flips(args) -> int:
    rng = random.Random(args.seed)
    texts = [json.loads(line)["text"] for line in GENERAL.open(encoding="utf-8")]
    held = [t for t in texts if held_out(t)]
    train = [t for t in texts if not held_out(t)]
    rng.shuffle(held)
    rng.shuffle(train)
    t0 = time.time()
    rows = [dict(r, split="held") for r in make_flips(held[:args.n_held], args.seed, args.window, args.shuf)]
    rows += [dict(r, split="train") for r in make_flips(train[:args.n_train], args.seed, args.window, args.shuf)]
    ROLE.mkdir(parents=True, exist_ok=True)
    with (ROLE / "flips.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    c = Counter((r["split"], r["kind"]) for r in rows)
    print(f"{len(rows)} flips in {time.time() - t0:.0f}s -> {ROLE / 'flips.jsonl'}")
    for split in ("train", "held"):
        fam = Counter()
        for (s, k), v in c.items():
            if s == split:
                fam[family(k)] += v
        print(f"  {split}: {sum(fam.values())}  " + "  ".join(f"{k} {v}" for k, v in sorted(fam.items())))
    print("  kinds: " + "  ".join(f"{k} {v}" for (s, k), v in sorted(c.items(), key=lambda x: -x[1]) if s == "train"))
    for r in [r for r in rows if r["split"] == "held"][:24]:
        print(f"    [{r['kind']}{' w' if r['window'] else ''}] {r['src']!r} -> {r['flip']!r}")
    return 0


def load_flips(split: str) -> list[dict]:
    return [r for r in map(json.loads, (ROLE / "flips.jsonl").open(encoding="utf-8")) if r["split"] == split]


# ── the teacher ───────────────────────────────────────────────────────────────

class Teacher:
    """all-MiniLM-L6-v2 as sentence-transformers runs it: mean over real
    tokens, unit length, 256 tokens at most."""

    def __init__(self, name: str = TEACHER):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModel, AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).eval()

    @torch.no_grad()
    def embed(self, texts: list[str], bs: int = 256, log_every: int = 0) -> torch.Tensor:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), 384)
        t0 = time.time()
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = self.tk([texts[i] for i in idx], padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
            h = self.m(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[idx] = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1)
            if log_every and (s // bs) % log_every == 0:
                done = s + len(idx)
                print(f"  teacher {done}/{len(texts)}  {done / (time.time() - t0):.0f}/s", flush=True)
        return out


def pair_cos(vec, a: list[str], b: list[str]) -> torch.Tensor:
    va, vb = vec(a), vec(b)
    return (va * vb).sum(1)


def hand_pairs():
    from chunk_probe import CALIBRATION, ROLE_PAIRS
    return ROLE_PAIRS, CALIBRATION


def cmd_teacher_probe(args) -> int:
    """Step 0: the teacher's cosine between a sentence and its flip, next to
    the shipped reader's. If the teacher keeps flips apart, copying it on
    flipped sentences is enough; if not, training needs a push-apart term."""
    from live_reader import LiveReader
    teacher = Teacher()
    shipped = TorchReader(TERN_DIR / "model-int4.bin")
    readers = {"teacher": teacher.embed, "ternlight": shipped.embed}
    rng = random.Random(0)
    held = load_flips("held")
    rng.shuffle(held)
    held = held[:args.n]
    role, calib = hand_pairs()
    report = {}
    print(f"{'':44s} {'teacher':>8s} {'ternlight':>10s}")
    for a, b in role + calib:
        c = {k: float(pair_cos(f, [a], [b])[0]) for k, f in readers.items()}
        report[f"{a} | {b}"] = c
        print(f"  {a[:20]!r:>22s} vs {b[:16]!r:18s} {c['teacher']:8.3f} {c['ternlight']:10.3f}")
    by = defaultdict(list)
    for r in held:
        by[family(r["kind"])].append(r)
    by["all"] = [r for r in held if family(r["kind"]) in ROLE_FAMILIES]
    shuf = [(t, s) for t in (r["src"] for r in by["all"]) if (s := shuffled(t, rng))]
    summary = {}
    for k, v in list(by.items()) + [("shuffled", None)]:
        a, b = ([r["src"] for r in v], [r["flip"] for r in v]) if v else ([p for p, _ in shuf], [s for _, s in shuf])
        row = {}
        for name, f in readers.items():
            c = pair_cos(f, a, b)
            row[name] = {"mean": round(float(c.mean()), 3), "p90": round(float(c.quantile(0.9)), 3),
                         "below_0.85": round(float((c < 0.85).float().mean()), 3)}
        summary[k] = dict(row, n=len(a))
        print(f"  {k:10s} n={len(a):4d}  teacher mean {row['teacher']['mean']:.3f} "
              f"(p90 {row['teacher']['p90']:.3f}, {100 * row['teacher']['below_0.85']:.0f}% < 0.85)   "
              f"ternlight mean {row['ternlight']['mean']:.3f} (p90 {row['ternlight']['p90']:.3f})")
    report["summary"] = summary
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def cmd_teacher(args) -> int:
    """Teacher vectors for every text training reads: the training flips'
    two sides and the held-out checks' texts. Texts already in teacher.pt
    keep their vectors."""
    rows = load_flips("train") + load_flips("held")
    texts = list(dict.fromkeys(t for r in rows for t in (r["src"], r["flip"])))
    old = {}
    if (ROLE / "teacher.pt").exists():
        tt = torch.load(ROLE / "teacher.pt")
        old = dict(zip(tt["texts"], tt["table"]))
    new = [t for t in texts if t not in old]
    t0 = time.time()
    if new:
        old.update(zip(new, Teacher().embed(new, log_every=200).half()))
    table = torch.stack([old[t] for t in texts])
    torch.save({"texts": texts, "table": table}, ROLE / "teacher.pt")
    print(f"{len(texts)} texts ({len(new)} new) in {time.time() - t0:.0f}s -> {ROLE / 'teacher.pt'}")
    return 0


# ── training (step 3) ─────────────────────────────────────────────────────────

class Packed:
    """Every text's token ids in one flat tensor plus offsets."""

    def __init__(self, texts: list[str], tk: Tokenizer):
        ids, lens = [], []
        for s in range(0, len(texts), 50_000):
            for i in tk.ids(texts[s:s + 50_000]):
                ids.extend(i)
                lens.append(len(i))
        self.flat = torch.tensor(ids, dtype=torch.int32)
        self.lens = torch.tensor(lens)
        self.off = torch.cat([torch.zeros(1, dtype=torch.long), self.lens.cumsum(0)])

    def batch(self, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        T = int(self.lens[idx].max())
        t = torch.zeros(len(idx), T, dtype=torch.long)
        for r, i in enumerate(idx):
            t[r, :self.lens[i]] = self.flat[self.off[i]:self.off[i + 1]]
        return t, t != 0


def param_groups(m: TernReader, args) -> list[dict]:
    """Each ternary matrix's training weights move at tern_rel x its median
    |weight|, so flipping a code takes about 0.5 / tern_rel steps of one
    sign whatever the matrix's scale; the rest at their own rates."""
    order = m.order_params()
    groups = [{"params": order, "lr": args.lr_pos, "name": "pos"},
              {"params": [m.emb.weight], "lr": args.lr_emb, "name": "emb"}]
    tern = set()
    for k, mod in enumerate(m.ternary_layers()):
        groups.append({"params": [mod.weight], "lr": args.tern_rel / float(mod.w_scale()),
                       "name": f"tern{k}"})
        tern.add(id(mod.weight))
    skip = tern | {id(p) for p in order} | {id(m.emb.weight)}
    rest = [p for p in m.parameters() if id(p) not in skip]
    groups.append({"params": rest, "lr": args.lr, "name": "rest"})
    for g in groups:
        g["base_lr"] = g["lr"]
    return groups


def losses(m, pk: Packed, T: torch.Tensor, si: list[int], fi: list[int], args,
           S: torch.Tensor | None = None) -> dict:
    """Copy the teacher on both sides; match its within-batch similarities
    except between a sentence and its own flip; push each pair below the
    ceiling. With `S` (the shipped reader's vectors) also keep each original
    sentence near the shipped reader's reading of it: the shipped student
    matches constants to request words better than its teacher does
    (results/logs/role/probe_teacher.json), and copying the teacher alone
    wears that off."""
    B = len(si)
    t, mask = pk.batch(si + fi)
    v = m(t, mask)
    tv = T[si + fi]
    distill = (1 - (v * tv).sum(1)).mean()
    anchor = (1 - (v[:B] * S[si]).sum(1)).mean() if S is not None else torch.zeros(())
    keep = ~torch.eye(2 * B, dtype=torch.bool)
    ar = torch.arange(B)
    keep[ar, ar + B] = False
    keep[ar + B, ar] = False
    rel = ((v @ v.T - tv @ tv.T)[keep] ** 2).mean()
    flip_cos = (v[:B] * v[B:]).sum(1)
    push = F.relu(flip_cos - args.ceiling).mean()
    total = (args.lam_teacher * distill + args.lam_rel * rel + args.lam_push * push
             + args.lam_anchor * anchor)
    return {"total": total, "distill": distill, "rel": rel, "push": push, "anchor": anchor,
            "flip_cos": flip_cos.mean(), "teacher_flip": (tv[:B] * tv[B:]).sum(1).mean()}


@torch.no_grad()
def quick_eval(m, pk: Packed, T: torch.Tensor, held: list[tuple[int, int]], tk: Tokenizer) -> dict:
    """Held-out flips: the pair cosine, and each side's cosine to the teacher;
    the hand role pairs."""
    m.eval()
    si, fi = [a for a, _ in held], [b for _, b in held]
    vs, vf = [], []
    for s in range(0, len(held), 512):
        for idx, out in ((si[s:s + 512], vs), (fi[s:s + 512], vf)):
            t, mask = pk.batch(idx)
            out.append(m(t, mask))
    vs, vf = torch.cat(vs), torch.cat(vf)
    role, _ = hand_pairs()
    t, mask = Tokenizer.batch(tk.ids([a for a, _ in role] + [b for _, b in role]))
    hv = m(t, mask)
    n = len(role)
    m.train()
    return {"held_flip": round(float((vs * vf).sum(1).mean()), 4),
            "held_teacher": round(float(((vs * T[si]).sum(1).mean() + (vf * T[fi]).sum(1).mean()) / 2), 4),
            "hand": [round(float(x), 3) for x in (hv[:n] * hv[n:]).sum(1)]}


def ternary_moved(m: TernReader, start: list[torch.Tensor]) -> float:
    """Share of ternary codes that differ from the shipped reader's."""
    diff = tot = 0
    for mod, q0 in zip(m.ternary_layers(), start):
        q, _ = mod.ternary()
        diff += int((q != q0).sum())
        tot += q.numel()
    return diff / tot


def cmd_train(args) -> int:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tt = torch.load(ROLE / "teacher.pt")
    texts, T = tt["texts"], tt["table"].float()
    index = {t: i for i, t in enumerate(texts)}
    tk = Tokenizer()
    t0 = time.time()
    pk = Packed(texts, tk)
    print(f"{len(texts)} texts tokenized in {time.time() - t0:.0f}s", flush=True)
    train = [(index[r["src"]], index[r["flip"]]) for r in load_flips("train")]
    held = [(index[r["src"]], index[r["flip"]]) for r in load_flips("held")
            if family(r["kind"]) in ROLE_FAMILIES]
    if args.limit:
        train = train[:args.limit]

    S = None
    if args.lam_anchor:
        path = ROLE / "shipped.pt"
        if path.exists() and torch.load(path, mmap=True)["n"] == len(texts):
            S = torch.load(path)["table"].float()
        else:
            t1 = time.time()
            S = TorchReader(TERN_DIR / "model-int4.bin", batch=1024).embed(texts)
            torch.save({"n": len(texts), "table": S.half()}, path)
            print(f"shipped reader on {len(texts)} texts in {time.time() - t1:.0f}s", flush=True)
    m = TernReader.from_bin(TERN_DIR / "model-int4.bin", pos=args.pos, rel_k=args.rel_k)
    m.int4_ste = args.int4_ste
    start = [mod.ternary()[0] for mod in m.ternary_layers()]
    opt = torch.optim.AdamW(param_groups(m, args), weight_decay=0.0)
    steps = args.epochs * (len(train) // args.batch)
    warm = max(1, int(0.03 * steps))
    log = (out / "log.jsonl").open("w", encoding="utf-8")
    ev = quick_eval(m, pk, T, held, tk)
    print(f"step 0: {ev}", flush=True)
    log.write(json.dumps({"step": 0, **ev}) + "\n")
    m.train()
    step, acc, t0 = 0, defaultdict(float), time.time()
    for ep in range(args.epochs):
        rng.shuffle(train)
        for b in range(0, len(train) - args.batch + 1, args.batch):
            chunk = train[b:b + args.batch]
            lr_mult = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_mult
            L = losses(m, pk, T, [a for a, _ in chunk], [c for _, c in chunk], args, S)
            opt.zero_grad(set_to_none=True)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step += 1
            for k, v in L.items():
                acc[k] += float(v.detach())
            if step % args.log_every == 0:
                row = {"step": step, "epoch": ep, **{k: round(v / args.log_every, 4) for k, v in acc.items()},
                       "moved": round(ternary_moved(m, start), 4),
                       "sec": round(time.time() - t0)}
                acc.clear()
                if step % args.eval_every == 0 or step == steps:
                    row.update(quick_eval(m, pk, T, held, tk))
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
    m.quantize_embedding_int4_()
    ev = quick_eval(m, pk, T, held, tk)
    ev["moved"] = round(ternary_moved(m, start), 4)
    print(f"final (int4 embeddings): {ev}", flush=True)
    log.write(json.dumps({"step": step, "final": True, **ev}) + "\n")
    m.save(out / "reader.pt", args=vars(args), final=ev)
    print(f"wrote {out / 'reader.pt'}")
    return 0


# ── the gates (step 4) ────────────────────────────────────────────────────────

def spearman_vs_teacher(vec, teacher_vec: torch.Tensor, texts: list[str], bs: int = 128) -> float:
    """Ternlight's own fidelity metric: within batches of 128, every pair's
    cosine under the reader against the teacher's, rank-correlated."""
    from scipy.stats import spearmanr
    v = vec(texts)
    a, b = [], []
    iu = torch.triu_indices(bs, bs, 1)
    for s in range(0, len(texts) - bs + 1, bs):
        x, y = v[s:s + bs], teacher_vec[s:s + bs]
        a.append((x @ x.T)[iu[0], iu[1]])
        b.append((y @ y.T)[iu[0], iu[1]])
    return float(spearmanr(torch.cat(a).numpy(), torch.cat(b).numpy()).statistic)


def exam_flips(n: int, rng: random.Random) -> list[dict]:
    """Our own requests, flipped the same way: never trained on, only checked."""
    reqs = []
    for name in ("s6_holdout_s500.jsonl", "clt_holdout_s500.jsonl"):
        for line in (ROOT / "data" / name).open(encoding="utf-8"):
            reqs.append(json.loads(line)["request"])
    reqs = list(dict.fromkeys(reqs))
    rng.shuffle(reqs)
    return make_flips(reqs[:n], 0, window_p=0.0)


def cmd_check(args) -> int:
    from chunk_probe import probe
    rng = random.Random(0)
    new = TorchReader(Path(args.model))
    old = TorchReader(TERN_DIR / "model-int4.bin")
    teacher = Teacher()
    readers = {"new": new.embed, "shipped": old.embed}
    rep, gates = {"model": args.model}, {}

    held = load_flips("held")
    by = defaultdict(list)
    for r in held:
        by[family(r["kind"])].append(r)
    by["all"] = [r for r in held if family(r["kind"]) in ROLE_FAMILIES]
    ex = exam_flips(600, rng)
    rep["flips"] = {}
    print(f"flip cosine (mean)            {'new':>7s} {'shipped':>8s} {'teacher':>8s}")
    for k, rows in list(by.items()) + [("our requests", ex)]:
        a, b = [r["src"] for r in rows], [r["flip"] for r in rows]
        c = {name: round(float(pair_cos(f, a, b).mean()), 3) for name, f in
             list(readers.items()) + [("teacher", teacher.embed)]}
        rep["flips"][k] = dict(c, n=len(rows))
        print(f"  {k:14s} n={len(rows):5d}  {c['new']:7.3f} {c['shipped']:8.3f} {c['teacher']:8.3f}")
    role, calib = hand_pairs()
    hand = {}
    print("hand role pairs                new  shipped")
    for a, b in role:
        c = {name: float(pair_cos(f, [a], [b])[0]) for name, f in readers.items()}
        hand[f"{a} | {b}"] = {k: round(v, 3) for k, v in c.items()}
        print(f"  {a!r:30s} {c['new']:.3f}  {c['shipped']:.3f}")
    rep["hand"] = hand
    gates["roles: held flips mean <= 0.85"] = rep["flips"]["all"]["new"] <= 0.85
    gates["roles: every hand pair lower than shipped"] = all(v["new"] < v["shipped"] for v in hand.values())

    srcs = [r["src"] for r in by["all"] if not r["window"] and len(r["src"].split()) >= 4]
    shuf = [(t, s) for t in srcs if (s := shuffled(t, rng))][:1000]
    rep["shuffled"] = {name: round(float(pair_cos(f, [p for p, _ in shuf], [s for _, s in shuf]).mean()), 3)
                       for name, f in readers.items()}
    print(f"shuffled (n={len(shuf)}): new {rep['shuffled']['new']:.3f}  shipped {rep['shuffled']['shipped']:.3f}")
    gates["order: shuffled mean < 0.95"] = rep["shuffled"]["new"] < 0.95

    gen = [t for t in (json.loads(line)["text"] for line in GENERAL.open(encoding="utf-8")) if held_out(t)]
    rng.shuffle(gen)
    gen = gen[:args.n_general]
    tv = teacher.embed(gen)
    rep["spearman"] = {name: round(spearman_vs_teacher(f, tv, gen), 4) for name, f in readers.items()}
    rep["teacher_cos"] = {name: round(float((f(gen) * tv).sum(1).mean()), 4) for name, f in readers.items()}
    print(f"spearman vs teacher (held-out general, n={len(gen)}): new {rep['spearman']['new']:.4f}  "
          f"shipped {rep['spearman']['shipped']:.4f}; cosine to teacher new {rep['teacher_cos']['new']:.4f} "
          f"shipped {rep['teacher_cos']['shipped']:.4f}")
    gates["meaning: spearman within 0.02 of shipped"] = rep["spearman"]["new"] >= rep["spearman"]["shipped"] - 0.02

    cal = {}
    for (a, pos), (_, neg) in zip(calib[0::2], calib[1::2]):
        c = {name: (float(pair_cos(f, [a], [pos])[0]), float(pair_cos(f, [a], [neg])[0])) for name, f in readers.items()}
        cal[f"{a} | {pos} > {neg}"] = {k: [round(x, 3) for x in v] for k, v in c.items()}
        print(f"  calibration {a!r}: {pos[:24]!r} {c['new'][0]:.3f} > {neg[:24]!r} {c['new'][1]:.3f}"
              f"   (shipped {c['shipped'][0]:.3f} > {c['shipped'][1]:.3f})")
    rep["calibration"] = cal
    gates["meaning: calibration pairs keep their order"] = all(v["new"][0] > v["new"][1] for v in cal.values())

    pr = probe(new, [4], quiet=True)
    rep["chunk_probe"] = {s: pr[s]["methods"]["chunk-4"] for s in ("clut", "plain", "demo")}
    print("chunk probe (chunk-4 pair): " + "  ".join(f"{s} {v['pair']}" for s, v in rep["chunk_probe"].items())
          + "   (shipped: clut 87.1, plain 66.6, demo 84.8)")
    gates["meaning: probe clut >= 86"] = rep["chunk_probe"]["clut"]["pair"] >= 86
    gates["meaning: probe plain >= 65.5"] = rep["chunk_probe"]["plain"]["pair"] >= 65.5

    rep["gates"] = gates
    print("\ngates:")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if all(gates.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("flips")
    p.add_argument("--n-train", type=int, default=480_000)
    p.add_argument("--n-held", type=int, default=5_000)
    p.add_argument("--window", type=float, default=0.4)
    p.add_argument("--shuf", type=float, default=0.15,
                   help="share of texts that also give a shuffled-words row")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("teacher-probe")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", default=None)
    sub.add_parser("teacher")
    p = sub.add_parser("train")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64, help="flip pairs per step (twice as many texts)")
    p.add_argument("--lr", type=float, default=1e-4, help="norms, biases, projection")
    p.add_argument("--lr-pos", type=float, default=1e-3)
    p.add_argument("--lr-emb", type=float, default=5e-5)
    p.add_argument("--tern-rel", type=float, default=5e-3,
                   help="ternary training weights: this x the matrix's median |weight| per step")
    p.add_argument("--ceiling", type=float, default=0.8, help="push a sentence and its flip below this cosine")
    p.add_argument("--lam-push", type=float, default=1.0)
    p.add_argument("--lam-rel", type=float, default=1.0)
    p.add_argument("--lam-teacher", type=float, default=1.0)
    p.add_argument("--lam-anchor", type=float, default=0.0,
                   help="keep original sentences near the shipped reader's vectors")
    p.add_argument("--pos", default="abs", choices=["abs", "rel", "nb"],
                   help="word order as a position table (abs, the plan's), as "
                        "per-head attention biases by word offset (rel), or as a "
                        "learned share of each token's neighbours (nb)")
    p.add_argument("--rel-k", type=int, default=8, help="offsets for rel / nb")
    p.add_argument("--int4-ste", action="store_true",
                   help="train on the int4 embedding rows the reader ships with")
    p.add_argument("--limit", type=int, default=None, help="first N training pairs (smoke)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("check")
    p.add_argument("--model", required=True)
    p.add_argument("--n-general", type=int, default=5120)
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    return {"flips": cmd_flips, "teacher-probe": cmd_teacher_probe, "teacher": cmd_teacher,
            "train": cmd_train, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
