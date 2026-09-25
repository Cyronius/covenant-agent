"""General English for the reading stages (.claude/plans/vocab-push.md,
parts B and C), tokenized for one cache.

  python general_prep.py --cache data_cache_s6g --general ../../data/general

Reads data/general (built by `python -m data.gen.general_corpus`):
  texts.jsonl     sentences: WordNet definitions and examples, Google and AWS
                  API documentation, open_pairs tool descriptions
  names.jsonl     identifiers and words, read the way the name stage reads
                  tool names
  siblings.jsonl  WordNet families: words under one parent (phone, email,
                  letter), each with its own definition

and writes <cache>/general.pt: every text in the cache's tokenizer beside the
teacher's vector for it, and the look-alike items (a word, or a sentence
using it, against its siblings' definitions; the answer is WordNet's). The
teacher's vectors do not depend on the tokenizer, so they are kept once in
<general>/teacher_<model>.pt and reused by every cache.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from prep import QUERY_PREFIX, teacher_name

K_MAX = 12          # siblings per family (the corpus builder caps it too)
QUERY_TOKENS = 48   # a word or one example sentence


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def teacher_vectors(texts: list[str], model: str, store: Path, device) -> tuple:
    """Unit-length CLS vectors (the way prep.teacher_tables embeds), cached
    by text in `store`; returns (text -> row, table)."""
    have = torch.load(store) if store.exists() else {"texts": [], "vecs": torch.zeros(0, 384).half()}
    index = {t: i for i, t in enumerate(have["texts"])}
    todo = sorted({t for t in texts if t not in index})
    if todo:
        from transformers import AutoModel, AutoTokenizer
        print(f"  teacher: embedding {len(todo)} new texts with {model} on {device} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(model)
        enc = AutoModel.from_pretrained(model).eval().to(device)
        rows = []
        bs = 512 if device.type == "cuda" else 128
        with torch.no_grad():
            for i in range(0, len(todo), bs):
                b = tok(todo[i:i + bs], padding=True, truncation=True, max_length=128,
                        return_tensors="pt").to(device)
                rows.append(torch.nn.functional.normalize(
                    enc(**b).last_hidden_state[:, 0], dim=-1).half().cpu())
                if (i // bs) % 200 == 0:
                    print(f"    {i + len(b['input_ids'])}/{len(todo)}", flush=True)
        have["vecs"] = torch.cat([have["vecs"]] + rows)
        have["texts"] = have["texts"] + todo
        torch.save(have, store)
        index = {t: i for i, t in enumerate(have["texts"])}
    return index, have["vecs"]


def tokens(tk: Tokenizer, texts: list[str], cap: int, pad: int) -> torch.Tensor:
    """(N, cap) int16, cut at `cap`: this is pretraining text, not a line a
    program has to bind, so a long one losing its tail costs nothing."""
    out = torch.full((len(texts), cap), pad, dtype=torch.int16)
    for i, e in enumerate(tk.encode_batch(texts)):
        ids = e.ids[:cap]
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--general", default=str(Path(__file__).parents[2] / "data" / "general"))
    ap.add_argument("--teacher", default="unsloth/bge-small-en-v1.5")
    ap.add_argument("--max-texts", type=int, default=400_000)
    ap.add_argument("--max-names", type=int, default=200_000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache, gdir = Path(args.cache), Path(args.general)
    meta = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    tk = Tokenizer.from_file(str(cache / "in_tok.json"))
    tk.no_truncation(); tk.no_padding()
    pad, DL, NL = meta["in_pad"], meta["max_desc"], meta["max_name"]
    rng = random.Random(0)

    texts = [r["text"] for r in read_jsonl(gdir / "texts.jsonl")]
    if len(texts) > args.max_texts:
        texts = rng.sample(texts, args.max_texts)
    # the name stage reads what split_tool_line hands it; the teacher always
    # reads words, as prep.teacher_tables does for tool names
    raw_names = [r["text"] for r in read_jsonl(gdir / "names.jsonl")]
    if len(raw_names) > args.max_names:
        raw_names = rng.sample(raw_names, args.max_names)
    stage_names = [teacher_name(n) if meta.get("name_words") else n for n in raw_names]
    teacher_names = [teacher_name(n) for n in raw_names]

    glosses: dict[str, int] = {}
    q_text, q_members, q_label, q_held = [], [], [], []
    for fam in read_jsonl(gdir / "siblings.jsonl"):
        mem = fam["members"][:K_MAX]
        gids = [glosses.setdefault(m["gloss"], len(glosses)) for m in mem]
        held = fam["split"] == "heldout"
        for j, m in enumerate(mem):
            for q in [m["lemma"]] + list(m.get("examples") or [])[:2]:
                q_text.append(q)
                q_members.append(gids + [-1] * (K_MAX - len(gids)))
                q_label.append(j)
                q_held.append(held)
    gloss_list = sorted(glosses, key=glosses.get)
    print(f"  {len(texts)} texts, {len(raw_names)} names, {len(gloss_list)} definitions, "
          f"{len(q_text)} look-alike items ({sum(q_held)} held out)", flush=True)

    store = gdir / f"teacher_{args.teacher.replace('/', '_')}.pt"
    t_queries = [QUERY_PREFIX + q for q in q_text]
    index, table = teacher_vectors(texts + teacher_names + gloss_list + t_queries,
                                   args.teacher, store, dev)
    vec = lambda ts: table[torch.tensor([index[t] for t in ts], dtype=torch.long)]  # noqa: E731

    out = {
        "text_tok": tokens(tk, texts, DL, pad), "text_t": vec(texts),
        "name_tok": tokens(tk, stage_names, NL, pad), "name_t": vec(teacher_names),
        "gloss_tok": tokens(tk, gloss_list, DL, pad), "gloss_t": vec(gloss_list),
        "sib_q_tok": tokens(tk, q_text, QUERY_TOKENS, pad), "sib_q_t": vec(t_queries),
        "sib_members": torch.tensor(q_members, dtype=torch.int32),
        "sib_label": torch.tensor(q_label, dtype=torch.long),
        "sib_heldout": torch.tensor(q_held, dtype=torch.bool),
        "teacher": args.teacher,
    }
    torch.save(out, cache / "general.pt")
    frag = sum(len(e.ids) for e in tk.encode_batch(texts[:20000])) / \
        max(1, sum(len(t.split()) for t in texts[:20000]))
    print(f"wrote {cache / 'general.pt'}; {frag:.2f} pieces per word on general text",
          flush=True)


if __name__ == "__main__":
    main()
