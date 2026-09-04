"""Vocabulary prune for the tuned planner (plan s2-consolidated-program §B4;
ship-model-track S2 rung).

The tokenizer has ~248k entries; the whole S1 corpus plus every eval prompt
uses ~7.8k. This drops the rest — minus a general-English floor — from BOTH
the tokenizer and the (tied) embedding matrix, without retraining, so the
result must be output-identical on the corpus distribution. That identity is
the rung's pass condition; only after it holds does the trim ride into the
S2 retrain.

Keep-set = tokens seen in (SFT corpus prompts+programs) ∪ (eval-suite
prompts) ∪ (real-request turns, if pulled) ∪ the 256 byte-level bases ∪
special/added tokens ∪ the result tokens of the first --floor merges (merge
rank ≈ pretraining frequency, i.e. ordinary English), then closed under
merge parents so every kept token is still reachable by BPE. Merges whose
result (or either side) was dropped are removed; texts made only of kept
tokens tokenize identically, texts with dropped tokens fall back to smaller
kept pieces (never to <unk>).

  python -m baselines.qwen.prune_vocab --src baselines/qwen/models/merged_s1_hf \
      --out baselines/qwen/models/merged_s1_pruned_hf --floor 40000
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TEXT_SOURCES = [
    ("data/sft_s1.jsonl", "messages"),
    ("data/sft_s2.jsonl", "messages"),
    ("data/holdout/e_real_sessions.jsonl", "input_text"),
    ("data/holdout/e_foreign.jsonl", "input_text"),
    ("data/holdout/e_crowded.jsonl", "input_text"),
    ("data/holdout/e_ood_english.jsonl", "input_text"),
    ("data/holdout/e_demo_requests.jsonl", "input_text"),
    ("data/r1_tasks.jsonl", "input_text"),
    ("data/real_sessions/turns.jsonl", "user_text"),
]


def resolve_data(rel: str) -> Path:
    """Resolve a repo-relative data path against ROOT (dev-machine layout);
    fall back to its basename in the cwd (a rented pod's upload is flattened,
    so ROOT does not point at a real repo there)."""
    p = ROOT / rel
    if p.exists():
        return p
    return Path(Path(rel).name)


def used_token_ids(tok) -> set:
    used = set()
    for rel, field in TEXT_SOURCES:
        p = ROOT / rel
        if not p.exists():
            print(f"  (skip {rel}: missing)")
            continue
        n = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                texts = ([m["content"] for m in r["messages"]] if field == "messages"
                         else [r.get(field, "")])
                for t in texts:
                    used.update(tok(t, add_special_tokens=False)["input_ids"])
                n += 1
        print(f"  {rel}: {n} rows, used so far {len(used)}")
    return used


def merge_parts(m):
    if isinstance(m, list):
        return m[0], m[1]
    a, b = m.split(" ", 1)
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser(prog="baselines.qwen.prune_vocab")
    ap.add_argument("--src", required=True, help="merged HF model dir (tokenizer + weights)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=int, default=40000,
                    help="keep the result tokens of the first N merges (general English)")
    ap.add_argument("--pad-to", type=int, default=64, help="pad the embedding rows to a multiple")
    ap.add_argument("--used-ids", default=None, metavar="JSON",
                    help="precomputed used-token-id list (from --dump-used on the dev "
                         "machine) instead of reading TEXT_SOURCES; lets a rented pod "
                         "prune without receiving any request text")
    ap.add_argument("--dump-used", default=None, metavar="JSON",
                    help="write the used-token-id list here and exit")
    ap.add_argument("--check-corpus", default="data/sft_s1.jsonl",
                    help="SFT corpus to verify tokenization identity against "
                         "after pruning (repo-relative); pass the corpus this "
                         "keep-set was actually built from, e.g. data/sft_s2.jsonl")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(src))
    tj = json.load(open(src / "tokenizer.json", encoding="utf-8"))
    model = tj["model"]
    vocab: dict = model["vocab"]                 # token string -> old id
    inv = {i: t for t, i in vocab.items()}
    merges = model["merges"]
    added = tj.get("added_tokens", [])
    print(f"tokenizer: {len(vocab)} vocab, {len(merges)} merges, {len(added)} added")

    if args.used_ids:
        keep = set(json.load(open(args.used_ids, encoding="utf-8")))
        print(f"used tokens from {args.used_ids}: {len(keep)}")
    else:
        print("collecting used tokens:")
        keep = used_token_ids(tok)
        if args.dump_used:
            json.dump(sorted(keep), open(args.dump_used, "w", encoding="utf-8"))
            print(f"wrote {len(keep)} used token ids -> {args.dump_used}")
            return
    n_used = len(keep)
    keep |= set(range(256))                       # byte-level bases (ids 0..255 in this vocab)
    keep |= {a["id"] for a in added}
    keep |= set(tok.all_special_ids)
    floor_ids = set()
    for m in merges[:args.floor]:
        a, b = merge_parts(m)
        r = vocab.get(a + b)
        if r is not None:
            floor_ids.add(r)
    keep |= floor_ids
    # close under merge parents: a kept token must stay reachable
    parents = {}
    for m in merges:
        a, b = merge_parts(m)
        r = vocab.get(a + b)
        if r is not None:
            parents[r] = (vocab[a], vocab[b])
    stack = list(keep)
    while stack:
        t = stack.pop()
        for p in parents.get(t, ()):
            if p not in keep:
                keep.add(p)
                stack.append(p)
    print(f"keep-set: used {n_used} + floor {len(floor_ids)} + bases/specials, closed -> {len(keep)} "
          f"of {len(vocab) + len([a for a in added if a['id'] not in vocab])}")

    # new ids: kept old ids in ascending order
    old_ids = sorted(i for i in keep if i in inv or any(a["id"] == i for a in added))
    remap = {o: n for n, o in enumerate(old_ids)}
    new_vocab = {inv[o]: remap[o] for o in old_ids if o in inv}
    new_merges = []
    for m in merges:
        a, b = merge_parts(m)
        r = vocab.get(a + b)
        if r in keep and vocab.get(a) in keep and vocab.get(b) in keep:
            new_merges.append(m)
    new_added = []
    for a in added:
        if a["id"] in remap:
            a2 = dict(a)
            a2["id"] = remap[a["id"]]
            new_added.append(a2)
            new_vocab.setdefault(a["content"], remap[a["id"]])
    tj["model"]["vocab"] = new_vocab
    tj["model"]["merges"] = new_merges
    tj["added_tokens"] = new_added
    json.dump(tj, open(out / "tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)
    n_vocab = len(old_ids)
    print(f"new tokenizer: {n_vocab} vocab, {len(new_merges)} merges")

    tc = json.load(open(src / "tokenizer_config.json", encoding="utf-8"))
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {str(remap[int(k)]): v for k, v in tc["added_tokens_decoder"].items()
                                      if int(k) in remap}
    json.dump(tc, open(out / "tokenizer_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for name in ("chat_template.jinja",):
        if (src / name).exists():
            (out / name).write_bytes((src / name).read_bytes())

    # weights: slice the tied embedding
    padded = ((n_vocab + args.pad_to - 1) // args.pad_to) * args.pad_to
    for shard in glob.glob(str(src / "*.safetensors")):
        tensors = load_file(shard)
        for k, v in list(tensors.items()):
            if k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight"):
                new = torch.zeros((padded, v.shape[1]), dtype=v.dtype)
                new[:n_vocab] = v[torch.tensor(old_ids)]
                print(f"  {k}: {tuple(v.shape)} -> {tuple(new.shape)}")
                tensors[k] = new
        save_file(tensors, str(out / Path(shard).name), metadata={"format": "pt"})

    cfg = json.load(open(src / "config.json", encoding="utf-8"))
    def remap_id(x):
        if isinstance(x, list):
            return [remap[i] for i in x if i in remap]
        return remap.get(x, x) if isinstance(x, int) else x
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in cfg:
            cfg[key] = remap_id(cfg[key])
    cfg["vocab_size"] = padded
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"]["vocab_size"] = padded
    cfg["_pruned_vocab"] = {"kept": n_vocab, "padded": padded, "floor": args.floor,
                            "source": str(src)}
    json.dump(cfg, open(out / "config.json", "w", encoding="utf-8"), indent=2)
    if (src / "generation_config.json").exists():
        g = json.load(open(src / "generation_config.json", encoding="utf-8"))
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if key in g:
                g[key] = remap_id(g[key])
        json.dump(g, open(out / "generation_config.json", "w", encoding="utf-8"), indent=2)
    json.dump({str(o): n for o, n in remap.items()}, open(out / "vocab_remap.json", "w"))
    print(f"wrote {out} (vocab {n_vocab}, embedding rows {padded})")

    # self-check: corpus tokenization identical under the remap
    tok2 = AutoTokenizer.from_pretrained(str(out))
    bad = 0
    checked = 0
    dropped: set = set()          # corpus tokens the keep-set does not cover
    with open(resolve_data(args.check_corpus), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            for m in json.loads(line)["messages"]:
                ids = tok(m["content"], add_special_tokens=False)["input_ids"]
                missing = [t for t in ids if t not in remap]
                checked += 1
                if missing:
                    # A stale --used-ids: this corpus uses tokens the keep-set
                    # was not built from, so they fall back to smaller pieces.
                    # Report it instead of dying on KeyError three hours into
                    # a run (hit 2026-09-04, S2R against the S2 keep-set).
                    dropped.update(missing)
                    bad += 1
                    continue
                if [remap[t] for t in ids] != tok2(m["content"], add_special_tokens=False)["input_ids"]:
                    bad += 1
    print(f"tokenization identity on 2000 corpus rows: {checked - bad}/{checked} identical")
    if dropped:
        ex = ", ".join(repr(tok.decode([t])) for t in sorted(dropped)[:8])
        print(f"  {len(dropped)} corpus token ids are outside the keep-set, e.g. {ex}")
        sys.exit("FAIL: --used-ids predates this corpus; rebuild the keep-set "
                 "to include it (union the old set with the corpus's tokens)")
    if bad:
        sys.exit("FAIL: pruned tokenizer diverges on the corpus")


if __name__ == "__main__":
    main()
