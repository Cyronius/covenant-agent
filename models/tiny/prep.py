"""Tokenize once, save tensors. The pod loads these and never touches the corpus.

Keeping preparation separate means the training machine needs neither the
covenant-agent checkout nor the corpus, only this cache plus the two vocabularies.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import torch

from corpus import COVENANT, Example, load, program_tokens, split
from tok import OutVocab, train_input_tokenizer

CACHE = Path(__file__).parent / "data_cache"


def compact(src: str, desc_chars: int = 60) -> str:
    """Trim each schema line's description to its first sentence, then to a cap.

    The signature part of a tool line carries the types and symbols the program
    must get right; the description is what generalization to an unseen tool
    rests on. Trimming the tail of long descriptions buys sequence length
    without touching either.
    """
    out = []
    for line in src.splitlines():
        if " :: " in line:
            head, desc = line.split(" :: ", 1)
            desc = desc.split(". ")[0]
            if len(desc) > desc_chars:
                desc = desc[:desc_chars].rsplit(" ", 1)[0]
            line = f"{head} :: {desc}"
        out.append(line)
    return "\n".join(out) + "\n"


SYM_LINE = re.compile(r"^([TFCSNBDI]\d+)\b")


def symbol_positions(source: str, enc, ov: OutVocab, max_in: int) -> list[int]:
    """For each output-vocabulary id, where that symbol is declared in the input.

    Returns one input-token index per output vocabulary entry, or -1 when the
    symbol is not a per-request symbol or does not appear in this task.

    This exists because symbol identity is assigned per request. `T2` is a
    different tool in every task, so there is nothing stable for an embedding to
    learn -- the first run scored 8.6% compile with 100% parse for exactly this
    reason. What the task actually asks is "read the descriptions and name the
    matching one", which is a pointer, and a pointer needs to know where each
    candidate lives in the input.
    """
    pos = [-1] * len(ov)
    # Character offset of the start of each declaration line.
    off = 0
    starts: dict[str, int] = {}
    for line in source.splitlines(keepends=True):
        m = SYM_LINE.match(line)
        if m and m.group(1) in ov.stoi:
            starts.setdefault(m.group(1), off)
        off += len(line)
    if not starts:
        return pos
    # Map character offsets to token indices via the tokenizer's offsets.
    by_char: dict[int, int] = {}
    for ti, (a, _b) in enumerate(enc.offsets[:max_in]):
        by_char.setdefault(a, ti)
    for sym, cstart in starts.items():
        # The symbol may not start exactly at a token boundary; take the first
        # token whose span begins at or after the line start.
        ti = by_char.get(cstart)
        if ti is None:
            cands = [t for c, t in by_char.items() if c >= cstart]
            ti = min(cands) if cands else None
        if ti is not None and ti < max_in:
            pos[ov.stoi[sym]] = ti
    return pos


def encode_split(examples: list[Example], tk, ov: OutVocab, max_in: int, canvas: int):
    srcs = [compact(e.source) for e in examples]
    encs = tk.encode_batch(srcs)
    src = torch.tensor([e.ids[:max_in] + [tk.token_to_id("<pad>")] * max(0, max_in - len(e.ids))
                        for e in encs], dtype=torch.long)
    pad = src == tk.token_to_id("<pad>")
    tgt = torch.tensor([ov.encode(program_tokens(e.target), canvas) for e in examples],
                       dtype=torch.long)
    sym = torch.tensor([symbol_positions(s, e, ov, max_in) for s, e in zip(srcs, encs)],
                       dtype=torch.long)
    meta = [{"task_id": e.task_id, "level": e.level, "world": e.world} for e in examples]
    return {"src": src, "pad": pad, "tgt": tgt, "sym": sym, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(COVENANT / "data" / "s5_plain.jsonl"))
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--in-vocab", type=int, default=4096)
    ap.add_argument("--max-in", type=int, default=1280)
    ap.add_argument("--canvas", type=int, default=64)
    ap.add_argument("--holdout-world", default=None,
                    help="world name to hold out entirely; default picks a mid-sized one")
    ap.add_argument("--out", default=str(CACHE))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.limit} from {Path(args.corpus).name} ...")
    ex = load(Path(args.corpus), limit=args.limit)
    print(f"  {len(ex)} single-segment examples, {len(set(e.world for e in ex))} worlds")

    if args.holdout_world is None:
        counts = {}
        for e in ex:
            counts[e.world] = counts.get(e.world, 0) + 1
        mid = sorted(counts.items(), key=lambda kv: -kv[1])
        args.holdout_world = mid[len(mid) // 2][0]
    print(f"  holding out world: {args.holdout_world}")

    tr, va, te, ho = split(ex, seed=0, holdout_world=args.holdout_world)
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}  holdout {len(ho)}")

    ov = OutVocab.build(ex)
    ov.save(out / "out_vocab.json")
    print(f"  output vocab {len(ov)}")

    print("training input tokenizer ...")
    tk = train_input_tokenizer([compact(e.source) for e in tr[:8000]],
                               args.in_vocab, max_length=args.max_in)
    tk.save(str(out / "in_tok.json"))

    for name, part in (("train", tr), ("val", va), ("test", te), ("holdout", ho)):
        if not part:
            continue
        d = encode_split(part, tk, ov, args.max_in, args.canvas)
        torch.save({k: v for k, v in d.items() if k != "meta"}, out / f"{name}.pt")
        (out / f"{name}_meta.json").write_text(json.dumps(d["meta"]), encoding="utf-8")
        lens = (d["src"] != tk.token_to_id("<pad>")).sum(1)
        n_cut = int((lens >= args.max_in).sum())
        print(f"  {name:8s} {len(part):6d}  longest input {int(lens.max())}/{args.max_in}"
              + (f"  TRUNCATED {n_cut}" if n_cut else ""))
        # A truncated input loses its CONSTANTS section, which the program
        # references by symbol. That is silent corruption, not a small loss of
        # context, so it fails here rather than showing up as a model error.
        if n_cut:
            raise SystemExit(
                f"{n_cut} {name} inputs hit the {args.max_in}-token cap. "
                f"Raise --max-in or drop the worlds with the longest schemas.")
        overflow = int((d["tgt"][:, -1] != ov.pad).sum())
        if overflow:
            raise SystemExit(f"{overflow} {name} programs fill the whole "
                             f"{args.canvas}-slot canvas; raise --canvas.")

    # The raw rows are what the scorer replays in the sandbox, kept separately
    # so the pod never needs them.
    with open(out / "rows.pkl", "wb") as fh:
        pickle.dump({n: [e.row for e in p]
                     for n, p in (("val", va), ("test", te), ("holdout", ho)) if p}, fh)

    (out / "config.json").write_text(json.dumps({
        "corpus": Path(args.corpus).name, "limit": args.limit,
        "in_vocab": tk.get_vocab_size(), "out_vocab": len(ov),
        "max_in": args.max_in, "canvas": args.canvas,
        "holdout_world": args.holdout_world,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
