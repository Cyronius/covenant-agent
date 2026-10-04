"""C0 caches read by the ELECTRA reader (.claude/plans/electra-only-reader.md, step 2).

From data_cache_c0 (prep.py --reader-lines), two caches in which every reader
vector is ELECTRA's (electra_reader.ElectraReader: the ternary body, head n,
the tagger). Tool descriptions, requests, constants and fields are read
alone; the request's pieces are pooled out of one pass over it.

  data_cache_c0e     t_chunk: the request's 4-word chunks
  data_cache_c0esc   t_chunk: 7 role slots (slot_tagger.TAGGED order, -1 when
                     absent), then the chunks

Both share one reader table: the source table's rows re-read by ELECTRA (the
teacher's texts, then reader_texts.jsonl, in the same order, so t_desc /
t_name / t_req / t_const / t_field still point where they did), then one row
per (request, piece). A piece's vector depends on its request, so its row is
keyed by both: reader_table.py can't rebuild this table.

The GPU work runs on a pod and the 1.3 GB splits never travel:

  python electra_cache.py requests    # laptop: each row's request -> reader/embed/c0_requests.json
  python electra_cache.py build       # GPU: the table and both t_chunk layouts -> reader/embed/c0_patch.pt
  python electra_cache.py apply       # laptop: data_cache_c0 + the patch -> both caches
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

from electra_reader import BODY, EMBED, HEAD, TAGGER, ElectraReader, chunk_spans, embed_in_context, tag_spans
from prep import QUERY_PREFIX
from slot_tagger import TAGGED

HERE = Path(__file__).parent
SPLITS = ("train", "val", "test", "holdout")
REQUESTS = EMBED / "c0_requests.json"
PATCH = EMBED / "c0_patch.pt"
VARIANTS = {"c0e": False, "c0esc": True}          # cache suffix -> role slots before the chunks


def requests_of(src: Path, cfg: dict) -> dict[str, list[str]]:
    """Each split's request per row, as the reader saw it (Lines' request
    line of the serialized context), the way prep.py built the chunks."""
    import pickle
    from corpus import COVENANT  # noqa: F401 -- puts the checkout on sys.path
    from harness.context import TaskContext, serialize_context
    from prep import Lines
    rows = pickle.load(open(src / "rows.pkl", "rb"))
    out = {}
    corpus = None
    for split in SPLITS:
        meta = json.loads((src / f"{split}_meta.json").read_text(encoding="utf-8"))
        if split in rows:
            rs = rows[split]
        else:
            if corpus is None:
                want = {m["task_id"].split("#")[0] for m in meta}
                corpus = {}
                with (HERE.parents[1] / "data" / cfg["corpus"]).open(encoding="utf-8") as fh:
                    for line in fh:
                        r = json.loads(line)
                        if r["id"] in want:
                            corpus[r["id"]] = r
            rs = [corpus[m["task_id"].split("#")[0]] for m in meta]
        reqs = []
        cache: dict[str, str] = {}
        for m, r in zip(meta, rs):
            key = m["task_id"].split("#")[0]
            if key not in cache:
                srcs = serialize_context(r["request"], TaskContext.from_json(r["context"]),
                                         names=bool(cfg.get("names")))
                cache[key] = Lines(srcs, cfg["desc_chars"]).request
            reqs.append(cache[key])
        out[split] = reqs
    return out


def cmd_requests(args) -> int:
    src = HERE / args.src
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    reqs = requests_of(src, cfg)
    Path(args.out).write_text(json.dumps(reqs), encoding="utf-8")
    print(f"{args.out}: " + ", ".join(f"{s} {len(v)}" for s, v in reqs.items())
          + f"; {len({r for v in reqs.values() for r in v})} distinct")
    return 0


def cmd_build(args) -> int:
    src = HERE / args.src
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    reqs = json.loads(Path(args.requests).read_text(encoding="utf-8"))
    texts = [t[len(QUERY_PREFIX):] if t.startswith(QUERY_PREFIX) else t
             for t in torch.load(src / "teacher.pt")["texts"]]
    base = len(texts)
    with (src / "reader_texts.jsonl").open(encoding="utf-8") as fh:
        texts += [json.loads(line) for line in fh]
    if args.limit:                                      # a smoke run: a corner of everything
        texts, reqs = texts[:args.limit], {k: v[:args.limit // 10] for k, v in reqs.items()}
    rd = ElectraReader(args.head, args.body, args.tagger)
    t0 = time.time()
    table = [rd.vectors(texts)]
    print(f"{len(texts)} texts read alone ({base} the teacher's) in {time.time() - t0:.0f}s", flush=True)

    uniq = sorted({r for v in reqs.values() for r in v})
    K, W, cw = len(TAGGED), cfg["max_chunk"], cfg["chunk_words"]
    t0 = time.time()
    tags = tag_spans(rd.tagger, rd.body, uniq)
    spans, kinds = [], []
    for req, tg in zip(uniq, tags):
        present = [k for k, s in enumerate(tg) if s is not None]
        ch = [(a, b) for a, b, _ in chunk_spans(req, cw)[:W]]
        spans.append([tg[k] for k in present] + ch)
        kinds.append((present, len(ch)))
    vecs = embed_in_context(rd.head, rd.body, uniq, spans)
    slot_rows = torch.full((len(uniq), K), -1, dtype=torch.int32)
    chunk_rows = torch.full((len(uniq), W), -1, dtype=torch.int32)
    piece_texts, nxt = [], len(texts)
    for u, (req, (present, nch), v) in enumerate(zip(uniq, kinds, vecs)):
        for j, k in enumerate(present):
            slot_rows[u, k] = nxt + j
        for i in range(nch):
            chunk_rows[u, i] = nxt + len(present) + i
        piece_texts += [req[a:b] for a, b in spans[u]]
        table.append(v)
        nxt += len(spans[u])
    table = torch.cat(table).half()
    print(f"{len(uniq)} requests tagged and read in one pass each, {nxt - len(texts)} pieces, "
          f"in {time.time() - t0:.0f}s; slots present: "
          + "  ".join(f"{r} {100 * float((slot_rows[:, k] >= 0).float().mean()):.0f}%" for k, r in enumerate(TAGGED)),
          flush=True)

    index = {r: u for u, r in enumerate(uniq)}
    t_chunk = {}
    for name, slots in VARIANTS.items():
        t_chunk[name] = {}
        for split, rs in reqs.items():
            ix = torch.tensor([index[r] for r in rs])
            t_chunk[name][split] = torch.cat([slot_rows[ix], chunk_rows[ix]], 1) if slots else chunk_rows[ix]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = {k: Path(v).resolve().relative_to(HERE.resolve()).as_posix() for k, v in rd.files.items()}
    torch.save({"base": base, "table": table, "reader": rd.name, "reader_sha": rd.sha, "files": files,
                "piece_texts": piece_texts, "t_chunk": t_chunk, "src": args.src,
                "max_chunk": {n: (K if s else 0) + W for n, s in VARIANTS.items()}}, out)
    print(f"{out}: table {tuple(table.shape)}, reader {rd.name} sha {rd.sha[:12]}")
    return 0


def cmd_apply(args) -> int:
    src = HERE / args.src
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    patch = torch.load(args.patch)
    with (src / "reader_texts.jsonl").open(encoding="utf-8") as fh:
        texts = [json.loads(line) for line in fh]
    first = None
    for name, slots in VARIANTS.items():
        dst = HERE / f"data_cache_{name}"
        dst.mkdir(parents=True, exist_ok=True)
        skip = {f"{s}.pt" for s in SPLITS} | {"config.json", "reader.pt", "reader_role.pt", "reader_texts.jsonl"}
        for f in src.iterdir():
            if f.name not in skip and not (dst / f.name).exists():
                try:
                    os.link(f, dst / f.name)            # same bytes, no copy
                except OSError:
                    shutil.copy2(f, dst / f.name)
        for split in SPLITS:
            d = torch.load(src / f"{split}.pt")
            tk = patch["t_chunk"][name][split]
            if tk.shape[0] != d["t_chunk"].shape[0]:
                raise SystemExit(f"{name} {split}: {tk.shape[0]} requests for {d['t_chunk'].shape[0]} rows")
            d["t_chunk"] = tk
            torch.save(d, dst / f"{split}.pt")
        if first is None:
            torch.save({k: patch[k] for k in ("base", "table", "reader", "reader_sha")}, dst / "reader.pt")
            first = dst / "reader.pt"
        elif not (dst / "reader.pt").exists():
            try:
                os.link(first, dst / "reader.pt")
            except OSError:
                shutil.copy2(first, dst / "reader.pt")
        with open(dst / "reader_texts.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for t in texts + patch["piece_texts"]:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        c = dict(cfg, max_chunk=patch["max_chunk"][name],
                 electra_reader={**patch["files"], "slots": slots, "chunks": cfg["max_chunk"]},
                 reader_table="electra_cache.py (one row per request piece; reader_table.py can't rebuild it)")
        c.pop("slot_reader", None)
        (dst / "config.json").write_text(json.dumps(c, indent=1), encoding="utf-8", newline="\n")
        print(f"{dst}: t_chunk width {c['max_chunk']}, reader {patch['reader']}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("requests")
    p.add_argument("--src", default="data_cache_c0")
    p.add_argument("--out", default=str(REQUESTS))
    p = sub.add_parser("build")
    p.add_argument("--src", default="data_cache_c0")
    p.add_argument("--requests", default=str(REQUESTS))
    p.add_argument("--head", default=str(HEAD))
    p.add_argument("--body", default=str(BODY))
    p.add_argument("--tagger", default=str(TAGGER))
    p.add_argument("--out", default=str(PATCH))
    p.add_argument("--limit", type=int, default=0, help="smoke: this many texts, a tenth as many requests")
    p = sub.add_parser("apply")
    p.add_argument("--src", default="data_cache_c0")
    p.add_argument("--patch", default=str(PATCH))
    args = ap.parse_args()
    return {"requests": cmd_requests, "build": cmd_build, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
