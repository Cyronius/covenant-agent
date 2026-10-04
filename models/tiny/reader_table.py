"""A frozen reader's table for a cache (.claude/plans/npu-planner.md, phase 2):
every text of the cache's teacher table (teacher.pt, written by
prep.teacher_tables with its texts), each read alone by the ELECTRA reader
(electra_reader.py), in the same order, so the cache's t_desc / t_name / t_req
columns index it unchanged. A cache built with prep.py --reader-lines (C0)
also has reader_texts.jsonl: its constants, fields and request chunks,
appended after the teacher's rows, which is where its t_const / t_field /
t_chunk columns point. Here every chunk is read alone too; electra_cache.py
builds C0 caches whose request pieces come out of one pass over the request.

  python reader_table.py --cache data_cache_s6off [--head reader/embed/n/head.pt] [--name reader.pt]

The table records the reader's sha256 (`reader_sha`), so a live run (play.py
--reader-live) can check it reads with the same reader the planner was
trained on. bge's query instruction on requests is the teacher's convention,
not the reader's, and is stripped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prep import QUERY_PREFIX


def main() -> int:
    from electra_reader import HEAD, ElectraReader
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--head", default=str(HEAD), help="the embedding head (electra_reader.py train)")
    ap.add_argument("--name", default="reader.pt", help="output file in the cache")
    args = ap.parse_args()
    cache = Path(args.cache)
    teacher = torch.load(cache / "teacher.pt")
    if "texts" not in teacher:
        raise SystemExit(f"{cache}/teacher.pt has no texts: rebuild the cache with today's prep.py")
    texts = [t[len(QUERY_PREFIX):] if t.startswith(QUERY_PREFIX) else t for t in teacher["texts"]]
    base = len(texts)
    extra = cache / "reader_texts.jsonl"
    if extra.exists():
        with extra.open(encoding="utf-8") as fh:
            texts += [json.loads(line) for line in fh]
    rd = ElectraReader(args.head)
    table = rd.vectors(texts).half()
    torch.save({"base": base, "table": table, "reader": rd.name, "reader_sha": rd.sha}, cache / args.name)
    print(f"{cache}/{args.name}: {len(texts)} texts ({base} the teacher's, "
          f"{len(texts) - base} reader lines), {rd.name}, norms "
          f"{table.float().norm(dim=-1).min():.3f}-{table.float().norm(dim=-1).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
