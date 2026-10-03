"""A C0 cache whose request reaches the planner as role slots instead of
4-word chunks (.claude/plans/electra-slot-reader.md, step 4).

C0 (prep.py --reader-lines) gives the planner each request chunk's reader
vector through `t_chunk`, with a learned embedding per chunk position. Here
`t_chunk` holds the 7 role slots in role order (action, object, destination,
source, condition, time, amount; -1 when absent), so that embedding becomes
a role embedding and the planner code is unchanged. Each slot is the text the
slot reader tagged (ternary ELECTRA + span tagger, slot_tagger.py); the
shipped Ternlight reads it, through the cache's reader table as before. With
--keep-chunks the chunks follow the 7 slots.

Everything else is the source cache's, byte for byte: the rows, targets,
constants, fields and the teacher's texts. The new span texts are appended
after the source's reader texts, so every old index still points where it
did.

  python slot_cache.py --src data_cache_c0 --dst data_cache_c0s \
      --body tern-electra:reader/tern_electra/e1/model.pt --tagger reader/slots/t_tern_e1/tagger.pt
  python reader_table.py --cache data_cache_c0s --ternlight reader
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import torch

from corpus import COVENANT  # noqa: F401 -- puts the checkout on sys.path
from harness.context import TaskContext, serialize_context
from prep import Lines, chunk_request

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SPLITS = ("train", "val", "test", "holdout")


class SlotTexts:
    """Request -> the 7 role slots' texts (None when absent), by the slot
    reader's body and tagger."""

    def __init__(self, body: str, tagger: str):
        from slot_reader import make_body
        from slot_tagger import SpanTagger, Tok
        ck = torch.load(tagger)
        self.tagger = SpanTagger(**ck["cfg"])
        self.tagger.load_state_dict(ck["state"])
        self.body = make_body(body)
        self.tok = Tok()
        self.spec = {"body": body, "tagger": tagger}

    def __call__(self, requests: list[str]) -> list[list[str | None]]:
        from slot_tagger import tag
        out = []
        for s in range(0, len(requests), 256):
            chunk = requests[s:s + 256]
            for text, spans in zip(chunk, tag(self.tagger, self.body, self.tok, chunk)):
                out.append([text[sp[0]:sp[1]] if sp is not None else None for sp in spans])
        return out


def requests_of(src: Path, cfg: dict) -> dict[str, list[str]]:
    """Each split's request per row, as the reader saw it (Lines' request
    line of the serialized context), the way prep.py built the chunks."""
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
                with (ROOT / "data" / cfg["corpus"]).open(encoding="utf-8") as fh:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_cache_c0")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--tagger", required=True)
    ap.add_argument("--keep-chunks", action="store_true")
    args = ap.parse_args()
    src, dst = HERE / args.src, HERE / args.dst
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("max_chunk"):
        raise SystemExit(f"{src} is not a C0 cache (prep.py --reader-lines)")
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name in {f"{s}.pt" for s in SPLITS} | {"config.json", "reader_texts.jsonl", "reader.pt",
                                                   "reader_role.pt"}:
            continue
        target = dst / f.name
        if not target.exists():
            try:
                os.link(f, target)                  # same bytes, no copy
            except OSError:
                shutil.copy2(f, target)

    reqs = requests_of(src, cfg)
    uniq = sorted({r for rs in reqs.values() for r in rs})
    slots = dict(zip(uniq, SlotTexts(args.body, args.tagger)(uniq)))
    texts = [json.loads(line) for line in (src / "reader_texts.jsonl").open(encoding="utf-8")]
    n_old = len(texts)
    base = torch.load(src / "reader.pt", mmap=True)["base"]
    index = {t: base + i for i, t in enumerate(texts)}

    def tid(t: str) -> int:
        if t not in index:
            index[t] = base + len(texts)
            texts.append(t)
        return index[t]

    n_roles = 7
    width = n_roles + (cfg["max_chunk"] if args.keep_chunks else 0)
    present = [0] * n_roles
    for split in SPLITS:
        d = torch.load(src / f"{split}.pt")
        tk = torch.full((len(reqs[split]), width), -1, dtype=torch.int32)
        for n, req in enumerate(reqs[split]):
            for k, s in enumerate(slots[req]):
                if s:
                    tk[n, k] = tid(s)
                    present[k] += 1
            if args.keep_chunks:
                for i, c in enumerate(chunk_request(req, cfg["chunk_words"])[:cfg["max_chunk"]]):
                    tk[n, n_roles + i] = tid(c)
        if len(reqs[split]) != d["t_chunk"].shape[0]:
            raise SystemExit(f"{split}: {len(reqs[split])} requests for {d['t_chunk'].shape[0]} rows")
        d["t_chunk"] = tk
        torch.save(d, dst / f"{split}.pt")
        print(f"  {split}: {len(reqs[split])} rows", flush=True)
    with open(dst / "reader_texts.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for t in texts:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    cfg.update({"max_chunk": width, "slot_reader": {"body": args.body, "tagger": args.tagger,
                                                    "keep_chunks": args.keep_chunks}})
    (dst / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8", newline="\n")
    total = sum(len(v) for v in reqs.values())
    print(f"{dst}: {len(uniq)} requests tagged; slots present per row: "
          + "  ".join(f"{r} {100 * p / total:.0f}%" for r, p in
                      zip(("action", "object", "destination", "source", "condition", "time", "amount"), present))
          + f"; reader texts {len(texts)} ({len(texts) - n_old} new span texts)")
    print(f"next: python reader_table.py --cache {args.dst} --ternlight reader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
