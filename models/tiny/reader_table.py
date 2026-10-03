"""A frozen external reader's table for a cache (.claude/plans/npu-planner.md,
phase 2): every text of the cache's teacher table (teacher.pt, written by
prep.teacher_tables with its texts), embedded by the reader, in the same
order, so the cache's t_desc / t_name / t_req columns index it unchanged.
A cache built with prep.py --reader-lines (C0) also has reader_texts.jsonl:
its constants, fields and request chunks, appended after the teacher's rows,
which is where its t_const / t_field / t_chunk columns point.

  python reader_table.py --cache data_cache_s6off --ternlight <dir with node_modules/@ternlight>
      [--tier mini]                       # -> <cache>/reader.pt {"table", "reader"}
  python reader_table.py --cache data_cache_c0 --reader-model reader/role/r1/reader.pt
      [--name reader_role.pt]             # a TernReader in PyTorch (tern_reader.py)

A table built from a --reader-model also records the model file's sha256
(`reader_sha`), so a live run (play.py --reader-live) can check it reads with
the same reader the planner was trained on.

bge's query instruction on requests is the teacher's convention, not the
reader's, and is stripped. Ternlight runs as its npm package under node
(results/logs/reader_bakeoff_embed.js is the scoring twin of this).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from prep import QUERY_PREFIX

NODE = r"""
const fs = require('fs'), path = require('path');
const [dir, tier, src, dst] = process.argv.slice(2);
const mod = require(path.join(dir, 'node_modules', '@ternlight', tier));
const texts = JSON.parse(fs.readFileSync(src, 'utf8'));
const fd = fs.openSync(dst, 'w');
const buf = Buffer.alloc(384 * 4);
texts.forEach((t, i) => {
  const v = mod.embed(t);
  for (let j = 0; j < 384; j++) buf.writeFloatLE(v[j], j * 4);
  fs.writeSync(fd, buf);
  if ((i + 1) % 100000 === 0) console.error(`  ${i + 1} / ${texts.length}`);
});
fs.closeSync(fd);
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ternlight", metavar="DIR")
    src.add_argument("--reader-model", metavar="FILE",
                     help="a TernReader (tern_reader.py): a trained .pt or the shipped .bin")
    ap.add_argument("--tier", default="mini", choices=["mini", "base"])
    ap.add_argument("--name", default=None,
                    help="output file in the cache (default reader.pt; reader_role.pt "
                         "for a --reader-model)")
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
    out = {"base": base}
    if args.reader_model:
        from tern_reader import TorchReader
        rd = TorchReader(Path(args.reader_model))
        table = rd.embed(texts).half()
        out.update(reader=rd.name, reader_sha=rd.sha)
        name = args.name or "reader_role.pt"
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "texts.json").write_text(json.dumps(texts), encoding="utf-8")
            (tmp / "embed.js").write_text(NODE, encoding="utf-8")
            subprocess.run(["node", "--max-old-space-size=8192", str(tmp / "embed.js"),
                            str(Path(args.ternlight).resolve()), args.tier,
                            str(tmp / "texts.json"), str(tmp / "vec.bin")], check=True)
            vec = np.fromfile(tmp / "vec.bin", dtype="<f4").reshape(len(texts), 384)
        table = torch.from_numpy(vec).half()
        out["reader"] = f"ternlight-{args.tier}"
        name = args.name or "reader.pt"
    torch.save(dict(out, table=table), cache / name)
    print(f"{cache}/{name}: {len(texts)} texts ({base} the teacher's, "
          f"{len(texts) - base} reader lines), {out['reader']}, norms "
          f"{table.float().norm(dim=-1).min():.3f}-{table.float().norm(dim=-1).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
