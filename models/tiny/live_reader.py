"""Ternlight run live, for contexts no cache has seen (the demo app, play.py):
.claude/plans/tiny-planner-demo.md step 2.

A cache's reader.pt holds precomputed vectors indexed by t_desc / t_req
(reader_table.py). A demo request and its world's tools are new text, so
here a persistent node process embeds them on demand. Descriptions are
cached by text, so a world's tools are read once per process.

`open_reader(spec)` also takes a TernReader file (tern_reader.py, the
role-aware reader of .claude/plans/role-aware-reader.md), run in PyTorch.

`reader_inputs(source, dims, reader)` pulls the same texts prep's teacher
table does (each tool line's description as split_tool_line cuts it, and the
request) and returns the `rd_tool` / `rd_req` inputs model.reader_vectors
takes in place of a table lookup.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from prep import Lines, chunk_request, reader_text, split_tool_line

NODE = r"""
const path = require('path'), readline = require('readline');
const mod = require(path.join(process.argv[2], 'node_modules', '@ternlight', process.argv[3]));
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  const texts = JSON.parse(line);
  process.stdout.write(JSON.stringify(texts.map((t) => Array.from(mod.embed(t)))) + '\n');
});
"""


class LiveReader:
    def __init__(self, ternlight_dir: str | Path, tier: str = "mini"):
        import tempfile
        script = Path(tempfile.gettempdir()) / "covenant_live_reader.js"
        script.write_text(NODE, encoding="utf-8")
        self.proc = subprocess.Popen(
            ["node", str(script), str(Path(ternlight_dir).resolve()), tier],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self.cache: dict[str, torch.Tensor] = {}
        self.name = f"ternlight-{tier}"

    def vectors(self, texts: list[str]) -> torch.Tensor:
        """(n, 384) unit vectors, in order."""
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            self.proc.stdin.write(json.dumps(new) + "\n")
            self.proc.stdin.flush()
            for t, v in zip(new, json.loads(self.proc.stdout.readline())):
                self.cache[t] = torch.tensor(v, dtype=torch.float32)
        return torch.stack([self.cache[t] for t in texts])

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def open_reader(spec: str | Path, tier: str = "mini"):
    """A reader by where it lives: a dir holding node_modules/@ternlight runs
    the npm engine under node (LiveReader); a file is a TernReader
    (tern_reader.py: a trained .pt, or the shipped model-int4.bin) run in
    PyTorch. Both give (n, 384) unit vectors through .vectors()."""
    p = Path(spec)
    if p.is_file():
        from tern_reader import TorchReader
        return TorchReader(p)
    return LiveReader(p, tier)


def check_reader(reader, cache: str | Path, cfg) -> None:
    """Refuse a live reader other than the one that built the table the
    checkpoint trained on (`cfg.reader_file` in the cache): a table from
    reader_table.py --reader-model records its model file's sha256, one from
    node records none. The shipped weights run in PyTorch count as node's
    (tern_reader.py --check). A cache without the table is not checked."""
    from tern_reader import BIN_SHA256
    path = Path(cache) / getattr(cfg, "reader_file", "reader.pt")
    if not path.exists():
        return
    want = torch.load(path, mmap=True).get("reader_sha")
    got = getattr(reader, "sha", None)
    if got == BIN_SHA256:
        got = None
    if want != got:
        raise SystemExit(f"{path} was built by another reader than {reader.name}: pass the "
                         f"reader it was built with (reader_table.py --reader-model)")


def reader_inputs(source: str, dims: dict, max_tool: int, reader: LiveReader,
                  device=None, cfg=None) -> dict:
    """{"rd_tool": (1, max_tool, 384), "rd_req": (1, 384)} for one serialized
    context, zero on padded tool slots. For a C0 model (`cfg.reader_lines`)
    also rd_const (1, max_const, 384), rd_field (1, max_field, 384) and
    rd_chunk (1, max_chunk, 384), from the same texts prep indexes; a
    request with more chunks than training saw keeps its first max_chunk."""
    ln = Lines(source, dims.get("desc_chars", 60))
    descs = [split_tool_line(t, bool(dims.get("name_words")))[2] for _, t in ln.tools][:max_tool]
    vec = reader.vectors([ln.request] + descs)
    rd_tool = torch.zeros(1, max_tool, vec.size(1))
    rd_tool[0, :len(descs)] = vec[1:]
    out = {"rd_tool": rd_tool, "rd_req": vec[:1]}
    if cfg is not None and getattr(cfg, "reader_lines", False):
        for key, texts, cap in (
                ("rd_const", [reader_text(t) for _, t in ln.consts], cfg.max_const),
                ("rd_field", [reader_text(t) for _, t in ln.fields], cfg.max_field),
                ("rd_chunk", request_pieces(ln.request, dims), cfg.max_chunk)):
            t = torch.zeros(1, cap, vec.size(1))
            keep = [(i, x) for i, x in enumerate(texts[:cap]) if x]
            if keep:
                t[0, [i for i, _ in keep]] = reader.vectors([x for _, x in keep])
            out[key] = t
    return {k: v.to(device) for k, v in out.items()} if device is not None else out


_SLOTS: dict = {}


def request_pieces(request: str, dims: dict) -> list:
    """The request as the planner's `rd_chunk` texts: C0's 4-word chunks, or,
    for a slot cache (slot_cache.py, `dims["slot_reader"]`), the 7 role slots'
    tagged texts in role order (None when absent), then the chunks if the
    cache kept them."""
    sr = dims.get("slot_reader")
    if not sr:
        return chunk_request(request, dims["chunk_words"])
    key = (sr["body"], sr["tagger"])
    if key not in _SLOTS:
        from slot_cache import SlotTexts
        here = Path(__file__).parent
        body = sr["body"]
        if body.startswith("tern-electra:"):
            body = "tern-electra:" + str(here / body.split(":", 1)[1])
        _SLOTS[key] = SlotTexts(body, str(here / sr["tagger"]))
    pieces = _SLOTS[key]([request])[0]
    if sr.get("keep_chunks"):
        pieces += chunk_request(request, dims["chunk_words"])
    return pieces
