"""The reader run live, for contexts no cache has seen (the demo app, play.py):
.claude/plans/tiny-planner-demo.md step 2, .claude/plans/electra-only-reader.md
step 2.

A cache's reader.pt holds precomputed vectors indexed by t_desc / t_req (and,
for a C0 cache, t_const / t_field / t_chunk). A demo request and its world's
tools are new text, so here the reader embeds them on demand:
electra_reader.ElectraReader, which reads descriptions, constants and fields
alone, and an electra_cache.py cache's request pieces (role slots, 4-word
chunks) out of one pass over the request.

`reader_inputs(source, dims, reader)` pulls the same texts prep's teacher
table does (each tool line's description as split_tool_line cuts it, and the
request) and returns the `rd_tool` / `rd_req` inputs model.reader_vectors
takes in place of a table lookup.
"""
from __future__ import annotations

from pathlib import Path

import torch

from prep import Lines, chunk_request, reader_text, split_tool_line


def open_reader(spec: str | Path = "electra"):
    """`electra` (R27's head n) or an embedding head's head.pt, with the
    ternary ELECTRA body and the tagger: an electra_reader.ElectraReader,
    which gives (n, 384) unit vectors through .vectors()."""
    from electra_reader import HEAD, ElectraReader
    p = Path(spec)
    if str(spec) != "electra" and p.name != "head.pt":
        raise SystemExit(f"unknown reader {spec!r}: `electra`, or an embedding head's head.pt")
    return ElectraReader(HEAD if str(spec) == "electra" else p)


def check_reader(reader, cache: str | Path, cfg) -> None:
    """Refuse a live reader other than the one that built the table the
    checkpoint trained on (`cfg.reader_file` in the cache): the table records
    its reader's sha256 (electra_cache.py, reader_table.py). A cache without
    the table is not checked."""
    path = Path(cache) / getattr(cfg, "reader_file", "reader.pt")
    if not path.exists():
        return
    want = torch.load(path, mmap=True).get("reader_sha")
    if want != getattr(reader, "sha", None):
        raise SystemExit(f"{path} was built by another reader than {reader.name}: rebuild the cache's "
                         f"table (electra_cache.py) or pass the head it was built with")


def reader_inputs(source: str, dims: dict, max_tool: int, reader, device=None, cfg=None) -> dict:
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
        er = dims.get("electra_reader")
        lines = [("rd_const", [reader_text(t) for _, t in ln.consts], cfg.max_const),
                 ("rd_field", [reader_text(t) for _, t in ln.fields], cfg.max_field)]
        if er:      # electra_cache.py: role slots and chunks pooled out of one pass
            out["rd_chunk"] = reader.request_pieces([ln.request], dims["chunk_words"], er["chunks"],
                                                    er["slots"])[0][None, :cfg.max_chunk]
        else:
            lines.append(("rd_chunk", chunk_request(ln.request, dims["chunk_words"]), cfg.max_chunk))
        for key, texts, cap in lines:
            t = torch.zeros(1, cap, vec.size(1))
            keep = [(i, x) for i, x in enumerate(texts[:cap]) if x]
            if keep:
                t[0, [i for i, _ in keep]] = reader.vectors([x for _, x in keep])
            out[key] = t
    return {k: v.to(device) for k, v in out.items()} if device is not None else out
