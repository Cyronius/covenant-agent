"""C0's checks (.claude/plans/tiny-general-agent-menu.md, C0): the reader on
every constant, every field and the request in chunks.

  python -m pytest test_reader_lines.py

- the request's chunks and a line's reader text come out as specified;
- the live path (play.py, the demo) embeds exactly the texts the cache
  indexed for the same row, so a model trained on the cache reads the same
  thing at run time;
- with the request's words dropped (train.py --no-req-words), the request's
  tokens really don't reach the logits, and with them kept, they do.

Needs data_cache_c0 (prep.py --reader-lines, reader_table.py).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

from corpus import COVENANT  # noqa: F401 -- puts the checkout on sys.path
from harness.context import TaskContext, serialize_context
from model import Config, build_model
from prep import QUERY_PREFIX, Lines, READER_KEYS, cache_keys, chunk_request, reader_text

HERE = Path(__file__).parent
CACHE = HERE / "data_cache_c0"


def test_chunk_request():
    assert chunk_request("delete all the cards owned by bob") == \
        ["delete all the cards", "the cards owned by", "cards owned by bob"]
    # the last window ends on the last word even when the step overshoots
    assert chunk_request("a b c d e f g h i") == \
        ["a b c d", "c d e f", "e f g h", "f g h i"]
    assert chunk_request("list all cards") == ["list all cards"]
    assert chunk_request("a b c d e", 2) == ["a b", "b c", "c d", "d e"]


def test_reader_text():
    assert reader_text("C0 ID:user :: Bob Alvarez") == "Bob Alvarez"
    assert reader_text('S0 STR enum vendor_booking.stage :: vendor_booking.stage "confirmed"') \
        == "vendor booking stage confirmed"
    assert reader_text("I2 ID:card :: card 2 - Write release notes for v4.2") \
        == "card 2 - Write release notes for v4.2"
    assert reader_text("F0 user STR :: user.name") == "user name"


def _table_texts():
    teacher = torch.load(CACHE / "teacher.pt")["texts"]
    texts = [t[len(QUERY_PREFIX):] if t.startswith(QUERY_PREFIX) else t for t in teacher]
    with (CACHE / "reader_texts.jsonl").open(encoding="utf-8") as fh:
        texts += [json.loads(line) for line in fh]
    return texts


def test_live_texts_match_the_cache():
    meta = json.loads((CACHE / "config.json").read_text())
    d = torch.load(CACHE / "holdout.pt", mmap=True)
    rows = pickle.load(open(CACHE / "rows.pkl", "rb"))["holdout"]
    task_ids = [m["task_id"] for m in json.loads((CACHE / "holdout_meta.json").read_text())]
    texts = _table_texts()
    checked = 0
    for n in range(0, len(rows), max(1, len(rows) // 40)):
        if "#s" in task_ids[n]:
            continue                        # a continuation's source carries registers
        r = rows[n]
        src = serialize_context(r["request"], TaskContext.from_json(r["context"]),
                                names=bool(meta.get("names")))
        ln = Lines(src, meta["desc_chars"])
        live = {"t_const": [reader_text(t) for _, t in ln.consts],
                "t_field": [reader_text(t) for _, t in ln.fields],
                "t_chunk": chunk_request(ln.request, meta["chunk_words"])}
        for k in READER_KEYS:
            idx = [int(i) for i in d[k][n] if i >= 0]
            assert [texts[i] for i in idx] == live[k], (task_ids[n], k)
        checked += 1
    assert checked >= 20


def test_electra_reader_live_vectors_match_its_table():
    """The ELECTRA reader (electra_cache.py, electra-only-reader.md step 2):
    run live on a row's texts it gives the vectors data_cache_c0esc's table
    holds, the request's role slots and chunks pooled out of one pass, zero
    where the table has no piece; check_reader accepts it for that table."""
    import pytest
    from live_reader import check_reader, open_reader, reader_inputs
    cache = HERE / "data_cache_c0esc"
    if not (cache / "reader.pt").exists():
        pytest.skip(f"missing {cache} (electra_cache.py apply)")
    tab = torch.load(cache / "reader.pt")
    rd = open_reader("electra")
    check_reader(rd, cache, type("Cfg", (), {"reader_file": "reader.pt"}))
    meta = json.loads((cache / "config.json").read_text())
    d = torch.load(cache / "holdout.pt", mmap=True)
    rows = pickle.load(open(cache / "rows.pkl", "rb"))["holdout"]
    task_ids = [m["task_id"] for m in json.loads((cache / "holdout_meta.json").read_text())]
    table = tab["table"].float()
    lay = meta["layout"]
    cfg = type("Cfg", (), {"reader_lines": True, "max_const": lay["max_const"],
                           "max_field": lay["max_field"], "max_chunk": meta["max_chunk"]})
    checked = 0
    for n in range(0, len(rows), max(1, len(rows) // 12)):
        if "#s" in task_ids[n]:
            continue
        r = rows[n]
        src = serialize_context(r["request"], TaskContext.from_json(r["context"]),
                                names=bool(meta.get("names")))
        live = reader_inputs(src, meta, lay["max_tool"], rd, cfg=cfg)
        for key, col in (("rd_const", "t_const"), ("rd_field", "t_field"), ("rd_chunk", "t_chunk"),
                         ("rd_tool", "t_desc"), ("rd_req", "t_req")):
            idx = d[col][n].reshape(-1)
            got = live[key].reshape(-1, table.size(1))[:len(idx)]
            for j, i in enumerate(idx.tolist()):
                if i >= 0:
                    assert float(got[j] @ table[i]) > 0.995, (task_ids[n], key, j)
                elif key == "rd_chunk":
                    assert float(got[j].abs().sum()) == 0.0, (task_ids[n], key, j)
        checked += 1
    assert checked >= 6


def _small_c0(req_words: bool):
    meta = json.loads((CACHE / "config.json").read_text())
    lay = meta["layout"]
    cfg = Config(binding="structural", causal=True, d=64, ff=256, enc_layers=2,
                 dec_layers=2, in_vocab=meta["in_vocab"], in_pad=meta["in_pad"],
                 max_line=meta["max_line"], max_req=meta["max_req"], canvas=64,
                 n_kw=lay["n_kw"], max_tool=lay["max_tool"], max_field=lay["max_field"],
                 max_const=lay["max_const"], n_reg=lay["n_reg"], reader=True,
                 reader_lines=True, max_chunk=meta["max_chunk"], req_words=req_words)
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    m.set_reader_table(torch.load(CACHE / "reader.pt")["table"])
    d = torch.load(CACHE / "holdout.pt", mmap=True)
    inputs = {k: d[k][:3].clone() for k in cache_keys(d) if k != "tgt"}
    return m, inputs, lay


def test_request_words_reach_the_logits_only_when_kept():
    for req_words in (True, False):
        m, inputs, lay = _small_c0(req_words)
        canvas = torch.randint(2, lay["n_kw"], (3, 64))
        other = dict(inputs, req_tok=inputs["req_tok"].roll(1, 0))
        with torch.no_grad():
            a, b = m.decode(inputs, canvas), m.decode(other, canvas)
        fin = torch.isfinite(a)
        assert torch.equal(fin, torch.isfinite(b))
        moved = float((a[fin] - b[fin]).abs().max())
        assert (moved > 1e-3) if req_words else (moved == 0.0), (req_words, moved)
