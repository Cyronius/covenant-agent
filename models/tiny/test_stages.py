"""The split encoder's equivalence checks (description-reading plan step 2).

  python -m pytest test_stages.py

- packed and padded line encoders give the same logits at the same weights
  (difference under 1e-5), on real rows through the real input path;
- a live line never reads a padded one in the schema graph (the layout bug
  `prep.layout_edges` fixed: before it, R9/R10's tools were linked to padded
  tool rows instead of their fields);
- the split model runs end to end, and a text stage packed equals padded.

Traces: description-reading plan step 2, gates "packed = padded" and
"flags off reproduces R10".
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import torch

from corpus import COVENANT  # noqa: F401 -- puts the checkout on sys.path
from harness.context import TaskContext, serialize_context
from model import Config, build_model
from prep import Lines, encode_one, layout_edges, split_tool_line
from stages import TextStage, multi_positive_nce, relational_loss
from canvas import Layout, context_symbols

HERE = Path(__file__).parent
PAD = 1


def _rows(n=4):
    """A few rows of a combined holdout: plain and decoyed, so the tool
    count differs from max_tool and the old layout bug would show."""
    rows = pickle.load(open(HERE / "data_cache_ho42_reg" / "rows.pkl", "rb"))["holdout"]
    rng = random.Random(0)
    return rng.sample(rows, n)


def _inputs(rows, tk, layout, dims, names=False):
    batch = []
    for r in rows:
        ctx = TaskContext.from_json(r["context"])
        src = serialize_context(r["request"], ctx, names=names)
        batch.append(encode_one(src, context_symbols(r["context"]), tk, layout, dims))
    return {k: torch.cat([b[k] for b in batch]) for k in batch[0]}


def _setup(split=False):
    from tokenizers import Tokenizer
    import json
    cache = HERE / "data_cache_ho42_reg"
    meta = json.loads((cache / "config.json").read_text())
    tk = Tokenizer.from_file(str(cache / "in_tok.json"))
    tk.no_truncation(); tk.no_padding()
    layout = Layout.from_dict(meta["layout"])
    dims = {"max_line": meta["max_line"], "max_req": meta["max_req"], "max_reg": 8}
    if split:
        dims.update({"split": True, "desc_chars": 120, "max_sig": 64,
                     "max_desc": 64, "max_name": 16})
    cfg = dict(binding="structural", causal=True, d=64, ff=256, enc_layers=2,
               dec_layers=2, in_vocab=meta["in_vocab"], in_pad=meta["in_pad"],
               max_line=meta["max_line"], max_req=meta["max_req"], canvas=64,
               n_kw=layout.n_kw, max_tool=layout.max_tool,
               max_field=layout.max_field, max_const=layout.max_const,
               n_reg=layout.n_reg)
    return tk, layout, dims, cfg


def test_no_live_line_reads_a_padded_one():
    tk, layout, dims, _ = _setup()
    for r in _rows(6):
        ln = Lines(serialize_context(r["request"], TaskContext.from_json(r["context"])))
        live = set(range(len(ln.tools))) | {layout.max_tool + j
                                            for j in range(len(ln.fields))}
        for i, j in layout_edges(ln, layout.max_tool):
            assert i in live and j in live
        # and every tool that names a field is linked to that field's row
        assert any(j >= layout.max_tool for _, j in layout_edges(ln, layout.max_tool))


def test_packed_equals_padded():
    tk, layout, dims, cfg = _setup()
    inputs = _inputs(_rows(4), tk, layout, dims)
    torch.manual_seed(0)
    m = build_model(Config(**cfg)).eval()
    canvas = torch.randint(2, layout.n_kw, (4, 64))
    with torch.no_grad():
        a = m.decode(inputs, canvas)
        m.c.pack = True
        b = m.decode(inputs, canvas)
    assert torch.equal(torch.isfinite(a), torch.isfinite(b))
    fin = torch.isfinite(a)
    assert float((a[fin] - b[fin]).abs().max()) < 1e-5


def test_split_model_runs_and_packs():
    tk, layout, dims, cfg = _setup(split=True)
    inputs = _inputs(_rows(3), tk, layout, dims, names=True)
    cfg = {**cfg, "split": True, "sig_w": 32, "desc_w": 48, "name_w": 32,
           "max_sig": 64, "max_desc": 64, "max_name": 16}
    torch.manual_seed(0)
    m = build_model(Config(**cfg)).eval()
    canvas = torch.randint(2, layout.n_kw, (3, 64))
    with torch.no_grad():
        a = m.decode(inputs, canvas)
        m.c.pack = True
        b = m.decode(inputs, canvas)
    fin = torch.isfinite(a)
    assert fin.any() and torch.equal(fin, torch.isfinite(b))
    assert float((a[fin] - b[fin]).abs().max()) < 1e-5
    assert m.last_gate.shape[-1] == 3


def test_split_tool_line():
    assert split_tool_line("T4 archive_card (I:card=F0) -> OBJ:card [M] :: Archive it") == \
        ("T4 (I:card=F0) -> OBJ:card [M]", "archive_card", "Archive it")
    assert split_tool_line("T4 (I:card=F0) -> - [] :: x") == ("T4 (I:card=F0) -> - []", "", "x")


def test_text_stage_packed_equals_padded():
    torch.manual_seed(0)
    st = TextStage(100, 32, 2, 20, PAD).eval()
    tok = torch.full((2, 5, 20), PAD)
    tok[0, 0, :7] = torch.randint(2, 100, (7,))
    tok[0, 3, :12] = torch.randint(2, 100, (12,))
    tok[1, 1, :3] = torch.randint(2, 100, (3,))
    with torch.no_grad():
        a = st(tok, packed=False)
        b = st(tok, packed=True)
    live = (tok != PAD).any(-1)
    assert float((a[live] - b[live]).abs().max()) < 1e-5
    assert float(b[~live].abs().max()) == 0.0


def test_losses_are_zero_where_they_should_be():
    logits = torch.tensor([[5.0, -5.0, 0.0]])
    pos = torch.tensor([[True, False, False]])
    live = torch.tensor([[True, True, False]])
    assert float(multi_positive_nce(logits, pos, live)) < 1e-3
    v = torch.randn(1, 3, 8)
    assert float(relational_loss(v, v, torch.ones(1, 3, dtype=torch.bool))) < 1e-6
