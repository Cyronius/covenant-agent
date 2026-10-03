"""The staged decoder (staged.py): stage specs, the program confidence the
early exit reads, task types, and that a tiny staged model memorizes a
handful of real rows through both of its readouts -- the refiner's and the
draft's -- then reloads as the same model.

  python -m pytest test_staged.py -q        # from models/tiny; CPU, about a minute
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from canvas import Layout, TaskCodec, load_keywords
from evaluate import load_model
from model import Config, build_model
from sample import ar_sample
from staged import fold_of, parse_stages, program_confidence, task_type
from train import load_split, unpack

HERE = Path(__file__).parent
CACHE = HERE / "data_cache_smoke"


def test_parse_stages():
    assert parse_stages("draft:4x4,refine:2x1") == [("draft", 4, 4), ("refine", 2, 1)]
    assert parse_stages("draft:2x4, draft:2x4") == [("draft", 2, 4), ("draft", 2, 4)]
    for bad in ("refine:2x1,draft:4x4", "draft:4", "loop:1x1", ""):
        with pytest.raises(ValueError):
            parse_stages(bad)


def test_program_confidence_reads_up_to_the_first_pad():
    tok = torch.tensor([[5, 6, 0, 0], [5, 6, 7, 8]])
    conf = torch.tensor([[0.9, 0.8, 0.7, 0.1], [0.9, 0.95, 0.6, 0.99]])
    assert program_confidence(tok, conf).tolist() == pytest.approx([0.7, 0.6])


def test_task_types():
    assert task_type("app_checkout") == "pages"
    assert task_type("app_store") == "data"          # a data theme, not a page app
    assert task_type("house") == "obsact"
    assert task_type("service_retail") == "data"
    assert task_type("bookstore") == "data"


def _tiny(stages: str) -> Config:
    meta = json.loads((CACHE / "config.json").read_text(encoding="utf-8"))
    lay = Layout.from_dict(meta["layout"])
    return Config(in_vocab=meta["in_vocab"], out_vocab=lay.size, d=64, ff=128, heads=4,
                  enc_layers=1, line_layers=1, graph_layers=1, dec_layers=0,
                  canvas=meta["canvas"], causal=True, binding="structural",
                  in_pad=meta["in_pad"], max_line=meta["max_line"], max_req=meta["max_req"],
                  n_kw=lay.n_kw, max_tool=lay.max_tool, max_field=lay.max_field,
                  max_const=lay.max_const, n_reg=lay.n_reg, dropout=0.0, stages=stages,
                  draft_noise=0.0)


def _codec(i: int):
    meta = json.loads((CACHE / "config.json").read_text(encoding="utf-8"))
    rows = json.loads((CACHE / "train_meta.json").read_text(encoding="utf-8"))
    return TaskCodec(load_keywords(CACHE / "keywords.json"),
                     Layout.from_dict(meta["layout"]), **rows[i]["syms"])


def _program(canvas: torch.Tensor) -> list[int]:
    ids = canvas[0].tolist()
    return ids[:ids.index(0)] if 0 in ids else ids


@pytest.mark.skipif(not CACHE.exists(), reason="needs data_cache_smoke")
def test_memorizes_and_reloads(tmp_path):
    torch.manual_seed(0)
    n = 12
    ds = load_split(CACHE, "train", "structural", limit=n)
    inputs, tgt = unpack(tuple(ds.tensors), "structural", 0, ds.keys)
    model = build_model(_tiny("draft:1x2,refine:1x1"))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(400):
        loss, _, _, _ = model.staged_loss(inputs, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    def readouts(m):
        ref, dra = [], []
        for i in range(n):
            one = {k: v[i:i + 1] for k, v in inputs.items()}
            with torch.no_grad():
                ref.append(_program(ar_sample(m, one, _codec(i))[0]))
                with m.exiting():
                    dra.append(_program(ar_sample(m, one, _codec(i))[0]))
        return ref, dra

    ref, dra = readouts(model)
    want = [_program(tgt[i:i + 1]) for i in range(n)]
    assert sum(r == w for r, w in zip(ref, want)) >= n - 1, "the refiner did not memorize"
    assert sum(d == w for d, w in zip(dra, want)) >= n - 2, "the draft did not memorize"

    # the checkpoint reloads as the same model, through evaluate.load_model
    path = tmp_path / "best.pt"
    torch.save({"cfg": model.c.__dict__, "model": model.state_dict()}, path)
    again = load_model(path, "cpu")
    assert readouts(again) == (ref, dra)

    # STAGED_EXIT=always reads the draft out; never always refines
    one = {k: v[:1] for k, v in inputs.items()}
    os.environ["STAGED_EXIT"] = "always"
    try:
        with torch.no_grad():
            assert _program(ar_sample(again, one, _codec(0))[0]) == dra[0]
            assert again.last["exit"]
        os.environ["STAGED_EXIT"] = "never"
        with torch.no_grad():
            assert _program(ar_sample(again, one, _codec(0))[0]) == ref[0]
            assert not again.last["exit"]
    finally:
        os.environ.pop("STAGED_EXIT", None)


@pytest.mark.skipif(not CACHE.exists(), reason="needs data_cache_smoke")
def test_draft_only_and_frozen_expert():
    """A draft-only model reads its draft out; an expert built on it trains
    its stages while the encoder and head stay put."""
    torch.manual_seed(0)
    ds = load_split(CACHE, "train", "structural", limit=4)
    inputs, tgt = unpack(tuple(ds.tensors), "structural", 0, ds.keys)
    base = build_model(_tiny("draft:1x2,refine:1x1"))
    cfg = _tiny("draft:1x2,refine:1x1")
    cfg.freeze_shared, cfg.train_parts = True, "draft"
    exp = build_model(cfg)
    exp.load_state_dict(base.state_dict())
    keep = {id(p) for p in exp.trainable_parts()}
    for p in exp.parameters():
        p.requires_grad_(id(p) in keep)
    before = {k: v.clone() for k, v in exp.state_dict().items()}
    opt = torch.optim.Adam([p for p in exp.parameters() if p.requires_grad], lr=1e-2)
    exp.train()
    loss, _, _, _ = exp.staged_loss(inputs, tgt)
    loss.backward()
    opt.step()
    after = exp.state_dict()
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved and all(k.startswith(("dec.0.", "draft_norm.")) for k in moved), moved

    solo = build_model(_tiny("draft:1x1"))
    solo.eval()
    with torch.no_grad():
        mem = solo.encode_inputs({k: v[:1] for k, v in inputs.items()})
        assert bool(mem.exit.all())
        out = solo.decode({k: v[:1] for k, v in inputs.items()}, tgt[:1], mem=mem)
    assert torch.equal(out, mem.draft_logits)


@pytest.mark.skipif(not CACHE.exists(), reason="needs data_cache_smoke")
def test_same_kind_swap_noise():
    cfg = _tiny("draft:1x1,refine:1x1")
    cfg.draft_swap = 1.0
    m = build_model(cfg)
    c = m.c
    b = [c.n_kw - 1, c.n_kw, c.n_kw + c.max_tool, c.n_kw + c.max_tool + c.max_field,
         c.n_kw + c.max_tool + c.max_field + c.max_const]
    assert m._kind_of(torch.tensor(b)).tolist() == [0, 1, 2, 3, 4]
    ds = load_split(CACHE, "train", "structural", limit=4)
    inputs, tgt = unpack(tuple(ds.tensors), "structural", 0, ds.keys)
    m.train()
    loss, _, _, _ = m.staged_loss(inputs, tgt)
    assert torch.isfinite(loss)


def test_fold_of_keeps_episodes_together():
    assert fold_of("app_checkout_ep20266925_t1") == fold_of("app_checkout_ep20266925_t7")
    ids = [f"bookstore_L{i % 19}_{20260000 + i}" for i in range(2000)]
    share = sum(fold_of(i) for i in ids) / len(ids)
    assert 0.45 < share < 0.55


@pytest.mark.skipif(not CACHE.exists(), reason="needs data_cache_smoke")
def test_cross_fitted_drafts(tmp_path, monkeypatch):
    """Stored drafts ride through load_split's fold filter row for row, and
    the refiner reads them in training -- its own draft stage's otherwise."""
    meta = json.loads((CACHE / "train_meta.json").read_text(encoding="utf-8"))
    n = len(meta)
    tok = torch.randint(2, 40, (n, 64), dtype=torch.int16)
    conf = torch.rand(n, 64).half()
    path = tmp_path / "x.pt"
    torch.save({"tok": tok, "conf": conf}, path)
    ds = load_split(CACHE, "train", "structural", fold=(1, 2), xdraft=path)
    keep = [i for i, m in enumerate(meta) if fold_of(m["task_id"]) == 1]
    got = ds.tensors[ds.keys.index("xdraft_tok")]
    assert len(got) == len(keep) and torch.equal(got, tok[keep])
    with pytest.raises(ValueError):
        torch.save({"tok": tok[:-1], "conf": conf[:-1]}, path)
        load_split(CACHE, "train", "structural", xdraft=path)

    inputs, tgt = unpack(tuple(t[:4] for t in ds.tensors), "structural", 0, ds.keys)
    m = build_model(_tiny("draft:1x1,refine:1x1"))           # draft_noise 0
    seen = []
    real = m.draft_rows
    monkeypatch.setattr(m, "draft_rows", lambda table, t, c: seen.append((t, c)) or real(table, t, c))
    m.train()
    m.staged_loss(inputs, tgt)
    assert torch.equal(seen[-1][0], inputs["xdraft_tok"].long())
    assert torch.allclose(seen[-1][1], inputs["xdraft_conf"].float())
    m.eval()
    with torch.no_grad():
        m.staged_loss(inputs, tgt)
        live = m.draft(inputs, mem := m._encode(inputs), m.slot_table(mem)).argmax(-1)
    assert torch.equal(seen[-1][0], live)


def test_old_checkpoint_loads_unchanged():
    ck = HERE / "runs/pod_rd/runs/fdc25_A0/best.pt"
    if not ck.exists():
        pytest.skip("no fdc25_A0 checkpoint here")
    m = load_model(ck, "cpu")
    assert type(m).__name__ == "StructuralModel" and not m.c.stages
