"""The role-aware reader's checks (.claude/plans/role-aware-reader.md):

  python -m pytest test_tern_reader.py

- the PyTorch rebuild of Ternlight-mini (tern_reader.py) gives node's
  embed() vectors, and its training weights round back to the published
  ternary codes and scales, and its embeddings to the same int4 rows;
- with the position table at zero the reader is order-blind, as shipped, and
  a nonzero table breaks that;
- role flips (tern_train.py) come out as specified.

Needs reader/ternlight-mini (python tern_reader.py --fetch) and, for the
node comparison, reader/node_modules/@ternlight/mini.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from tern_reader import TERN_DIR, TernReader, Tokenizer, TorchReader, read_bin
from tern_train import Flipper, make_flips

HERE = Path(__file__).parent
BIN = TERN_DIR / "model-int4.bin"
TEXTS = ["move it to bob", "assign card 6 to Bob", "cards that are not done",
         "Copies the object from the source bucket to the destination bucket.",
         "record status disabled", "the release notes", "a", ""]


def _need(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"missing {p}")


def test_rebuild_matches_node():
    _need(BIN, HERE / "reader" / "node_modules" / "@ternlight" / "mini")
    from live_reader import LiveReader
    rng = random.Random(0)
    words = "the card to from bob list move status done not before after".split()
    texts = TEXTS + [" ".join(rng.choices(words, k=rng.randint(1, 40))) for _ in range(100)]
    live = LiveReader(HERE / "reader")
    ref = live.vectors(texts)
    live.close()
    got = TorchReader(BIN).embed(texts)
    assert float((ref * got).sum(1).min()) >= 0.9999


def test_training_weights_round_back():
    _need(BIN)
    w = read_bin(BIN)
    m = TernReader.from_bin(BIN)
    for layer, L in zip(m.layers, w["layers"]):
        for name in ("q", "k", "v", "out", "fc1", "fc2"):
            q, scale = getattr(layer, name).ternary()
            assert torch.equal(q.to(torch.int8), L[name]["q"]), name
            assert abs(scale - L[name]["scale"]) <= 1e-6 * L[name]["scale"], name
    before = m.emb.weight.detach().clone()
    m.quantize_embedding_int4_()
    assert torch.equal(m.emb.weight.detach(), before)


def test_positions_zero_is_order_blind():
    _need(BIN)
    m = TernReader.from_bin(BIN).eval()
    tk = Tokenizer()
    a, b = "assign card 6 to Bob", "assign Bob to card 6"
    with torch.no_grad():
        v = m(*Tokenizer.batch(tk.ids([a, b])))
        assert float(v[0] @ v[1]) > 0.99999
        torch.manual_seed(0)
        m.pos.normal_(0, 0.05)
        v = m(*Tokenizer.batch(tk.ids([a, b])))
        assert float(v[0] @ v[1]) < 0.999


def test_offset_biases_zero_is_order_blind():
    _need(BIN)
    m = TernReader.from_bin(BIN, pos="rel").eval()
    ship = TernReader.from_bin(BIN).eval()
    tk = Tokenizer()
    a, b = "assign card 6 to Bob", "assign Bob to card 6"
    batch = Tokenizer.batch(tk.ids([a, b, "a longer text, so the other two are padded here"]))
    with torch.no_grad():
        v = m(*batch)
        assert torch.allclose(v, ship(*batch), atol=1e-6)
        assert float(v[0] @ v[1]) > 0.99999
        torch.manual_seed(0)
        for layer in m.layers:
            layer.rel.normal_(0, 1.0)
        v = m(*batch)
        assert float(v[0] @ v[1]) < 0.999
        # padding still takes no part: alone, a text reads the same
        alone = m(*Tokenizer.batch(tk.ids([a])))
        assert torch.allclose(alone[0], v[0], atol=1e-5)


def test_padding_does_not_move_a_vector():
    _need(BIN)
    r = TorchReader(BIN, batch=len(TEXTS))
    alone = torch.cat([r.embed([t]) for t in TEXTS])
    together = r.embed(TEXTS)
    assert torch.allclose(alone, together, atol=1e-5)


def _flips(text):
    words, sites = Flipper().sites(text)
    return {(k, " ".join(words[:a] + new + words[b:])) for k, a, b, new in sites}


def test_role_word_flips():
    assert ("word:to", "move it from bob") in _flips("move it to bob")
    assert ("word:before", "orders placed after March") in _flips("orders placed before March")
    assert ("word:into", "Moves the file out of the archive") in _flips("Moves the file into the archive")
    assert ("word:out of", "Moves the file into the archive") in _flips("Moves the file out of the archive")
    # "to" + a verb is an infinitive, not a direction
    assert not any(k == "word:to" for k, _ in _flips("she was able to swim"))


def test_not_flips():
    assert ("not:add", "cards that are not done") in _flips("cards that are done")
    assert ("not:drop", "The request is valid.") in _flips("The request is not valid.")
    assert ("not:drop", "This value can be changed") in _flips("This value cannot be changed")


def test_phrase_swaps():
    assert ("swap:from-to", "Copies the object from the destination bucket to the source bucket.") \
        in _flips("Copies the object from the source bucket to the destination bucket.")
    assert ("swap:to", "Adds the group to the user") in _flips("Adds the user to the group")


def test_flip_choice_is_seeded_by_the_text():
    texts = ["Adds the user to the group", "cards that are done", "move it to bob"]
    one = make_flips(texts, 0, 0.0)
    two = make_flips(list(reversed(texts)), 0, 0.0)
    assert sorted(map(str, one)) == sorted(map(str, two))
