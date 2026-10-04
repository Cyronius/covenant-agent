"""The slot reader's checks (.claude/plans/electra-slot-reader.md):

  python -m pytest test_slot_reader.py

- role spans come out of spaCy's parse as specified;
- the swap and to->from check pairs are built as specified;
- the slot head gives one unit vector and one present score per role, and
  padding takes no part.

Needs spaCy's en_core_web_sm.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from slot_reader import ROLES, SlotHead, roles_of, span_texts, swap_pairs, tofrom_pairs

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    return spacy.load("en_core_web_sm")


def _roles(nlp, text):
    return span_texts(text, roles_of(nlp(text)))


def test_roles_of_a_request(nlp):
    assert _roles(nlp, "move it to bob") == {
        "whole": "move it to bob", "action": "move", "object": "it", "destination": "bob"}
    assert _roles(nlp, "send priya the list")["destination"] == "priya"
    r = _roles(nlp, "Moves the file out of the archive folder")
    assert (r["object"], r["source"]) == ("the file", "the archive folder")


def test_roles_of_a_description(nlp):
    r = _roles(nlp, "Copies the object from the source bucket to the destination bucket.")
    assert (r["action"], r["object"], r["source"], r["destination"]) == \
        ("Copies", "the object", "the source bucket", "the destination bucket")


def test_conditions_time_and_amount(nlp):
    r = _roles(nlp, "list the top 5 orders placed before March")
    assert (r["object"], r["condition"], r["time"], r["amount"]) == \
        ("the top 5 orders", "placed before March", "before March", "5")
    r = _roles(nlp, "delete the cards that are not done")
    assert (r["object"], r["condition"]) == ("the cards", "that are not done")
    # a number the tagger calls a date is not a time
    assert "time" not in _roles(nlp, "a unit of information equal to 1024 bytes")


def test_swap_and_to_from_pairs():
    rows = [{"text": "copy the report to the inbox",
             "roles": {"object": [5, 15], "destination": [19, 28]}}]
    assert swap_pairs(rows, "object", "destination") == [
        ("copy the report to the inbox", "copy the inbox to the report", "the report", "the inbox")]
    assert tofrom_pairs(rows) == [("copy the report to the inbox", "copy the report from the inbox", "the inbox")]


def test_ternary_electra_rebuild_matches_electra():
    """tern_electra.py: with rounding off, the student is ELECTRA-small, state
    for state and attention for attention; int4 rounding of its word table is
    stable once applied."""
    pytest.importorskip("transformers")
    from tern_electra import Teacher, TernElectra
    t = Teacher()
    s = TernElectra.from_electra(t.m).eval()
    s.set_quant(False)
    ids, mask = t.batch(["move it to bob", "Copies the object from the source bucket to the destination bucket."])
    ts, ta = t(ids, mask)
    with torch.no_grad():
        ss, sa = s(ids, mask)
    assert max(float((a - b)[mask].abs().max()) for a, b in zip(ss, ts)) < 1e-4
    assert max(float((a - b).abs().max()) for a, b in zip(sa, ta)) < 1e-4
    s.quantize_embedding_int4_()
    before = s.word.weight.detach().clone()
    s.quantize_embedding_int4_()
    assert torch.equal(before, s.word.weight.detach())


def test_head_shapes_and_padding():
    torch.manual_seed(0)
    head = SlotHead(n_states=3, d=32, out=48, heads=4).eval()
    states = torch.randn(3, 2, 6, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]).bool()
    slots, logits = head(states, mask)
    assert slots.shape == (2, len(ROLES), 48) and logits.shape == (2, len(ROLES))
    assert torch.allclose(slots.norm(dim=-1), torch.ones(2, len(ROLES)), atol=1e-5)
    noisy = states.clone()
    noisy[:, 1, 3:] = 100.0                 # padded words of the second text
    with torch.no_grad():
        again, _ = head(noisy, mask)
    assert torch.allclose(again[1], slots[1], atol=1e-5)


def test_tagger_decodes_spans():
    """slot_tagger.decode: the best start/end pair per role, or None when the
    [CLS] (absent) score is higher; special and padding positions never start
    or end a span."""
    from slot_tagger import TAGGED, decode
    R, T = len(TAGGED), 6
    offsets = torch.tensor([[[0, 0], [0, 4], [5, 7], [8, 10], [0, 0], [0, 0]]])   # [CLS] w w w [SEP] pad
    start = torch.full((1, R, T), -5.0)
    end = torch.full((1, R, T), -5.0)
    start[0, 0, 1], end[0, 0, 2] = 3.0, 3.0           # role 0: words 1-2
    start[0, 1, 0], end[0, 1, 0] = 9.0, 9.0           # role 1: absent wins
    start[0, 2, 4], end[0, 2, 4] = 9.0, 9.0           # role 2: only [SEP] scores high -> best real word pair
    out = decode(start, end, offsets)[0]
    assert out[0] == (0, 7)
    assert out[1] is None
    assert out[2] is None or out[2][1] <= 10


def _need(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"missing {p}")


def test_packed_reader_reads_back_exactly():
    """pack_reader.py: the packed body, tagger and embedding head give the
    trained model's word states, tags and (to fp16 rounding) vectors."""
    from pack_reader import BODY, HEAD, TAGGER, _load, pack, unpack
    from slot_tagger import Tok, decode
    _need(BODY, TAGGER, HEAD)
    body, tagger, head = _load(BODY, TAGGER, HEAD)
    b2, t2, h2 = unpack(pack(body, tagger, head))
    b2.eval(), t2.eval(), h2.eval()
    ids, mask, offsets = Tok()(["move it to bob", "copy the report from the archive to the inbox"])
    with torch.no_grad():
        s1, _ = body(ids, mask)
        s2, _ = b2(ids, mask)
        assert all(torch.equal(a[mask], b[mask]) for a, b in zip(s1, s2))
        assert decode(*tagger(torch.stack(s1), mask), offsets) == decode(*t2(torch.stack(s2), mask), offsets)
        word = offsets[..., 1] > offsets[..., 0]
        rows = torch.arange(2)
        v1, v2 = head(torch.stack(s1), rows, word), h2(torch.stack(s2), rows, word)
        assert float((v1 * v2).sum(-1).min()) > 0.999
