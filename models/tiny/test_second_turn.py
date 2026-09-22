"""The tiny model's second turn (plan step 4).

Before this the model could not take one at all: `corpus.load` admitted
single-segment tasks only, `prep.Lines` refused a `REGISTERS:` section
outright, and region A had nowhere to put one. A continuation is not a longer
program — it is the *same* context plus what is already bound, and the target
is only the part that starts there.

What has to hold: a bound register is rendered in a form a model can read and
a line parser can take back apart; a paused task becomes two training rows,
the second carrying the registers the first one left; the encoder grows by
that region and its pointer bank does not (a canvas slot still points at a
tool, a field or a constant, and `r1` was always an output row).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

COVENANT = Path(__file__).resolve().parents[2]
if str(COVENANT) not in sys.path:
    sys.path.insert(0, str(COVENANT))

from corpus import load  # noqa: E402
from model import Config, StructuralModel  # noqa: E402
from prep import Lines, encode_one  # noqa: E402

from harness.context import render_register, serialize_context  # noqa: E402
from core.ir import TaskContext  # noqa: E402

CURRICULUM = COVENANT / "data" / "curriculum_tasks.jsonl"


def test_a_list_register_says_how_many_and_names_a_few():
    cards = [{"id": f"card_{i}", "title": "x" * 80} for i in range(9)]
    line = render_register("r1", ("LIST", ("OBJ", "card")), cards)
    assert line == "r1 LIST OBJ:card n=9 [card_0 card_1 card_2 ...]"
    # the cap is what keeps a register line inside the line budget: nine full
    # kanban cards are 3.6 KB
    assert len(line) < 80


@pytest.mark.parametrize("type_,value,expected", [
    (("LIST", ("OBJ", "card")), [], "r0 LIST OBJ:card n=0 []"),
    (("OBJ", "card"), {"id": "card_3"}, "r0 OBJ:card card_3"),
    (("INT",), 7, "r0 INT 7"),
    (("BOOL",), False, "r0 BOOL false"),
    (("STR",), "x" * 60, "r0 STR " + "x" * 40 + "..."),
    (("LIST", ("OBJ", "card")), None, "r0 LIST OBJ:card"),
])
def test_every_register_shape_renders(type_, value, expected):
    assert render_register("r0", type_, value) == expected


def test_the_registers_section_parses_back_out():
    """`serialize_context` writes it and `prep.Lines` reads it: before step 4
    the parser exited on the section rather than dropping it."""
    row = next(_rows(paused=True))
    ctx = TaskContext.from_json(row["context"])
    object.__setattr__(ctx, "initial_registers",
                       {"r0": ("LIST", ("OBJ", "card")), "r3": ("INT",)})
    source = serialize_context(row["request"], ctx,
                               {"r0": [{"id": "card_1"}], "r3": 4})
    ln = Lines(source)
    assert [sym for sym, _ in ln.regs] == ["r0", "r3"]
    assert ln.regs[1][1] == "r3 INT 4"
    assert ln.request == row["request"]
    assert len(ln.tools) == len(row["context"]["tools"])


def _rows(paused: bool):
    import json
    with open(CURRICULUM, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            segs = row.get("reference", {}).get("segments") or []
            if (len(segs) > 1) == paused:
                yield row


def test_a_paused_task_becomes_one_row_per_segment():
    examples = load(CURRICULUM)
    by_id = {e.task_id: e for e in examples}
    first = [e for e in examples if e.task_id.endswith("#s0")]
    cont = [e for e in examples if e.task_id.endswith("#s1")]
    assert first and cont, "the curriculum's L10 tasks produced no segments"
    for e in cont:
        head = by_id[e.task_id.replace("#s1", "#s0")]
        segs = e.row["reference"]["segments"]
        assert e.target == segs[1] and head.target == segs[0]
        # the continuation reads what the first segment left bound; the first
        # segment has nothing to read
        assert "REGISTERS:" in e.source
        assert "REGISTERS:" not in head.source
        assert "PAUSE" in head.target and "PAUSE" not in e.target


def test_a_task_whose_reference_does_not_replay_is_dropped_not_guessed():
    """`_replay` runs the reference for real; a row it cannot execute has no
    honest register state, so it produces no training rows at all."""
    from corpus import _replay
    row = next(_rows(paused=True))
    broken = dict(row, reference={"segments": ["CALL @nope\nPAUSE\n", "STOP\n"]})
    assert _replay(broken, broken["reference"]["segments"]) is None


class _CharTok:
    """A stand-in tokenizer: `encode_one` only needs ids and a pad id."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def token_to_id(self, _tok):
        return 0

    def encode_batch(self, texts):
        return [self._Enc([ord(ch) % 97 + 1 for ch in t]) for t in texts]


def _layout():
    from canvas import Layout
    return Layout(54, 4, 5, 3)


def test_encode_one_puts_the_registers_in_their_own_region():
    source = ("REQUEST: archive them\n"
              "TOOLS:\nT0 () -> LIST OBJ:card [] :: list cards\n"
              "FIELDS:\nF0 card STR :: title\n"
              "CONSTANTS:\nC0 BOOL :: true\n"
              "REGISTERS:\nr0 LIST OBJ:card n=4 [card_1 card_2]\nr1 INT 4\n")
    syms = {"tools": ["T0"], "fields": ["F0"], "consts": ["C0"]}
    dims = {"max_line": 32, "max_req": 16, "max_reg": 8}
    inputs = encode_one(source, syms, _CharTok(), _layout(), dims)
    assert inputs["reg_tok"].shape == (1, 8, 32)
    assert int(inputs["n_reg_bound"]) == 2
    assert int(inputs["n_tool"]) == 1 and int(inputs["n_const"]) == 1
    # rows past the bound count stay padding
    assert int((inputs["reg_tok"][0, 2:] != 0).sum()) == 0


def _model(max_reg=4):
    c = Config(binding="structural", d=32, heads=2, ff=64, enc_layers=1,
               dec_layers=1, line_layers=1, graph_layers=1, canvas=8,
               n_kw=4, max_tool=2, max_field=2, max_const=1, n_reg=4,
               in_vocab=32, max_line=6, max_req=5, dropout=0.0)
    return c, StructuralModel(c)


def _inputs(c, n_reg_lines):
    B, TL = 1, c.max_line
    return {
        "tool_tok": torch.ones(B, c.max_tool, TL, dtype=torch.long),
        "field_tok": torch.ones(B, c.max_field, TL, dtype=torch.long),
        "const_tok": torch.ones(B, c.max_const, TL, dtype=torch.long),
        "reg_tok": torch.ones(B, n_reg_lines, TL, dtype=torch.long),
        "req_tok": torch.ones(B, c.max_req, dtype=torch.long),
        "n_tool": torch.tensor([2]), "n_field": torch.tensor([2]),
        "n_const": torch.tensor([1]),
        "n_reg_bound": torch.tensor([min(2, n_reg_lines)]),
        "adj": torch.ones(B, c.max_tool + c.max_field,
                          c.max_tool + c.max_field, dtype=torch.bool),
    }


def test_region_a_grows_by_the_register_lines_and_the_pointer_bank_does_not():
    c, m = _model()
    m.eval()
    with torch.no_grad():
        bare = m.encode_inputs(_inputs(c, 0))
        withr = m.encode_inputs(_inputs(c, 3))
    assert withr.mem.size(1) == bare.mem.size(1) + 3
    assert withr.n_ptr == bare.n_ptr == c.max_tool + c.max_field + c.max_const


def test_a_continuation_forward_pass_produces_the_same_logit_shape():
    c, m = _model()
    m.eval()
    canvas = torch.zeros(1, c.canvas, dtype=torch.long)
    with torch.no_grad():
        a = m(_inputs(c, 0), canvas)
        b = m(_inputs(c, 3), canvas)
    assert a.shape == b.shape == (1, c.canvas,
                                  c.n_kw + c.max_tool + c.max_field
                                  + c.max_const + c.n_reg)
    # the register region is read, not ignored: the same canvas against a
    # different register state must not give identical logits
    assert not torch.allclose(a, b)
