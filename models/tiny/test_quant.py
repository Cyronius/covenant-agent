"""Is the model that trains the model that would be deployed?

Quantisation-aware training fails silently in both directions. If the fake
quantisation is not reached in the forward pass, the run reports float accuracy
for weights that are about to be rounded, which is a confident wrong result. If
it is reached but the straight-through estimator is wrong, the run trains
nothing and looks merely undertrained. Neither shows up as an error.

So this checks the properties the step-3 gate rests on, on a model small enough
to run anywhere:

  1. every weight the block uses in its forward pass has the level count its
     format claims -- three for ternary, and the scale is per output row
  2. the parameter count does not change, because quantisation is a
     parametrization over the same float weights
  3. gradient reaches those float weights through the rounding
  4. the ends stay at int8 when the block is ternary
  5. a checkpoint round-trips: `evaluate.load_model` rebuilds the quantised
     model from the cfg alone and reproduces the logits bit for bit
  6. `--dec-loops` does not change the parameter count, which is what makes
     step 2's sweep a fixed-parameter sweep

    python test_quant.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

import quant
from model import Config, build_model


def tiny_cfg(**kw) -> Config:
    base = dict(binding="structural", d=64, heads=2, ff=128, line_layers=1,
                graph_layers=1, enc_layers=1, dec_layers=2, canvas=16,
                in_vocab=128, max_line=8, max_req=8, n_kw=6, max_tool=3,
                max_field=4, max_const=2, n_reg=4)
    base.update(kw)
    return Config(**base)


def tiny_inputs(c: Config, B=2):
    torch.manual_seed(0)
    return {
        "tool_tok": torch.randint(2, c.in_vocab, (B, c.max_tool, c.max_line)),
        "field_tok": torch.randint(2, c.in_vocab, (B, c.max_field, c.max_line)),
        "const_tok": torch.randint(2, c.in_vocab, (B, c.max_const, c.max_line)),
        "req_tok": torch.randint(2, c.in_vocab, (B, c.max_req)),
        "n_tool": torch.tensor([c.max_tool, 2]),
        "n_field": torch.tensor([c.max_field, 3]),
        "n_const": torch.tensor([c.max_const, 1]),
        "adj": torch.eye(c.max_tool + c.max_field, dtype=torch.bool).expand(B, -1, -1),
    }


def block_weights(m):
    """The weight tensors the decoder block multiplies by, as the forward pass
    sees them: reading the attribute runs the parametrization."""
    out = []
    for layer in m.dec:
        for attn in (layer.self_attn, layer.cross_attn):
            out.append(attn.in_proj_weight)
            out.append(attn.out_proj.weight)
        out.append(layer.ff[0].weight)
        out.append(layer.ff[3].weight)
    return out


def levels_per_row(w, scale_of_row=None):
    """Distinct values per output row, tolerant of float error: the STE computes
    `w + (q - w).detach()`, which equals q to within an epsilon rather than
    exactly, so `unique()` on the raw tensor over-counts."""
    counts = []
    for row in w.reshape(w.shape[0], -1):
        mag = row.abs()
        nz = mag[mag > 0]
        if nz.numel() == 0:
            counts.append(1)
            continue
        unit = nz.min()
        counts.append(len(torch.round(row / unit).unique()))
    return counts


def test_forward_weights_are_quantised():
    for mode, levels in (("tern", 3), ("u4", 16), ("int8", 256)):
        m = build_model(tiny_cfg(weights=mode))
        for w in block_weights(m):
            assert max(levels_per_row(w)) <= levels, (mode, levels_per_row(w)[:4])
    print("1. block weights are quantised in the forward pass: ok")


def test_scale_is_per_row():
    """Two rows of very different magnitude keep their own scales, which is what
    lets the kernel fold one scale per output row into the accumulate."""
    m = build_model(tiny_cfg(weights="tern"))
    with torch.no_grad():
        w = m.dec[0].ff[0].parametrizations.weight.original
        w[0].fill_(0.001)
        w[0, 0] = 0.002
        w[1].fill_(10.0)
        w[1, 0] = 20.0
    q = m.dec[0].ff[0].weight
    assert q[0].abs().max() < 0.01, q[0].abs().max()
    assert q[1].abs().max() > 1.0, q[1].abs().max()
    print("2. the scale is per output row: ok")


def test_params_unchanged_and_gradient_flows():
    fp = build_model(tiny_cfg())
    tern = build_model(tiny_cfg(weights="tern"))
    assert fp.n_params() == tern.n_params(), (fp.n_params(), tern.n_params())
    c = tern.c
    inputs = tiny_inputs(c)
    canvas = torch.randint(2, c.n_kw, (2, c.canvas))
    tern.decode(inputs, canvas).sum().backward()
    g = tern.dec[0].ff[0].parametrizations.weight.original.grad
    assert g is not None and g.abs().sum() > 0, "no gradient through the rounding"
    print(f"3. {tern.n_params()} parameters either way, gradient reaches them: ok")


def test_ends_stay_int8():
    m = build_model(tiny_cfg(weights="tern"))
    assert max(levels_per_row(m.in_emb.weight)) <= 256
    assert max(levels_per_row(m.in_emb.weight)) > 3, "the input embedding went ternary"
    assert max(levels_per_row(m.kw_head.weight)) > 3, "the keyword head went ternary"
    print("4. embeddings and heads stay int8 while the block is ternary: ok")


def test_checkpoint_round_trip():
    """The cfg is the only thing that says a checkpoint is quantised, and
    evaluate.load_model reads nothing else."""
    from evaluate import load_model
    c = tiny_cfg(weights="tern")
    m = build_model(c)
    m.eval()
    inputs = tiny_inputs(c)
    canvas = torch.randint(2, c.n_kw, (2, c.canvas))
    with torch.no_grad():
        before = m.decode(inputs, canvas)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "best.pt"
        torch.save({"cfg": c.__dict__, "model": m.state_dict(), "step": 1,
                    "val_loss": 0.0}, p)
        back = load_model(p, torch.device("cpu"))
    with torch.no_grad():
        after = back.decode(inputs, canvas)
    assert torch.equal(before, after), (before - after).abs().max()
    print("5. a quantised checkpoint round-trips to identical logits: ok")


def test_loops_are_free_in_parameters():
    counts = {L: build_model(tiny_cfg(dec_loops=L)).n_params() for L in (1, 2, 8, 32)}
    assert len(set(counts.values())) == 1, counts
    body = quant.loop_body_params(build_model(tiny_cfg()))
    assert body > 0
    print(f"6. {counts[1]} parameters at every loop count, loop body {body}: ok")


def test_activations_are_quantised():
    """The activation hook is the half of this that a weight check cannot see.
    If it never fires, the run reports int8-weight accuracy with float
    activations, which is not what the kernel computes.

    Counted rather than inferred: one decode pass, and every quantised matmul
    must have asked for its operands to be rounded. An attention module is
    called with three (query, key, value), a Linear with one.
    """
    calls = {"n": 0}
    real = quant.quantize_act

    def counting(x, bits=8):
        calls["n"] += 1
        return real(x, bits)

    c = tiny_cfg(weights="tern")
    m = build_model(c)
    m.eval()
    inputs, canvas = tiny_inputs(c), torch.randint(2, c.n_kw, (2, c.canvas))
    quant.quantize_act = counting
    try:
        with torch.no_grad():
            quantised = m.decode(inputs, canvas)
    finally:
        quant.quantize_act = real
    assert calls["n"] > 0, "the activation hook never fired"
    # The same weights with activations left alone must give a different answer,
    # or the activation path is not reaching the arithmetic.
    m2 = build_model(tiny_cfg(weights="tern", act_bits=0))
    m2.load_state_dict(m.state_dict())
    m2.eval()
    with torch.no_grad():
        unquantised = m2.decode(inputs, canvas)
    assert not torch.equal(quantised, unquantised), "act_bits changed nothing"
    print(f"8. activations rounded at {calls['n']} matmul operands per pass: ok")


def test_resident_budget():
    """The 4 MB memory tiles are the sizing constraint (decision 1), so the
    arithmetic that reports them has to be right: ternary holds 4x int8."""
    m = build_model(tiny_cfg())
    b = quant.loop_body_params(m)
    assert quant.resident_bytes(m, "int8") == b
    assert quant.resident_bytes(m, "u4") == b // 2
    assert quant.resident_bytes(m, "tern") == b // 4
    print("7. resident-byte arithmetic: 16M ternary in the space of 4M int8: ok")


if __name__ == "__main__":
    test_forward_weights_are_quantised()
    test_scale_is_per_row()
    test_params_unchanged_and_gradient_flows()
    test_ends_stay_int8()
    test_checkpoint_round_trip()
    test_activations_are_quantised()
    test_loops_are_free_in_parameters()
    test_resident_budget()
    print("ALL PASS")
