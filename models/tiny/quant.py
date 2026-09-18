"""Quantisation-aware training for the resident block. Step 3 of
`.claude/plans/npu-native-planner.md`.

The NPU holds a stage's weights in 4 MB of memory tiles, so the weight format
decides how many parameters are resident: 4M at int8, 8M at 4-bit, 16M at
ternary. Decision 3 picks ternary and trains it from step zero, because
post-training conversion to ternary fails at this size. This module is the
training side of that: fake quantisation with a straight-through estimator, so
a run reports the accuracy the deployed weights will have rather than the
accuracy of weights that are about to be rounded.

Three formats, one flag (`Config.weights`):

  fp      no quantisation; the baseline every other number is compared against
  int8    symmetric per-output-row, 4M resident
  u4      symmetric per-output-row, 4-bit signed, 8M resident. The part has an
          `mmul<4,16,16,int8,uint4>` that takes packed nibbles straight from
          memory with the unpack folded into the MAC, so this format pays no
          unpack at all
  tern    {-1, 0, +1} with one scale per output row, 16M resident, and a
          software unpack of 0.0156 cycles per weight

`kernels/tern_mk/README.md` measured those unpack costs and contradicts
decision 3's claim that ternary strictly dominates 4-bit: on this silicon 4-bit
gets its unpack free in hardware and ternary does not, so the choice is 2x the
resident parameters against a 12.5% tax at a 64-slot canvas. That is a capacity
question, which is what this module exists to answer, and it is why `u4` is
here as a first-class arm rather than as int8's fallback.

What is quantised, and what is not:

  the stacks   every attention and feed-forward weight in the line encoder, the
               graph pass, the turn encoder and the looped decoder block --
               everything that is a matmul inside a transformer layer
  the ends     input embedding, keyword embedding, register embedding, tag
               embedding, the keyword head and the two pointer projections:
               int8, never ternary. Decision 3 keeps the ends at int8 because
               the layer-influence probe on the 0.8B found the ends move the
               residual stream far more than the middle. In this architecture
               the ends are the embedding side and the head side; the looped
               block has no distinguished first or last layer, because the same
               weights are every iteration.
  norms        left in floating point. A LayerNorm is a per-channel affine with
               no matmul, its parameters are 2d per layer, and quantising it
               buys no resident parameters. Decision 3's "norms stay int8" is
               about the ends being high precision, which this satisfies.
  activations  int8, per token (`act_bits`), at the input of every quantised
               matmul. This is the native 512-MAC-per-issue path: the weights
               become sign and routing and the activations carry the precision.

The quantisation is a `torch.nn.utils.parametrize` parametrization on the weight
tensor, which is what lets it cover `nn.MultiheadAttention`: that module keeps
its projections in one `in_proj_weight` tensor rather than in `nn.Linear`
children, so a module-swapping scheme would have to reimplement attention.
A parametrization also means the checkpoint keeps the master float weights
(`...parametrizations.weight.original`), so a run can be resumed and the
rounding is never baked in twice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

MODES = ("fp", "int8", "u4", "tern")
BITS = {"fp": 32, "int8": 8, "u4": 4, "tern": 2}   # 2, not log2(3): see decision 3


def _ste(w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward sees the rounded weight, backward
    sees the identity, so the master float weight keeps receiving gradient."""
    return w + (q - w).detach()


def rounded_weight(w: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The integer codes and the per-row scale, with no estimator attached.

    Returns `(codes, scale)` such that `codes * scale` is the dequantised
    weight: codes in {-1, 0, 1} for ternary, in [-8, 7] for u4, in [-128, 127]
    for int8. This is what a packer writes into the kernel's weight pool, and
    what the tests below check the level count of.
    """
    flat = w.reshape(w.shape[0], -1)
    if mode == "tern":
        # BitNet b1.58's absmean: the scale is the row's mean magnitude, so
        # weights below half of it round to the exact zero. Keeping that zero is
        # most of why ternary trains, and it is why the fourth 2-bit code is
        # left unused (decision 3, Rejected).
        s = flat.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        codes = (flat / s).round().clamp(-1, 1)
    else:
        qmax = 127.0 if mode == "int8" else 7.0
        s = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / qmax
        codes = (flat / s).round().clamp(-qmax - 1, qmax)
    return codes.reshape(w.shape), s


def quantize_weight(w: torch.Tensor, mode: str) -> torch.Tensor:
    """Fake-quantise a weight, one scale per output row (dim 0).

    A per-row scale is what the kernel can fold into the int32 accumulate for
    free, and it is the finest granularity that costs nothing at run time.
    """
    if mode == "fp":
        return w
    codes, s = rounded_weight(w, mode)
    return _ste(w, (codes.reshape(w.shape[0], -1) * s).reshape(w.shape))


def quantize_act(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Fake-quantise an activation, one scale per token (the last dim is the
    channel). Per-token scaling is what the NPU path does: a token's scale is
    computed as it is written and applied in the accumulate."""
    if not bits:
        return x
    qmax = float(2 ** (bits - 1) - 1)
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / qmax
    return _ste(x, (x / s).round().clamp(-qmax - 1, qmax) * s)


class WeightQuant(nn.Module):
    """A weight parametrization. `right_inverse` is the identity, so assigning
    a float weight to a parametrized module stores it unchanged."""

    def __init__(self, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError("unknown weight mode " + repr(mode))
        self.mode = mode

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return quantize_weight(w, self.mode)

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        return w


def _act_hook(bits: int):
    def hook(module, args, kwargs):
        if not args:
            return None
        # nn.MultiheadAttention is called with (query, key, value); a Linear
        # with one input. Quantise every floating tensor argument, which is
        # exactly the operands of the matmuls this module is about to do.
        new = tuple(quantize_act(a, bits)
                    if torch.is_tensor(a) and a.is_floating_point() else a
                    for a in args)
        return new, kwargs
    return hook


def _quantize_module(m: nn.Module, mode: str, act_bits: int, seen: set) -> int:
    """Parametrise every weight tensor `m` owns directly. Returns the number of
    quantised parameters."""
    if isinstance(m, nn.Linear):
        names = ["weight"]
    elif isinstance(m, nn.MultiheadAttention):
        # out_proj is an nn.Linear child and is reached on its own.
        names = [k for k in ("in_proj_weight", "q_proj_weight", "k_proj_weight",
                             "v_proj_weight") if getattr(m, k, None) is not None]
    else:
        return 0
    if id(m) in seen:
        return 0
    seen.add(id(m))
    n = 0
    for name in names:
        if mode != "fp":
            P.register_parametrization(m, name, WeightQuant(mode), unsafe=True)
        n += getattr(m, name).numel()
    if act_bits and mode != "fp":
        m.register_forward_pre_hook(_act_hook(act_bits), with_kwargs=True)
    return n


def quantize_tree(root: nn.Module, mode: str, act_bits: int,
                  seen: set | None = None) -> int:
    """Quantise every matmul weight under `root`. Returns parameters covered."""
    seen = seen if seen is not None else set()
    n = 0
    for m in root.modules():
        n += _quantize_module(m, mode, act_bits, seen)
    return n


STACKS = ("line_enc", "graph", "turn", "dec", "enc")
ENDS = ("in_emb", "kw_emb", "reg_emb", "tag_emb", "kw_head", "q", "k", "out_head")


def apply_quant(model: nn.Module, weights: str, act_bits: int = 8,
                ends: str = "int8") -> dict:
    """Quantise a built model in place; returns a report for the run config.

    `weights` is the format of every transformer stack, including the looped
    block. `ends` is the format of the embeddings and the output heads, which
    decision 3 keeps at int8 whatever the block does.
    """
    if weights == "fp" and ends == "fp":
        return {"weights": "fp", "act_bits": 0, "ends": "fp",
                "quant_block_params": 0, "quant_end_params": 0}
    seen: set = set()
    block = 0
    for name in STACKS:
        s = getattr(model, name, None)
        if s is not None:
            block += quantize_tree(s, weights, act_bits, seen)
    end = 0
    for name in ENDS:
        m = getattr(model, name, None)
        if m is None:
            continue
        if isinstance(m, nn.Embedding):
            if ends != "fp":
                P.register_parametrization(m, "weight", WeightQuant(ends), unsafe=True)
            end += m.weight.numel()
        else:
            end += quantize_tree(m, ends, act_bits, seen)
    return {"weights": weights, "act_bits": act_bits, "ends": ends,
            "quant_block_params": block, "quant_end_params": end}


def loop_body_params(model: nn.Module) -> int:
    """Parameters in the part that is reapplied per loop: the decoder stack.

    This is the number the 4 MB memory-tile budget applies to (decision 1):
    4M at int8, 8M at 4-bit, 16M at ternary. It is not the model's parameter
    count, because the encoders run once per turn or once per world.
    """
    dec = getattr(model, "dec", None)
    if dec is None:
        return 0
    return sum(p.numel() for p in dec.parameters())


def resident_bytes(model: nn.Module, weights: str) -> int:
    """Bytes the loop body occupies on chip in the given format."""
    return loop_body_params(model) * BITS[weights] // 8


if __name__ == "__main__":
    torch.manual_seed(0)
    # Every format rounds to the number of levels it claims, per output row.
    w = torch.randn(4, 64)
    for mode, levels in (("tern", 3), ("u4", 16), ("int8", 256)):
        codes, _ = rounded_weight(w, mode)
        per_row = [len(codes[r].unique()) for r in range(4)]
        assert max(per_row) <= levels, (mode, per_row)
        err = (quantize_weight(w, mode) - w).abs().mean() / w.abs().mean()
        print(f"{mode:5s} levels/row {per_row} relative error {err:.3f}")
    # The gradient reaches the float weight through the rounding.
    v = torch.randn(4, 8, requires_grad=True)
    quantize_weight(v, "tern").sum().backward()
    assert v.grad is not None and v.grad.abs().sum() > 0
    # Activations round per token, not per tensor: a row 100x larger than its
    # neighbour keeps its own resolution.
    x = torch.tensor([[[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]]])
    qx = quantize_act(x, 8)
    assert torch.allclose(qx / x, torch.ones_like(x), atol=0.02), qx
    # A parametrised MultiheadAttention still runs, and its weight is ternary.
    mha = nn.MultiheadAttention(16, 2, batch_first=True)
    n = quantize_tree(mha, "tern", 8)
    assert n == 16 * 16 * 3 + 16 * 16, n
    h = torch.randn(2, 5, 16)
    out, _ = mha(h, h, h, need_weights=False)
    assert out.shape == h.shape
    codes, _ = rounded_weight(mha.parametrizations.in_proj_weight.original, "tern")
    assert len(codes.unique()) <= 3, codes.unique()
    print("quant.py self-test ok")
