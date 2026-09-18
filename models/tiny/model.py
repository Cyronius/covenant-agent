"""Encoder-decoder over a fixed-size canvas of program slots.

One class serves both arms of the experiment. The only difference is whether
decoder self-attention is causal:

  causal=False  the diffusion arm. Input is the canvas with some slots masked,
                output is a prediction for every slot, and every slot can see
                every other one.
  causal=True   the control. Input is the target shifted right, output is the
                next token, and a slot can only see what came before it.

Everything else -- the encoder, the cross-attention, the widths, the parameter
count -- is shared, so a difference in results is a difference in how the
output is produced and not in model capacity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    in_vocab: int = 4096
    out_vocab: int = 512
    d: int = 256
    heads: int = 4
    ff: int = 1024
    enc_layers: int = 3
    dec_layers: int = 4
    max_in: int = 1280
    canvas: int = 64
    dropout: float = 0.1
    causal: bool = False
    pointer: bool = False    # symbol slots decided by pointing at the input
    dec_loops: int = 1          # knob A: apply the decoder stack this many times


def sinusoids(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1).float()
    i = torch.arange(0, d, 2).float()
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class EncoderLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.attn = nn.MultiheadAttention(c.d, c.heads, dropout=c.dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.ff = nn.Sequential(nn.Linear(c.d, c.ff), nn.GELU(),
                                nn.Dropout(c.dropout), nn.Linear(c.ff, c.d))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, pad_mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(c.d, c.heads, dropout=c.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(c.d, c.heads, dropout=c.dropout, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(c.d), nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.ff = nn.Sequential(nn.Linear(c.d, c.ff), nn.GELU(),
                                nn.Dropout(c.dropout), nn.Linear(c.ff, c.d))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mem, mem_pad, causal_mask):
        h = self.n1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.n2(x)
        a, _ = self.cross_attn(h, mem, mem, key_padding_mask=mem_pad, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n3(x)))


class CanvasModel(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.in_emb = nn.Embedding(c.in_vocab, c.d)
        self.out_emb = nn.Embedding(c.out_vocab, c.d)
        self.register_buffer("in_pos", sinusoids(c.max_in, c.d), persistent=False)
        self.register_buffer("out_pos", sinusoids(c.canvas, c.d), persistent=False)
        self.enc = nn.ModuleList([EncoderLayer(c) for _ in range(c.enc_layers)])
        self.dec = nn.ModuleList([DecoderLayer(c) for _ in range(c.dec_layers)])
        self.enc_norm, self.dec_norm = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.out_vocab)
        self.drop = nn.Dropout(c.dropout)
        if c.causal:
            m = torch.triu(torch.full((c.canvas, c.canvas), float("-inf")), diagonal=1)
            self.register_buffer("causal_mask", m, persistent=False)
        else:
            self.causal_mask = None
        # Which vocabulary entries are per-request symbols. Fixed by the
        # vocabulary, so it is a buffer rather than something derived per batch.
        # Not persistent: it is derived from the vocabulary, not learned, and
        # keeping it out of the state dict means checkpoints saved before the
        # pointer head existed still load.
        self.register_buffer("symbol_ids", torch.zeros(c.out_vocab, dtype=torch.bool),
                             persistent=False)
        self.apply(self._init)

    def set_symbol_ids(self, ids) -> None:
        """Mark which output-vocabulary entries are per-request symbols."""
        m = torch.zeros(self.c.out_vocab, dtype=torch.bool)
        m[list(ids)] = True
        self.symbol_ids.copy_(m)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, src, src_pad):
        x = self.drop(self.in_emb(src) * math.sqrt(self.c.d) + self.in_pos[:src.size(1)])
        for layer in self.enc:
            x = layer(x, src_pad)
        return self.enc_norm(x)

    def point(self, x, mem, sym):
        """Score each canvas slot against where each symbol is declared.

        `sym` gives, per output-vocabulary id, the input token index where that
        symbol is declared, or -1 if it is not a per-request symbol or is absent
        from this task.

        Two things fall out, and both matter. A symbol's logit now comes from
        reading its own declaration line rather than from an embedding that has
        to mean something different in every task. And a symbol this task does
        not declare is driven to -inf, so naming a tool that does not exist
        stops being possible rather than merely unlikely.
        """
        present = sym >= 0                                   # (B, V)
        idx = sym.clamp(min=0)
        vec = mem.gather(1, idx.unsqueeze(-1).expand(-1, -1, mem.size(-1)))   # (B,V,d)
        scores = torch.einsum("bcd,bvd->bcv", x, vec) / math.sqrt(x.size(-1))
        return scores, present

    def forward(self, src, src_pad, canvas, loops: int | None = None, mem=None,
                sym=None):
        """src: (B, S) input ids. src_pad: (B, S) True where padding.
        canvas: (B, C) output ids -- masked canvas, or shifted target for the
        control arm. Returns logits (B, C, out_vocab).

        `mem` is a pre-computed encoder output. The input runs about 1100 tokens
        against a 64-slot canvas, so the encoder dominates the cost of a pass and
        re-running it per sampling step would swamp the thing being measured.
        Both arms cache it once per example, which leaves the pass count
        measuring what actually differs between them: decoder work.
        """
        if mem is None:
            mem = self.encode(src, src_pad)
        x = self.drop(self.out_emb(canvas) * math.sqrt(self.c.d) + self.out_pos)
        # Looping the stack is knob A: more compute at the same parameter count.
        for _ in range(loops or self.c.dec_loops):
            for layer in self.dec:
                x = layer(x, mem, src_pad, self.causal_mask)
        h = self.dec_norm(x)
        logits = self.head(h)
        if sym is not None and self.c.pointer:
            scores, present = self.point(h, mem, sym)
            # Symbol slots are decided by pointing; everything else (keywords,
            # registers, newlines) keeps the ordinary vocabulary head, because
            # those tokens do mean the same thing in every task.
            here = present.unsqueeze(1)                       # (B,1,V) declared here
            logits = torch.where(here, scores, logits)
            # A vocabulary entry that is a per-request symbol but is not declared
            # by *this* task cannot be the answer. Which ids are symbols at all is
            # a fixed property of the vocabulary, not of the batch.
            absent = self.symbol_ids.view(1, 1, -1) & ~here
            logits = logits.masked_fill(absent, float("-inf"))
        return logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def mask_canvas(target: torch.Tensor, mask_id: int, generator=None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The masked-diffusion training objective.

    Draw one mask ratio per example uniformly from (0, 1], then hide that
    fraction of slots. A ratio near 1 teaches generation from nothing; a ratio
    near 0 teaches repair of an almost-finished program. Training across the
    whole range in proportion is what makes one network serve every step of the
    sampler. Padding slots are masked too, so the model learns where a program
    ends rather than being told.
    """
    b, n = target.shape
    ratio = torch.rand(b, 1, generator=generator, device=target.device).clamp(min=1e-3)
    keep = torch.rand(b, n, generator=generator, device=target.device) >= ratio
    # Guarantee at least one masked slot per row, or the row has no loss term.
    empty = keep.all(dim=1)
    if empty.any():
        keep[empty, 0] = False
    canvas = torch.where(keep, target, torch.full_like(target, mask_id))
    return canvas, ~keep, ratio.squeeze(1)


if __name__ == "__main__":
    for causal in (False, True):
        c = Config(causal=causal)
        m = CanvasModel(c)
        src = torch.randint(0, c.in_vocab, (2, 900))
        pad = torch.zeros(2, 900, dtype=torch.bool)
        pad[:, 800:] = True
        tgt = torch.randint(2, c.out_vocab, (2, c.canvas))
        canvas, loss_mask, ratio = mask_canvas(tgt, mask_id=1)
        out = m(src, pad, canvas if not causal else tgt)
        print(f"causal={causal}: {m.n_params()/1e6:.2f}M params, logits {tuple(out.shape)}, "
              f"masked {loss_mask.float().mean():.2f}")
