"""Encoder-decoder over a fixed-size canvas of program slots.

Two bindings, one flag (`Config.binding`):

  flat        `CanvasModel`, the R3 model. One transformer over the whole
              serialized context (about 1,100 tokens); an output head with a
              row per symbol.
  structural  `StructuralModel`, decisions 6 and 7 of
              `.claude/plans/npu-native-planner.md`. A line encoder turns each
              tool, field and constant line into one vector; a sparse graph
              pass lets a tool see its own fields and a field its entity and
              tools; a turn encoder runs over those vectors plus the request
              tokens. The result is region A, a tagged sequence of context
              vectors. A canvas slot is either a keyword (a fixed head of a
              few dozen rows) or a pointer: a dot product against the tool,
              field and constant vectors of THIS task, or a register
              embedding. There is no `T23` row. A symbol the task does not
              declare has no vector and cannot be produced.

Within each binding, one class serves both arms of the experiment. The only
difference is whether decoder self-attention is causal:

  causal=False  the diffusion arm. Input is the canvas with some slots masked,
                output is a prediction for every slot, and every slot can see
                every other one.
  causal=True   the control. Input is the target shifted right, output is the
                next token, and a slot can only see what came before it.

Everything else -- the encoder, the cross-attention, the widths -- is shared,
so a difference in results is a difference in how the output is produced.
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
    pointer: bool = False    # flat: symbol slots decided by pointing at the input
    dec_loops: int = 1          # knob A: apply the decoder stack this many times
    # structural binding
    binding: str = "flat"
    line_layers: int = 2        # per-line encoder depth
    graph_layers: int = 1       # sparse schema-graph pass depth
    in_pad: int = 1             # the input tokenizer's <pad> id
    max_line: int = 64          # tokens per schema line
    max_req: int = 128          # request tokens
    n_kw: int = 62              # keyword rows
    max_tool: int = 18
    max_field: int = 23
    max_const: int = 10
    n_reg: int = 32             # r0..r15 then r0...r15. (canvas.Layout)


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

    def forward(self, x, pad_mask, attn_mask=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, attn_mask=attn_mask,
                         need_weights=False)
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


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


class CanvasModel(nn.Module):
    """The flat binding: the R3 model, unchanged."""

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
        self.apply(_init)

    def set_symbol_ids(self, ids) -> None:
        """Mark which output-vocabulary entries are per-request symbols."""
        m = torch.zeros(self.c.out_vocab, dtype=torch.bool)
        m[list(ids)] = True
        self.symbol_ids.copy_(m)

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

    # The sampler protocol shared with StructuralModel: `inputs` is a dict of
    # tensors for one batch, whatever the binding needs.
    def encode_inputs(self, inputs: dict):
        return self.encode(inputs["src"], inputs["pad"])

    def decode(self, inputs: dict, canvas, mem=None, loops=None):
        return self.forward(inputs["src"], inputs["pad"], canvas, loops=loops, mem=mem,
                            sym=inputs.get("sym"))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Kind tags for region A and the canvas (decision 4).
TAG_TOOL, TAG_FIELD, TAG_CONST, TAG_REQUEST, TAG_CANVAS = range(5)


class Memory:
    """Region A: the context vectors, their pad mask, and the pointer bank."""

    def __init__(self, mem, pad, n_ptr):
        self.mem, self.pad = mem, pad
        self.n_ptr = n_ptr                 # tools + fields + consts come first

    @property
    def pointers(self):
        return self.mem[:, :self.n_ptr]


class StructuralModel(nn.Module):
    """The structural binding. See the module docstring.

    Input dict (from prep.py's structural cache, one batch, on device):
      tool_tok  (B, MT, TL)   field_tok (B, MF, TL)   const_tok (B, MC, TL)
      req_tok   (B, RL)       n_tool/n_field/n_const (B,)
      adj       (B, MT+MF, MT+MF) bool, True where a line may attend to another
      kw_allowed (B, K) bool, optional: the grammar's keyword half for this task
    Canvas ids are joint ids over [keywords | tools | fields | consts | registers]
    (canvas.Layout).
    """

    def __init__(self, c: Config):
        super().__init__()
        assert c.binding == "structural"
        self.c = c
        d = c.d
        self.in_emb = nn.Embedding(c.in_vocab, d)
        self.kw_emb = nn.Embedding(c.n_kw, d)           # canvas input for keyword slots
        self.kw_head = nn.Linear(d, c.n_kw)              # the fixed keyword output rows
        self.reg_emb = nn.Embedding(c.n_reg, d)          # r0..r15, r0...r15.
        self.tag_emb = nn.Embedding(5, d)
        self.cls = nn.Parameter(torch.zeros(d))          # the line-pooling token
        self.register_buffer("line_pos", sinusoids(c.max_line + 1, d), persistent=False)
        self.register_buffer("req_pos", sinusoids(c.max_req, d), persistent=False)
        self.register_buffer("out_pos", sinusoids(c.canvas, d), persistent=False)
        self.line_enc = nn.ModuleList([EncoderLayer(c) for _ in range(c.line_layers)])
        self.line_norm = nn.LayerNorm(d)
        self.graph = nn.ModuleList([EncoderLayer(c) for _ in range(c.graph_layers)])
        self.graph_norm = nn.LayerNorm(d)
        self.turn = nn.ModuleList([EncoderLayer(c) for _ in range(c.enc_layers)])
        self.enc_norm = nn.LayerNorm(d)
        self.dec = nn.ModuleList([DecoderLayer(c) for _ in range(c.dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.slot_norm = nn.LayerNorm(d)                 # one scale for keyword and pointer inputs
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)   # pointer projections
        self.drop = nn.Dropout(c.dropout)
        if c.causal:
            m = torch.triu(torch.full((c.canvas, c.canvas), float("-inf")), diagonal=1)
            self.register_buffer("causal_mask", m, persistent=False)
        else:
            self.causal_mask = None
        self.apply(_init)
        nn.init.normal_(self.cls, std=0.02)

    # -- per-world: cacheable per decision 6 -----------------------------

    def encode_lines(self, tok: torch.Tensor, tag: int) -> torch.Tensor:
        """(B, L, T) token ids -> (B, L, d): one vector per line, from a CLS
        token that attends over the line's tokens."""
        B, L, T = tok.shape
        flat = tok.reshape(B * L, T).long()
        pad = flat == self.c.in_pad
        x = self.in_emb(flat) * math.sqrt(self.c.d) + self.line_pos[1:T + 1]
        cls = (self.cls + self.line_pos[0] + self.tag_emb.weight[tag]).expand(B * L, 1, -1)
        x = torch.cat([cls, x], 1)
        pad = torch.cat([torch.zeros(B * L, 1, dtype=torch.bool, device=pad.device), pad], 1)
        x = self.drop(x)
        for layer in self.line_enc:
            x = layer(x, pad)
        return self.line_norm(x[:, 0]).reshape(B, L, -1)

    def encode_world(self, tool_tok, field_tok, adj):
        """Line vectors for every tool and field, then the sparse graph pass.
        Depends only on the world's schema, so it can be computed once per
        world and shipped as data; nothing here reads the request."""
        tv = self.encode_lines(tool_tok, TAG_TOOL)
        fv = self.encode_lines(field_tok, TAG_FIELD)
        x = torch.cat([tv, fv], 1)                                   # (B, MT+MF, d)
        B, L, _ = x.shape
        # nn.MultiheadAttention takes (B*heads, L, L) with True = blocked.
        # Padded rows attend to themselves (adj has self edges everywhere), so
        # no row is fully masked and nothing goes NaN.
        blocked = (~adj).repeat_interleave(self.c.heads, dim=0)
        for layer in self.graph:
            x = layer(x, None, blocked)
        x = self.graph_norm(x)
        return x[:, :tv.size(1)], x[:, tv.size(1):]

    # -- per-turn -------------------------------------------------------

    def encode_turn(self, tool_vecs, field_vecs, const_tok, req_tok,
                    n_tool, n_field, n_const) -> Memory:
        """Region A: tools, fields, constants, request tokens, each tagged."""
        c = self.c
        B = tool_vecs.size(0)
        cv = self.encode_lines(const_tok, TAG_CONST)
        req = req_tok.long()
        rq = (self.in_emb(req) * math.sqrt(c.d) + self.req_pos[:req.size(1)]
              + self.tag_emb.weight[TAG_REQUEST])
        x = torch.cat([tool_vecs + self.tag_emb.weight[TAG_TOOL],
                       field_vecs + self.tag_emb.weight[TAG_FIELD],
                       cv + self.tag_emb.weight[TAG_CONST], rq], 1)
        dev = x.device
        pad = torch.cat([
            torch.arange(tool_vecs.size(1), device=dev)[None] >= n_tool.long()[:, None],
            torch.arange(field_vecs.size(1), device=dev)[None] >= n_field.long()[:, None],
            torch.arange(cv.size(1), device=dev)[None] >= n_const.long()[:, None],
            req == c.in_pad], 1)
        x = self.drop(x)
        for layer in self.turn:
            x = layer(x, pad)
        n_ptr = tool_vecs.size(1) + field_vecs.size(1) + cv.size(1)
        return Memory(self.enc_norm(x), pad, n_ptr)

    def encode_inputs(self, inputs: dict) -> Memory:
        tv, fv = self.encode_world(inputs["tool_tok"], inputs["field_tok"], inputs["adj"])
        return self.encode_turn(tv, fv, inputs["const_tok"], inputs["req_tok"],
                                inputs["n_tool"], inputs["n_field"], inputs["n_const"])

    # -- the canvas ------------------------------------------------------

    def present_mask(self, inputs: dict, device) -> torch.Tensor:
        """(B, J) True where a joint id exists for this task. Keyword rows obey
        `kw_allowed` when given; MASK is never an output; pointer rows exist
        only up to the task's declared counts; registers always exist."""
        c = self.c
        B = inputs["n_tool"].size(0)
        kw = torch.ones(B, c.n_kw, dtype=torch.bool, device=device)
        if inputs.get("kw_allowed") is not None:
            kw &= inputs["kw_allowed"].to(device)
        kw[:, 1] = False                                          # MASK
        ar = torch.arange
        present = torch.cat([
            kw,
            ar(c.max_tool, device=device)[None] < inputs["n_tool"].long()[:, None],
            ar(c.max_field, device=device)[None] < inputs["n_field"].long()[:, None],
            ar(c.max_const, device=device)[None] < inputs["n_const"].long()[:, None],
            torch.ones(B, c.n_reg, dtype=torch.bool, device=device)], 1)
        return present

    def slot_table(self, mem: Memory) -> torch.Tensor:
        """(B, J, d): the input vector for every joint id. Keyword rows are a
        fixed embedding; pointer rows ARE the region-A vectors of this task;
        register rows are a fixed embedding. Binding is exact by construction:
        the vector a canvas slot holds for T8 is T8's line vector."""
        B = mem.mem.size(0)
        table = torch.cat([self.kw_emb.weight.expand(B, -1, -1), mem.pointers,
                           self.reg_emb.weight.expand(B, -1, -1)], 1)
        return self.slot_norm(table)

    def decode(self, inputs: dict, canvas, mem: Memory | None = None, loops=None):
        """Logits (B, C, J) over keywords and pointers, -inf where a target
        does not exist for this task."""
        c = self.c
        if mem is None:
            mem = self.encode_inputs(inputs)
        table = self.slot_table(mem)                                     # (B, J, d)
        x = table.gather(1, canvas.long().unsqueeze(-1).expand(-1, -1, c.d))
        x = self.drop(x + self.out_pos + self.tag_emb.weight[TAG_CANVAS])
        for _ in range(loops or c.dec_loops):
            for layer in self.dec:
                x = layer(x, mem.mem, mem.pad, self.causal_mask)
        h = self.dec_norm(x)
        kw_logits = self.kw_head(h)                                      # (B, C, K)
        keys = self.k(table[:, c.n_kw:])                                 # (B, J-K, d)
        ptr_logits = torch.einsum("bcd,bjd->bcj", self.q(h), keys) / math.sqrt(c.d)
        logits = torch.cat([kw_logits, ptr_logits], -1)
        present = self.present_mask(inputs, logits.device)
        return logits.masked_fill(~present.unsqueeze(1), float("-inf"))

    def forward(self, inputs: dict, canvas, loops=None, mem=None):
        return self.decode(inputs, canvas, mem=mem, loops=loops)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(c: Config) -> nn.Module:
    return StructuralModel(c) if c.binding == "structural" else CanvasModel(c)


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
        print(f"flat causal={causal}: {m.n_params()/1e6:.2f}M params, logits {tuple(out.shape)}, "
              f"masked {loss_mask.float().mean():.2f}")
    for causal in (False, True):
        c = Config(causal=causal, binding="structural", enc_layers=2)
        m = StructuralModel(c)
        B = 2
        inputs = {
            "tool_tok": torch.randint(2, c.in_vocab, (B, c.max_tool, c.max_line)),
            "field_tok": torch.randint(2, c.in_vocab, (B, c.max_field, c.max_line)),
            "const_tok": torch.randint(2, c.in_vocab, (B, c.max_const, c.max_line)),
            "req_tok": torch.randint(2, c.in_vocab, (B, c.max_req)),
            "n_tool": torch.tensor([18, 10]), "n_field": torch.tensor([23, 14]),
            "n_const": torch.tensor([5, 3]),
            "adj": torch.eye(c.max_tool + c.max_field, dtype=torch.bool).expand(B, -1, -1),
        }
        J = c.n_kw + c.max_tool + c.max_field + c.max_const + c.n_reg
        tgt = torch.randint(2, c.n_kw, (B, c.canvas))
        canvas, loss_mask, ratio = mask_canvas(tgt, mask_id=1)
        out = m(inputs, canvas if not causal else tgt)
        assert out.shape == (B, c.canvas, J)
        assert torch.isfinite(out[0, :, 2:c.n_kw + 18]).all()     # MASK (id 1) is always -inf
        assert torch.isinf(out[1, :, c.n_kw + 10]).all()          # tool 10 absent in row 1
        print(f"structural causal={causal}: {m.n_params()/1e6:.2f}M params, "
              f"logits {tuple(out.shape)}")
