"""The split encoder's parts (.claude/plans/description-reading.md §3).

A tool line used to be one token row squeezed into one vector by the line
encoder, and twins -- tools whose signatures are identical -- came out of it
at cosine 0.93-0.97, using about 4 of 128 dimensions (results/R10.md §9).
Every training tool had a unique signature, so the signature always sufficed
and the description was squeezed out. The teacher agrees that mixing parts
hurts even a good encoder: bge-small reads the description alone at 65.0%
on the twin benchmark and the whole line at 56.1% (§9b).

So each part of the line gets its own small stage:

  signature   types, field links, effect -- trained by the program loss
  description the description text        -- pretrained on "which tool does
  name        the tool name               -- this request mean" (step 3)

The description and name stages also read the *request*, with a query
marker, as E5 and BGE are trained: one encoder for both sides is what puts a
request and the description it means in the same space.

`TextStage` is one such encoder. `packed_encode` is the unpadding every line
encoder shares (88% of line-encoder positions were padding): only occupied
line slots run, bucketed by length, and the result is scattered back so
downstream shapes never change. The losses at the bottom are the two the
plan keeps on from pretraining into planner training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EncoderLayer, Config, sinusoids, _init


# ---------------------------------------------------------------- packing
def packed_encode(tok: torch.Tensor, pad_id: int, run, buckets: int = 4
                  ) -> torch.Tensor:
    """(B, L, T) token rows -> (B, L, d) via `run(tok_rows (n, t), pad (n, t))
    -> (n, d)`, running only rows that hold a token, each length bucket
    trimmed to its own longest row. Empty rows come back as zeros.

    Identical to running every row at full width, up to float error: a row's
    tokens are left-aligned, positions depend only on the index, and a
    padded key contributes exactly nothing to attention (packed == padded is
    `test_stages.py`'s check). Empty rows are the only difference -- a padded
    line slot used to get whatever an all-pad row pools to -- and every
    reader downstream masks them out (the turn encoder's pad mask, the graph
    pass's self-only edges, the pointer's present mask).
    """
    B, L, T = tok.shape
    flat = tok.reshape(B * L, T).long()
    lens = (flat != pad_id).sum(1)
    live = torch.nonzero(lens > 0).squeeze(1)
    out = None
    if live.numel():
        order = live[torch.argsort(lens[live])]
        chunks = torch.chunk(order, max(1, min(buckets, order.numel())))
        for idx in chunks:
            t = int(lens[idx].max())
            rows = flat[idx, :t]
            v = run(rows, rows == pad_id)
            if out is None:
                out = v.new_zeros(B * L, v.size(-1))
            out[idx] = v
    if out is None:
        d = run(flat[:1, :1], torch.ones(1, 1, dtype=torch.bool,
                                         device=flat.device)).size(-1)
        out = torch.zeros(B * L, d, device=tok.device)
    return out.reshape(B, L, -1)


# ---------------------------------------------------------------- a stage
class TextStage(nn.Module):
    """A small transformer that turns one token row into one vector.

    Its own token table at its own width (host-side lookups on the NPU), a
    kind embedding for document vs query, and either a summary token or
    mean pooling. `forward(tok (N, T), query)` -> (N, width).
    """

    def __init__(self, vocab: int, width: int, layers: int, max_len: int,
                 pad_id: int, heads: int = 4, dropout: float = 0.1,
                 pool: str = "cls"):
        super().__init__()
        self.width, self.pad_id, self.pool = width, pad_id, pool
        c = Config(d=width, heads=heads, ff=4 * width, dropout=dropout)
        self.emb = nn.Embedding(vocab, width)
        self.kind = nn.Embedding(2, width)             # 0 document, 1 query
        self.cls = nn.Parameter(torch.zeros(width))
        self.register_buffer("pos", sinusoids(max_len + 1, width), persistent=False)
        self.layers = nn.ModuleList([EncoderLayer(c) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)
        self.apply(_init)
        nn.init.normal_(self.cls, std=0.02)

    def run(self, tok: torch.Tensor, pad: torch.Tensor, query: bool = False
            ) -> torch.Tensor:
        N, T = tok.shape
        if T > self.pos.size(0) - 1:
            raise ValueError(f"row of {T} tokens, stage built for "
                             f"{self.pos.size(0) - 1}")
        x = self.emb(tok.long()) * math.sqrt(self.width) + self.pos[1:T + 1]
        x = x + self.kind.weight[int(query)]
        cls = (self.cls + self.pos[0] + self.kind.weight[int(query)]).expand(N, 1, -1)
        x = self.drop(torch.cat([cls, x], 1))
        pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=pad.device),
                         pad], 1)
        for layer in self.layers:
            x = layer(x, pad)
        x = self.norm(x)
        if self.pool == "mean":
            keep = (~pad[:, 1:]).unsqueeze(-1).float()
            return (x[:, 1:] * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return x[:, 0]

    def forward(self, tok: torch.Tensor, query: bool = False,
                packed: bool = True) -> torch.Tensor:
        """(N, T) -> (N, width); (B, L, T) -> (B, L, width)."""
        three = tok.dim() == 3
        t3 = tok if three else tok.unsqueeze(1)
        if packed:
            out = packed_encode(t3, self.pad_id,
                                lambda r, p: self.run(r, p, query))
        else:
            B, L, T = t3.shape
            flat = t3.reshape(B * L, T)
            out = self.run(flat, flat == self.pad_id, query).reshape(B, L, -1)
        return out if three else out.squeeze(1)


class ReadHead(nn.Module):
    """The description and name stages side by side, scored against a
    request, with a per-candidate gate between them: the pretraining model
    (`stage_pretrain.py`) and the twin benchmark's reader.

    score_j = g_j * cos(q_desc, desc_j) + (1 - g_j) * cos(q_name, name_j),
    all times a learned temperature. g_j reads the query and candidate j's
    name vector, which is how it can learn to lean on the description when
    the names say nothing (`foo17`).
    """

    def __init__(self, desc: TextStage, name: TextStage | None):
        super().__init__()
        self.desc, self.name = desc, name
        self.scale = nn.Parameter(torch.tensor(math.log(1 / 0.05)))
        if name is not None:
            self.gate = nn.Sequential(nn.Linear(desc.width + name.width, 64),
                                      nn.GELU(), nn.Linear(64, 1))

    def encode(self, req_tok, desc_tok, name_tok=None):
        q_d = self.desc(req_tok, query=True)                 # (B, wd)
        d = self.desc(desc_tok)                              # (B, M, wd)
        out = {"q_desc": q_d, "desc": d}
        if self.name is not None and name_tok is not None:
            out["q_name"] = self.name(req_tok, query=True)
            out["name"] = self.name(name_tok)
        return out

    def scores(self, enc: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(B, M) logits and the gate (B, M) or None."""
        s = self.scale.exp().clamp(max=100)
        sd = torch.einsum("bw,bmw->bm", F.normalize(enc["q_desc"], dim=-1),
                          F.normalize(enc["desc"], dim=-1))
        if "name" not in enc:
            return s * sd, None
        sn = torch.einsum("bw,bmw->bm", F.normalize(enc["q_name"], dim=-1),
                          F.normalize(enc["name"], dim=-1))
        M = enc["name"].size(1)
        g = torch.sigmoid(self.gate(torch.cat(
            [enc["q_desc"].unsqueeze(1).expand(-1, M, -1), enc["name"]], -1)
        )).squeeze(-1)
        return s * (g * sd + (1 - g) * sn), g


# ---------------------------------------------------------------- losses
def multi_positive_nce(logits: torch.Tensor, pos: torch.Tensor,
                       live: torch.Tensor) -> torch.Tensor:
    """Per-positive InfoNCE: every positive against every live negative,
    -log( exp s_p / (exp s_p + sum_neg exp s_n) ), mean over positives.

    Not -log(sum_pos / sum_all): that is satisfied as soon as ANY called
    tool scores high, and every row calls an easy generic tool (the list),
    so the action tool -- where every twin decision is -- never had to beat
    its twins. The first stage runs trained that way sat at chance on flip
    slots. logits, pos, live: (B, M); pos/live bool."""
    neg_inf = torch.finfo(logits.dtype).min
    p = pos & live
    if not p.any():
        return logits.sum() * 0.0
    neg = torch.logsumexp(logits.masked_fill(~live | pos, neg_inf), 1, keepdim=True)
    per = torch.logaddexp(logits, neg.expand_as(logits)) - logits
    return per[p].mean()


def twin_nce(logits: torch.Tensor, pos: torch.Tensor, live: torch.Tensor,
             group: torch.Tensor) -> torch.Tensor:
    """Twins first: each positive against the tools that share its signature
    (`sig_group`), and nothing else -- the one decision a pointer cannot make
    from type shape. Rows whose positives have no twin contribute nothing."""
    neg_inf = torch.finfo(logits.dtype).min
    same = (group.unsqueeze(2) == group.unsqueeze(1)) & live.unsqueeze(1) & live.unsqueeze(2)
    sizes = same.sum(2)
    p = pos & live & (sizes >= 2)
    if not p.any():
        return logits.sum() * 0.0
    # other positives in the group are not negatives
    cand = same & ~(pos.unsqueeze(1) & ~torch.eye(same.size(1), dtype=torch.bool,
                                                  device=same.device))
    lse = torch.logsumexp(logits.unsqueeze(1).expand_as(cand).masked_fill(~cand, neg_inf), 2)
    return (lse - logits)[p].mean()


def relational_loss(vecs: torch.Tensor, teacher: torch.Tensor,
                    live: torch.Tensor) -> torch.Tensor:
    """Match the stage's pairwise cosines to the teacher's, within each task
    (MiniLM/TinyBERT-style relational distillation: the geometry, not the
    coordinates, so the widths need not agree). vecs (B, M, w), teacher
    (B, M, t), live (B, M) bool."""
    a = F.normalize(vecs, dim=-1)
    b = F.normalize(teacher.float(), dim=-1)
    sa = a @ a.transpose(1, 2)
    sb = b @ b.transpose(1, 2)
    m = live.unsqueeze(1) & live.unsqueeze(2)
    eye = torch.eye(m.size(1), dtype=torch.bool, device=m.device)
    m = m & ~eye
    if not m.any():
        return vecs.sum() * 0.0
    return F.mse_loss(sa[m], sb[m])


def participation_ratio(x: torch.Tensor) -> float:
    """How many dimensions a set of vectors actually spreads across:
    (sum lambda)^2 / sum lambda^2 over the covariance eigenvalues."""
    x = x - x.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh((x.T @ x) / max(len(x) - 1, 1)).clamp(min=0)
    return float(ev.sum() ** 2 / (ev ** 2).sum().clamp(min=1e-12))
