"""ELECTRA-small made ternary (.claude/plans/electra-slot-reader.md, step 1).

The student has ELECTRA-small's exact shape: a 128-wide word / position / type
embedding, a layer norm, a 128->256 projection, then post-norm layers
(attention, add, norm; feed-forward, add, norm). Its matrices (q, k, v, out
and the two feed-forward ones) are ternary: weights -1/0/+1 times one scale
per matrix (1 / median |W|), activations rounded to int8 per word
(127 / max |x|), rounding passed straight through to the gradient. ELECTRA
already normalizes after each sublayer, so there is no extra norm at each
matrix's input; the bias is added after the rescale. The word table trains as
int4 rows (per row: scale = max |row| / 7). Norms, biases, the position and
type tables and the 128->256 projection stay full precision. (The weight and
table scheme is the one Ternlight-mini used, R21; nothing here depends on it.)

Training copies full-precision ELECTRA at every word and every layer: the
states (mean squared error) and the attention patterns (KL), as TernaryBERT
did for BERT. Copying only a teacher's final average loses word order.

  python tern_electra.py train --out reader/tern_electra/e1 [--layers 12] [--n-texts 150000]
  python tern_electra.py check --model reader/tern_electra/e1/model.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
ELECTRA = "google/electra-small-discriminator"
EPS = 1e-5


def _hf():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    return AutoModel, AutoTokenizer


class TernLinear(nn.Module):
    """A ternary matrix with int8 activations; `quant` off gives the plain
    full-precision layer (for checking the rebuild against ELECTRA)."""

    def __init__(self, fin: int, fout: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(fout, fin))
        self.bias = nn.Parameter(torch.zeros(fout))
        self.quant = True

    def w_scale(self) -> torch.Tensor:
        return 1.0 / self.weight.detach().abs().median().clamp(min=EPS)

    def ternary(self) -> tuple[torch.Tensor, float]:
        s = self.w_scale()
        return (self.weight.detach() * s).round().clamp(-1, 1), float(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        xs = 127.0 / x.detach().abs().amax(-1, keepdim=True).clamp(min=EPS)
        xq = x * xs
        xq = xq + (xq.round().clamp(-128, 127) - xq).detach()
        ws = self.w_scale()
        wq = self.weight * ws
        wq = wq + (wq.round().clamp(-1, 1) - wq).detach()
        return F.linear(xq, wq) / (ws * xs) + self.bias


class Layer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, ln_eps: float):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.o = (TernLinear(d, d) for _ in range(4))
        self.ln1 = nn.LayerNorm(d, eps=ln_eps)
        self.fc1, self.fc2 = TernLinear(d, ffn), TernLinear(ffn, d)
        self.ln2 = nn.LayerNorm(d, eps=ln_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x (B, T, d), mask (B, T) bool -> (x, attention probabilities (B, H, T, T))."""
        B, T, D = x.shape
        q, k, v = (m(x).view(B, T, self.heads, -1).transpose(1, 2) for m in (self.q, self.k, self.v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        att = scores.softmax(-1)
        a = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = self.ln1(x + self.o(a))
        x = self.ln2(x + self.fc2(F.gelu(self.fc1(x))))
        return x, att


class TernElectra(nn.Module):
    """Tokens -> the word states after the embeddings and after every layer
    (the same list ELECTRA's `hidden_states` gives), and every layer's
    attention probabilities."""

    def __init__(self, vocab=30522, emb=128, d=256, heads=4, ffn=1024, layers=12,
                 max_pos=512, ln_eps=1e-12):
        super().__init__()
        self.dims = dict(vocab=vocab, emb=emb, d=d, heads=heads, ffn=ffn, layers=layers,
                         max_pos=max_pos, ln_eps=ln_eps)
        self.word = nn.Embedding(vocab, emb)
        self.pos = nn.Embedding(max_pos, emb)
        self.tok_type = nn.Embedding(2, emb)
        self.ln_e = nn.LayerNorm(emb, eps=ln_eps)
        self.proj = nn.Linear(emb, d)
        self.layers = nn.ModuleList(Layer(d, heads, ffn, ln_eps) for _ in range(layers))
        self.int4 = True

    def set_quant(self, on: bool) -> None:
        self.int4 = on
        for m in self.modules():
            if isinstance(m, TernLinear):
                m.quant = on

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        w = self.word(ids)
        if self.int4:
            s = (w.detach().abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-8)
            w = w + ((w.detach() / s).round().clamp(-7, 7) * s - w.detach())
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.ln_e(w + self.pos(pos)[None] + self.tok_type.weight[0])
        x = self.proj(x)
        states, atts = [x], []
        for layer in self.layers:
            x, att = layer(x, mask)
            states.append(x)
            atts.append(att)
        return states, atts

    @classmethod
    def from_electra(cls, hf, layers: int | None = None) -> "TernElectra":
        """ELECTRA-small's weights, its first `layers` layers."""
        c = hf.config
        n = layers or c.num_hidden_layers
        m = cls(c.vocab_size, c.embedding_size, c.hidden_size, c.num_attention_heads,
                c.intermediate_size, n, c.max_position_embeddings, c.layer_norm_eps)
        e = hf.embeddings
        with torch.no_grad():
            m.word.weight.copy_(e.word_embeddings.weight)
            m.pos.weight.copy_(e.position_embeddings.weight)
            m.tok_type.weight.copy_(e.token_type_embeddings.weight)
            m.ln_e.load_state_dict(e.LayerNorm.state_dict())
            m.proj.load_state_dict(hf.embeddings_project.state_dict())
            for mine, theirs in zip(m.layers, hf.encoder.layer[:n]):
                at = theirs.attention
                for a, b in ((mine.q, at.self.query), (mine.k, at.self.key), (mine.v, at.self.value),
                             (mine.o, at.output.dense), (mine.fc1, theirs.intermediate.dense),
                             (mine.fc2, theirs.output.dense)):
                    a.weight.copy_(b.weight)
                    a.bias.copy_(b.bias)
                mine.ln1.load_state_dict(at.output.LayerNorm.state_dict())
                mine.ln2.load_state_dict(theirs.output.LayerNorm.state_dict())
        return m

    def ternary_layers(self) -> list[TernLinear]:
        return [m for m in self.modules() if isinstance(m, TernLinear)]

    def quantize_embedding_int4_(self) -> None:
        with torch.no_grad():
            w = self.word.weight
            s = (w.abs().amax(1, keepdim=True) / 7.0).clamp(min=1e-8)
            w.copy_((w / s).round().clamp(-7, 7) * s)

    def save(self, path: Path, **meta) -> None:
        torch.save({"dims": self.dims, "state": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path: Path) -> "TernElectra":
        ck = torch.load(path, map_location="cpu")
        m = cls(**ck["dims"])
        m.load_state_dict(ck["state"])
        return m


class Teacher:
    """Full-precision ELECTRA-small: its states and attention probabilities."""

    def __init__(self, layers: int | None = None):
        AutoModel, AutoTokenizer = _hf()
        self.tk = AutoTokenizer.from_pretrained(ELECTRA)
        self.m = AutoModel.from_pretrained(ELECTRA, output_hidden_states=True,
                                           output_attentions=True, attn_implementation="eager").eval()
        self.n = layers or self.m.config.num_hidden_layers

    def batch(self, texts: list[str]):
        enc = self.tk(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"].bool()

    @torch.no_grad()
    def __call__(self, ids, mask):
        out = self.m(input_ids=ids, attention_mask=mask.long())
        return list(out.hidden_states[:self.n + 1]), list(out.attentions[:self.n])


def distill_loss(s_states, s_atts, t_states, t_atts, mask, lam_att: float):
    """Mean squared error of every layer's word states (real words only) and
    KL of every layer's attention rows (real query and key words)."""
    m = mask.unsqueeze(-1).float()
    n = m.sum() * s_states[0].size(-1)
    mse = sum(((s - t) ** 2 * m).sum() / n for s, t in zip(s_states, t_states)) / len(s_states)
    rows = mask[:, None, :, None] & mask[:, None, None, :]
    kl = 0.0
    for sa, ta in zip(s_atts, t_atts):
        lp = sa.clamp(min=1e-9).log()
        kl = kl + (ta * (ta.clamp(min=1e-9).log() - lp)).masked_fill(~rows, 0).sum() / (
            mask.sum() * sa.size(1))
    kl = kl / max(1, len(s_atts))
    return mse + lam_att * kl, mse, kl


def texts_for(n: int, seed: int) -> tuple[list[str], list[str]]:
    """Training and held-out texts from data/general (the same crc32 split as
    R25 and the slot reader)."""
    from slot_reader import GENERAL, held_out
    rng = random.Random(seed)
    texts = [json.loads(line)["text"] for line in GENERAL.open(encoding="utf-8")]
    texts = [t for t in texts if 3 <= len(t.split()) <= 40]
    held = [t for t in texts if held_out(t)]
    train = [t for t in texts if not held_out(t)]
    rng.shuffle(held)
    rng.shuffle(train)
    return train[:n], held[:2000]


@torch.no_grad()
def state_match(student: TernElectra, teacher: Teacher, texts: list[str], bs: int = 128) -> list[float]:
    """Per state (embeddings, then each layer): mean cosine between the
    student's and the teacher's word states, over real words."""
    student.eval()
    tot, n = None, 0
    for s in range(0, len(texts), bs):
        ids, mask = teacher.batch(texts[s:s + bs])
        ss, _ = student(ids, mask)
        ts, _ = teacher(ids, mask)
        c = torch.stack([F.cosine_similarity(a, b, dim=-1)[mask].sum() for a, b in zip(ss, ts)])
        tot = c if tot is None else tot + c
        n += int(mask.sum())
    student.train()
    return [round(float(x) / n, 4) for x in tot]


def cmd_train(args) -> int:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(args.layers)
    student = TernElectra.from_electra(teacher.m, args.layers)
    train, held = texts_for(args.n_texts, args.seed)
    held = held[:args.n_held]
    tern = student.ternary_layers()
    start = [m.ternary()[0] for m in tern]
    groups = [{"params": [m.weight], "lr": args.tern_rel / float(m.w_scale())} for m in tern]
    ids_t = {id(m.weight) for m in tern}
    groups.append({"params": [p for p in student.parameters() if id(p) not in ids_t], "lr": args.lr})
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    steps = args.epochs * (len(train) // args.batch)
    warm = max(1, steps // 30)
    log = (out / "log.jsonl").open("w", encoding="utf-8")
    row = {"step": 0, "match": state_match(student, teacher, held)}
    print(json.dumps(row), flush=True)
    log.write(json.dumps(row) + "\n")
    step, t0, acc = 0, time.time(), defaultdict(float)
    for ep in range(args.epochs):
        rng.shuffle(train)
        for b in range(0, len(train) - args.batch + 1, args.batch):
            mult = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult
            ids, mask = teacher.batch(train[b:b + args.batch])
            ts, ta = teacher(ids, mask)
            ss, sa = student(ids, mask)
            loss, mse, kl = distill_loss(ss, sa, ts, ta, mask, args.lam_att)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            acc["loss"] += float(loss.detach()); acc["mse"] += float(mse.detach()); acc["kl"] += float(kl.detach())
            if step % args.log_every == 0 or step == steps:
                moved = sum(int((m.ternary()[0] != q).sum()) for m, q in zip(tern, start)) / sum(q.numel() for q in start)
                row = {"step": step, **{k: round(v / args.log_every, 5) for k, v in acc.items()},
                       "moved": round(moved, 4), "sec": round(time.time() - t0)}
                acc.clear()
                if step % args.eval_every == 0 or step == steps:
                    row["match"] = state_match(student, teacher, held)
                    student.save(out / "last.pt", args=vars(args), step=step)
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
    student.quantize_embedding_int4_()
    row = {"step": step, "final": True, "match": state_match(student, teacher, held)}
    print(json.dumps(row), flush=True)
    log.write(json.dumps(row) + "\n")
    student.save(out / "model.pt", args=vars(args), final=row)
    print(f"wrote {out / 'model.pt'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=None, help="ELECTRA layers kept (default all 12)")
    p.add_argument("--n-texts", type=int, default=150_000)
    p.add_argument("--n-held", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="norms, biases, tables, projection")
    p.add_argument("--tern-rel", type=float, default=5e-3,
                   help="ternary training weights: this x the matrix's median |weight| per step")
    p.add_argument("--lam-att", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return {"train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
