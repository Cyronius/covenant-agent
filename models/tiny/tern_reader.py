"""Ternlight-mini rebuilt in PyTorch, with a position table
(.claude/plans/role-aware-reader.md, step 1).

The shipped reader (`@ternlight/mini` 0.1.1, the npm engine live_reader.py
runs under node) has no position table, so it reads a sentence and any
shuffle of its words as the same vector. This file reads the published
weights (`model-int4.bin`, byte-identical to the ones inside the npm wasm)
into a trainable copy:

  - ternary layers: each training weight is set to its ternary value times
    the matrix's median |weight| (1 / the stored scale). Rounding then gives
    back the same ternaries and the same scale, because fewer than half of a
    matrix's weights are zero (checked on load);
  - the int4 embeddings convert back exactly (value = code x row scale);
  - a position table (128 x 256) added to each token's vector before layer 1,
    starting at zero, so the untrained copy is the shipped reader.

The forward pass is the engine's (ternlight `training/pack/unpack.py`,
`bitlinear_forward`), with straight-through gradients so it trains.

  python tern_reader.py --fetch          # model-int4.bin + tokenizer.json -> reader/ternlight-mini/
  python tern_reader.py --check [--n 1000]   # step 1's gate: matches node's embed()

Adapted from ternlight (https://github.com/soycaporal/ternlight), MIT,
Copyright (c) 2026 Wenshu Tang: full notice in reader/LICENSE-ternlight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
TERN_DIR = HERE / "reader" / "ternlight-mini"
HF = "https://huggingface.co/wenshutang/ternlight/resolve/main/"
BIN_SHA256 = "07d8cfdba5773ad69a3fe6164b6c964e87b2368cc3ad6c2bdaf8566f2e5b6c98"
MAX_LEN = 128          # the engine truncates to 128 tokens, [CLS] and [SEP] included
EPS = 1e-5


def fetch(dst: Path = TERN_DIR) -> None:
    """Download the mini tier's weights and tokenizer from Hugging Face and
    check the weights' hash against the published sidecar's."""
    import urllib.request
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("model-int4.bin", "tokenizer.json"):
        if not (dst / name).exists():
            urllib.request.urlretrieve(HF + name, dst / name)
    got = hashlib.sha256((dst / "model-int4.bin").read_bytes()).hexdigest()
    if got != BIN_SHA256:
        raise SystemExit(f"{dst}/model-int4.bin: sha256 {got}, expected {BIN_SHA256}")


# ── the .bin v1 format (ternlight training/pack/format.py, unpack.py) ──────────

_HEADER = struct.Struct("<4sHBBIHBBHHH10s")


def _ternary(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """2-bit codes, four per byte, lowest bits first: 00 -> 0, 01 -> +1, 10 -> -1."""
    out = torch.zeros(rows, cols, dtype=torch.int8)
    for k in range(4):
        c = (packed >> (2 * k)) & 0b11
        out[:, k::4] = (c == 1).to(torch.int8) - (c == 2).to(torch.int8)
    return out


def read_bin(path: Path) -> dict:
    """The int4 .bin as tensors: header dims, embedding codes and row scales,
    and per layer the ternary codes, scale and bias of each matrix."""
    buf = Path(path).read_bytes()
    body, sha = buf[:-32], buf[-32:]
    if hashlib.sha256(body).digest() != sha:
        raise ValueError(f"{path}: trailing sha256 does not match")
    magic, ver, emb_fmt, w_fmt, vocab, d, n_layers, n_heads, ffn, out_dim, max_seq, _ = \
        _HEADER.unpack(body[:32])
    if magic != b"TERN" or ver != 1 or emb_fmt != 3 or w_fmt != 0:
        raise ValueError(f"{path}: not a v1 int4-embedding ternary .bin")
    off = 32

    def take(n: int, dtype) -> torch.Tensor:
        nonlocal off
        size = torch.empty(0, dtype=dtype).element_size()
        t = torch.frombuffer(bytearray(body[off:off + n * size]), dtype=dtype).clone()
        off += n * size
        return t

    packed = take(vocab * d // 2, torch.uint8).view(vocab, d // 2)
    nib = torch.stack([packed & 0x0F, packed >> 4], -1).view(vocab, d).to(torch.int16)
    emb_q = torch.where(nib < 8, nib, nib - 16).to(torch.int8)
    emb_s = take(vocab, torch.float32)

    def bitlinear(fin: int, fout: int, bias: bool) -> dict:
        codes = _ternary(take(fout * fin // 4, torch.uint8).view(fout, fin // 4), fout, fin)
        return {"q": codes, "scale": float(take(1, torch.float32)),
                "bias": take(fout, torch.float32) if bias else None}

    def norm() -> tuple:
        return take(d, torch.float32), take(d, torch.float32)

    layers = []
    for _ in range(n_layers):
        L = {"ln1": norm(), "q": bitlinear(d, d, False), "k": bitlinear(d, d, False),
             "v": bitlinear(d, d, False), "out": bitlinear(d, d, True), "ln2": norm(),
             "fc1": bitlinear(d, ffn, True), "fc2": bitlinear(ffn, d, True)}
        layers.append(L)
    ln_f = norm()
    proj_w = take(out_dim * d, torch.float32).view(out_dim, d)
    proj_b = take(out_dim, torch.float32)
    if off != len(body):
        raise ValueError(f"{path}: {len(body) - off} trailing bytes")
    return {"dims": {"vocab": vocab, "d": d, "n_layers": n_layers, "n_heads": n_heads,
                     "ffn": ffn, "out_dim": out_dim, "max_len": max_seq},
            "emb_q": emb_q, "emb_s": emb_s, "layers": layers, "ln_f": ln_f,
            "proj": (proj_w, proj_b)}


# ── the model ─────────────────────────────────────────────────────────────────

class BitLinear(nn.Module):
    """The engine's ternary linear layer (unpack.py `bitlinear_forward`):
    a parameterless layer norm, activations rounded to int8 per token
    (scale 128 / max |x|), weights rounded to {-1, 0, +1} (scale
    1 / median |W|), then (x_q @ w_q + bias) / (w_scale * x_scale). The bias
    is added before the rescale, as the engine does. Rounding passes the
    gradient straight through; the scales are constants to the gradient."""

    def __init__(self, fin: int, fout: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(fout, fin))
        self.bias = nn.Parameter(torch.zeros(fout)) if bias else None

    def w_scale(self) -> torch.Tensor:
        return 1.0 / self.weight.detach().abs().median().clamp(min=EPS)

    def ternary(self) -> tuple[torch.Tensor, float]:
        s = self.w_scale()
        return (self.weight.detach() * s).round().clamp(-1, 1), float(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = F.layer_norm(x, [x.size(-1)], eps=EPS)
        xs = 128.0 / xn.detach().abs().amax(-1, keepdim=True).clamp(min=EPS)
        xq = xn * xs
        xq = xq + (xq.round().clamp(-128, 127) - xq).detach()
        ws = self.w_scale()
        wq = self.weight * ws
        wq = wq + (wq.round().clamp(-1, 1) - wq).detach()
        return F.linear(xq, wq, self.bias) / (ws * xs)


class Layer(nn.Module):
    """One pre-norm block. With rel_k > 0 each head adds a learned bias to its
    attention scores by word offset (key position - query position, clipped
    to +-rel_k): order-aware, but a phrase reads the same wherever it sits."""

    def __init__(self, d: int, n_heads: int, ffn: int, rel_k: int = 0):
        super().__init__()
        self.n_heads, self.rel_k = n_heads, rel_k
        self.ln1, self.ln2 = nn.LayerNorm(d, eps=EPS), nn.LayerNorm(d, eps=EPS)
        self.q, self.k, self.v = (BitLinear(d, d, False) for _ in range(3))
        self.out = BitLinear(d, d, True)
        self.fc1, self.fc2 = BitLinear(d, ffn, True), BitLinear(ffn, d, True)
        if rel_k:
            self.rel = nn.Parameter(torch.zeros(n_heads, 2 * rel_k + 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = (m(h).view(B, T, self.n_heads, -1).transpose(1, 2) for m in (self.q, self.k, self.v))
        att = mask[:, None, None, :]
        if self.rel_k:
            ar = torch.arange(T, device=x.device)
            off = (ar[None, :] - ar[:, None]).clamp(-self.rel_k, self.rel_k) + self.rel_k
            att = self.rel[:, off].unsqueeze(0).masked_fill(~att, float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=att)
        x = x + self.out(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TernReader(nn.Module):
    """Tokens -> 384-number unit vector: embeddings (+ positions), two
    pre-norm layers, a final norm, a mean over real tokens, an fp32
    projection. Word order comes in one of two ways, both zero at the start
    so the untrained copy is the shipped reader:
      pos="abs"  a table of one vector per position, added to each token's
                 (the plan's table; reader r1);
      pos="rel"  each layer's per-head attention bias by word offset
                 (Layer.rel), which keeps a phrase's reading independent of
                 where it sits in the text;
      pos="nb"   each token's vector plus a learned per-channel share of its
                 neighbours' (offsets +-1 .. +-rel_k, `nb`), before layer 1:
                 also independent of where a phrase sits."""

    def __init__(self, vocab=30522, d=256, n_layers=2, n_heads=4, ffn=1024,
                 out_dim=384, max_len=MAX_LEN, pos="abs", rel_k=8):
        super().__init__()
        self.dims = dict(vocab=vocab, d=d, n_layers=n_layers, n_heads=n_heads,
                         ffn=ffn, out_dim=out_dim, max_len=max_len, pos=pos, rel_k=rel_k)
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(max_len, d)) if pos == "abs" else None
        self.nb = nn.Parameter(torch.zeros(2 * rel_k, d)) if pos == "nb" else None
        self.layers = nn.ModuleList(Layer(d, n_heads, ffn, rel_k if pos == "rel" else 0)
                                    for _ in range(n_layers))
        # training only: read the embeddings as their int4 rows (the shipped
        # format), gradients passed straight through to the full-precision table
        self.int4_ste = False
        self.ln_f = nn.LayerNorm(d, eps=EPS)
        self.proj = nn.Linear(d, out_dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ids (B, T) with 0 as padding, mask (B, T) bool, True = real."""
        x = self.emb(ids)
        if self.int4_ste:
            s = (x.detach().abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-8)
            x = x + ((x.detach() / s).round().clamp(-7, 7) * s - x.detach())
        if self.pos is not None:
            x = x + self.pos[:ids.size(1)]
        if self.nb is not None:
            k, T = self.dims["rel_k"], ids.size(1)
            xp = F.pad(x, (0, 0, k, k))         # padding rows are zero, as the pad token's embedding is
            offs = [o for o in range(-k, k + 1) if o]
            x = x + sum(self.nb[j] * xp[:, k + o:k + o + T] for j, o in enumerate(offs))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(self.proj(pooled), dim=-1)

    @classmethod
    def from_bin(cls, path: Path, pos: str = "abs", rel_k: int = 8) -> "TernReader":
        """The shipped reader, training weights rebuilt, positions zero."""
        w = read_bin(path)
        m = cls(**w["dims"], pos=pos, rel_k=rel_k)
        with torch.no_grad():
            m.emb.weight.copy_(w["emb_q"].float() * w["emb_s"].unsqueeze(1))
            for layer, L in zip(m.layers, w["layers"]):
                layer.ln1.weight.copy_(L["ln1"][0]); layer.ln1.bias.copy_(L["ln1"][1])
                layer.ln2.weight.copy_(L["ln2"][0]); layer.ln2.bias.copy_(L["ln2"][1])
                for name in ("q", "k", "v", "out", "fc1", "fc2"):
                    mod, src = getattr(layer, name), L[name]
                    zero = float((src["q"] == 0).float().mean())
                    if zero >= 0.5:
                        raise ValueError(f"{name}: {zero:.0%} zeros, the median is not the scale")
                    mod.weight.copy_(src["q"].float() / src["scale"])
                    if src["bias"] is not None:
                        mod.bias.copy_(src["bias"])
            m.ln_f.weight.copy_(w["ln_f"][0]); m.ln_f.bias.copy_(w["ln_f"][1])
            m.proj.weight.copy_(w["proj"][0]); m.proj.bias.copy_(w["proj"][1])
        return m

    def order_params(self) -> list[nn.Parameter]:
        """The parameters that carry word order: the table or the biases."""
        if self.pos is not None:
            return [self.pos]
        if self.nb is not None:
            return [self.nb]
        return [l.rel for l in self.layers]

    def ternary_layers(self):
        return [mod for layer in self.layers for mod in layer.modules() if isinstance(mod, BitLinear)]

    def quantize_embedding_int4_(self) -> None:
        """Back to the shipped embedding format: per row, scale = max |row| / 7,
        codes rounded into [-7, 7]. A row training never touched comes back
        unchanged."""
        with torch.no_grad():
            w = self.emb.weight
            s = (w.abs().amax(1, keepdim=True) / 7.0).clamp(min=1e-8)
            w.copy_((w / s).round().clamp(-7, 7) * s)
            w[0].zero_()

    def save(self, path: Path, **meta) -> None:
        torch.save({"dims": self.dims, "state": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path: Path) -> "TernReader":
        """A saved TernReader (.pt), or the shipped .bin."""
        if Path(path).suffix == ".bin":
            return cls.from_bin(path)
        ck = torch.load(path, map_location="cpu")
        m = cls(**ck["dims"])
        m.load_state_dict(ck["state"])
        return m


class Tokenizer:
    """The engine's BERT WordPiece tokenizer (same tokenizer.json): [CLS] and
    [SEP] added, the first 128 ids kept."""

    def __init__(self, path: Path = TERN_DIR / "tokenizer.json"):
        from tokenizers import Tokenizer as HFTokenizer
        self.tk = HFTokenizer.from_file(str(path))
        self.tk.no_padding()
        self.tk.no_truncation()

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [e.ids[:MAX_LEN] for e in self.tk.encode_batch(texts)]

    @staticmethod
    def batch(ids: list[list[int]], device=None) -> tuple[torch.Tensor, torch.Tensor]:
        T = max(len(i) for i in ids)
        t = torch.zeros(len(ids), T, dtype=torch.long)
        for r, i in enumerate(ids):
            t[r, :len(i)] = torch.tensor(i)
        return t.to(device), (t != 0).to(device)


class TorchReader:
    """live_reader.LiveReader's interface over a TernReader: (n, 384) unit
    vectors in order, cached by text."""

    def __init__(self, path: Path, device="cpu", batch: int = 256):
        self.model = TernReader.load(path).eval().to(device)
        self.tk = Tokenizer()
        self.device, self.bs = device, batch
        self.cache: dict[str, torch.Tensor] = {}
        self.name = f"tern-torch:{Path(path).parent.name}/{Path(path).name}"
        self.sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        """No cache: texts batched by length, returned in order."""
        ids = self.tk.ids(texts)
        order = sorted(range(len(texts)), key=lambda i: len(ids[i]))
        out = torch.empty(len(texts), self.model.dims["out_dim"])
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            t, m = self.tk.batch([ids[i] for i in idx], self.device)
            out[idx] = self.model(t, m).float().cpu()
        return out

    def vectors(self, texts: list[str]) -> torch.Tensor:
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            for t, v in zip(new, self.embed(new)):
                self.cache[t] = v
        return torch.stack([self.cache[t] for t in texts])

    def close(self) -> None:
        pass


def shuffled(text: str, rng: random.Random) -> str | None:
    """The same words in another order, or None when there is no other order."""
    w = text.split()
    if len(set(w)) < 2:
        return None
    s = w[:]
    while s == w:
        rng.shuffle(s)
    return " ".join(s)


def check(n: int, node_dir: Path) -> int:
    """Step 1's gate: on n texts of data_cache_c0's reader lines the rebuilt
    reader matches node's embed() (cosine >= 0.9999 on every one), and with
    the position table at zero a shuffled sentence gives the same vector."""
    from live_reader import LiveReader
    texts = [json.loads(line) for line in (HERE / "data_cache_c0" / "reader_texts.jsonl").open(encoding="utf-8")]
    rng = random.Random(0)
    texts = rng.sample(texts, n)
    ref = LiveReader(node_dir).vectors(texts)
    tr = TorchReader(TERN_DIR / "model-int4.bin")
    got = tr.embed(texts)
    cos = (ref * got).sum(1)
    worst = int(cos.argmin())
    print(f"match node: {n} texts, cosine min {float(cos.min()):.6f} mean {float(cos.mean()):.6f}"
          f" (worst {texts[worst]!r})")
    pairs = [(t, s) for t in texts if (s := shuffled(t, rng))][:200]
    a, b = tr.embed([p for p, _ in pairs]), tr.embed([s for _, s in pairs])
    sc = (a * b).sum(1)
    print(f"shuffled, positions zero: {len(pairs)} pairs, cosine min {float(sc.min()):.6f}")
    ok = float(cos.min()) >= 0.9999 and float(sc.min()) >= 0.9999
    print("gate:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--node", default=str(HERE / "reader"), metavar="DIR",
                    help="dir with node_modules/@ternlight/mini, the reference")
    args = ap.parse_args()
    if args.fetch:
        fetch()
        print(f"{TERN_DIR}: model-int4.bin (sha256 ok), tokenizer.json")
    if args.check:
        return check(args.n, Path(args.node))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
