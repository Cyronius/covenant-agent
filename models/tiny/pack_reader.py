"""Pack the slot reader's ternary ELECTRA body and span tagger into one file
(.claude/plans/electra-slot-reader.md, step 3).

The layout follows Ternlight's .bin v1 (tern_reader.read_bin): a fixed
header, sections in a fixed order, a trailing sha256 of everything before
it. What it adds for ELECTRA: a position table, the type-0 row, the
embedding norm and 128->256 projection, post-sublayer norms, a bias on every
ternary matrix (added after the rescale), and the tagger. The word table is
int4 (per row: codes in [-7, 7], scale = max |row| / 7, two codes per byte,
low nibble first); ternary matrices are 2-bit codes (00 = 0, 01 = +1,
10 = -1, four per byte, lowest bits first) with one fp32 scale each.
Everything else is fp32; the tagger is fp16.

Ternlight-mini (model-int4.bin) is the other half of the reader and ships as
it is.

  python pack_reader.py pack --body reader/tern_electra/e1/model.pt \
      --tagger reader/slots/t_tern_e1/tagger.pt --out reader/slot_reader.bin
  python pack_reader.py verify --bin reader/slot_reader.bin --body ... --tagger ...
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import torch

MAGIC = b"TSLR"
VERSION = 1
# magic, version, vocab, emb, d, heads, ffn, layers, max_pos, n_states, n_roles, tagger heads, reserved
HEADER = struct.Struct("<4sHIHHBHBHBBB9s")


def _f32(t: torch.Tensor) -> bytes:
    return t.detach().float().contiguous().numpy().astype("<f4").tobytes()


def _f16(t: torch.Tensor) -> bytes:
    return t.detach().half().contiguous().numpy().astype("<f2").tobytes()


def _int4(w: torch.Tensor) -> bytes:
    s = (w.abs().amax(1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = (w / s).round().clamp(-7, 7).to(torch.int8)
    u = (q & 0x0F).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous().numpy().tobytes() + _f32(s.squeeze(1))


def _ternary(mod) -> bytes:
    q, scale = mod.ternary()
    if float((q == 0).float().mean()) >= 0.5:
        raise ValueError("half or more of a matrix is zero: its median is not its scale")
    codes = torch.zeros_like(q, dtype=torch.uint8)
    codes[q > 0], codes[q < 0] = 1, 2
    c = codes.view(q.size(0), -1, 4)
    packed = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return packed.contiguous().numpy().tobytes() + struct.pack("<f", scale) + _f32(mod.bias)


def _norm(ln) -> bytes:
    return _f32(ln.weight) + _f32(ln.bias)


def tagger_tensors(tagger) -> list[tuple[str, torch.Tensor]]:
    """The tagger's tensors in a fixed order (its state dict, names sorted)."""
    return sorted(tagger.state_dict().items())


def pack(body, tagger) -> bytes:
    dm = body.dims
    parts = [HEADER.pack(MAGIC, VERSION, dm["vocab"], dm["emb"], dm["d"], dm["heads"], dm["ffn"],
                         dm["layers"], dm["max_pos"], tagger.cfg["n_states"], 7, tagger.cfg["heads"], b"\0" * 9)]
    with torch.no_grad():
        parts += [_int4(body.word.weight), _f32(body.pos.weight), _f32(body.tok_type.weight[0]),
                  _norm(body.ln_e), _f32(body.proj.weight), _f32(body.proj.bias)]
        for L in body.layers:
            parts += [_ternary(m) for m in (L.q, L.k, L.v, L.o)]
            parts += [_norm(L.ln1), _ternary(L.fc1), _ternary(L.fc2), _norm(L.ln2)]
        parts += [_f16(t) for _, t in tagger_tensors(tagger)]
    body_bytes = b"".join(parts)
    return body_bytes + hashlib.sha256(body_bytes).digest()


def unpack(blob: bytes):
    """-> (TernElectra, SpanTagger) rebuilt from the file: ternary training
    weights = code / scale, so the forward pass rounds back to the same codes."""
    from slot_tagger import SpanTagger
    from tern_electra import TernElectra
    body_b, sha = blob[:-32], blob[-32:]
    if hashlib.sha256(body_b).digest() != sha:
        raise ValueError("sha256 does not match")
    magic, ver, vocab, emb, d, heads, ffn, layers, max_pos, n_states, n_roles, t_heads, _ = HEADER.unpack(
        body_b[:HEADER.size])
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not a slot reader v1 file")
    off = HEADER.size

    def take(n: int, dtype) -> torch.Tensor:
        nonlocal off
        size = torch.empty(0, dtype=dtype).element_size()
        t = torch.frombuffer(bytearray(body_b[off:off + n * size]), dtype=dtype).clone()
        off += n * size
        return t

    m = TernElectra(vocab, emb, d, heads, ffn, layers, max_pos)
    with torch.no_grad():
        packed = take(vocab * emb // 2, torch.uint8).view(vocab, emb // 2)
        nib = torch.stack([packed & 0x0F, packed >> 4], -1).view(vocab, emb).to(torch.int16)
        q = torch.where(nib < 8, nib, nib - 16).float()
        m.word.weight.copy_(q * take(vocab, torch.float32).unsqueeze(1))
        m.pos.weight.copy_(take(max_pos * emb, torch.float32).view(max_pos, emb))
        m.tok_type.weight.zero_()
        m.tok_type.weight[0].copy_(take(emb, torch.float32))

        def norm(ln, n):
            ln.weight.copy_(take(n, torch.float32))
            ln.bias.copy_(take(n, torch.float32))

        def tern(mod):
            fout, fin = mod.weight.shape
            p = take(fout * fin // 4, torch.uint8).view(fout, fin // 4)
            codes = torch.zeros(fout, fin)
            for k in range(4):
                c = (p >> (2 * k)) & 0b11
                codes[:, k::4] = (c == 1).float() - (c == 2).float()
            scale = float(take(1, torch.float32))
            mod.weight.copy_(codes / scale)
            mod.bias.copy_(take(fout, torch.float32))

        norm(m.ln_e, emb)
        m.proj.weight.copy_(take(d * emb, torch.float32).view(d, emb))
        m.proj.bias.copy_(take(d, torch.float32))
        for L in m.layers:
            for mod in (L.q, L.k, L.v, L.o):
                tern(mod)
            norm(L.ln1, d)
            tern(L.fc1)
            tern(L.fc2)
            norm(L.ln2, d)
        t = SpanTagger(n_states, d, t_heads)
        state = {}
        for name, ref in tagger_tensors(t):
            state[name] = take(ref.numel(), torch.float16).view(ref.shape).float()
        t.load_state_dict(state)
    if off != len(body_b):
        raise ValueError(f"{len(body_b) - off} trailing bytes")
    return m, t


def _load(body_path: str, tagger_path: str):
    from slot_tagger import SpanTagger
    from tern_electra import TernElectra
    body = TernElectra.load(Path(body_path)).eval()
    ck = torch.load(tagger_path)
    tagger = SpanTagger(**ck["cfg"])
    tagger.load_state_dict(ck["state"])
    return body, tagger.eval()


def cmd_pack(args) -> int:
    body, tagger = _load(args.body, args.tagger)
    blob = pack(body, tagger)
    Path(args.out).write_bytes(blob)
    print(f"{args.out}: {len(blob):,} bytes")
    return 0


def cmd_verify(args) -> int:
    """The file read back tags exactly as the trained model does, and its
    word states match to float rounding."""
    from slot_tagger import Tok, decode
    body, tagger = _load(args.body, args.tagger)
    b2, t2 = unpack(Path(args.bin).read_bytes())
    b2.eval(), t2.eval()
    texts = ["move it to bob", "copy the report from the archive to the inbox",
             "delete the cards that are not done", "list orders placed before March",
             "Returns the number of records created after the given date"]
    ids, mask, offsets = Tok()(texts)
    with torch.no_grad():
        s1, _ = body(ids, mask)
        s2, _ = b2(ids, mask)
        diff = max(float((a - b)[mask].abs().max()) for a, b in zip(s1, s2))
        d1 = decode(*tagger(torch.stack(s1), mask), offsets)
        d2 = decode(*t2(torch.stack(s2), mask), offsets)
    print(f"states max diff {diff:.2e}; tags identical: {d1 == d2}")
    return 0 if d1 == d2 and diff < 1e-3 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--body", required=True)
        p.add_argument("--tagger", required=True)
        if name == "pack":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--bin", required=True)
    args = ap.parse_args()
    return {"pack": cmd_pack, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
