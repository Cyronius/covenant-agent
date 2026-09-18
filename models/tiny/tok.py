"""Two tokenizers: a byte-pair one for the English input, a symbol table for the program.

The input side is prose and schema text, so it needs subwords. The output side is
a closed set of a few hundred keywords and symbols, so a lookup table is both
exact and small -- and an exact output vocabulary is what lets the diffusion
model put a distribution over "every legal token at this slot".
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from corpus import Example, program_tokens

PAD, MASK, BOS = "PAD", "MASK", "BOS"


class OutVocab:
    """Exact symbol table for program tokens. PAD is 0, MASK is 1."""

    def __init__(self, tokens: list[str]):
        self.itos = [PAD, MASK] + [t for t in tokens if t not in (PAD, MASK)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad(self) -> int:
        return 0

    @property
    def mask(self) -> int:
        return 1

    def encode(self, toks: list[str], length: int) -> list[int]:
        ids = [self.stoi[t] for t in toks if t in self.stoi][:length]
        return ids + [self.pad] * (length - len(ids))

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OutVocab":
        itos = json.loads(path.read_text(encoding="utf-8"))
        v = cls.__new__(cls)
        v.itos = itos
        v.stoi = {t: i for i, t in enumerate(itos)}
        return v

    @classmethod
    def build(cls, examples: list[Example], extra_symbols: int = 40) -> "OutVocab":
        """Every token seen, plus headroom of unseen symbol indices.

        Symbols are assigned per request, so a held-out world can hand us a T or
        F index this corpus never used. Reserving the indices up front costs a
        few rows of an embedding table and avoids an out-of-vocabulary failure
        that would look like a model error.
        """
        seen: set[str] = set()
        for e in examples:
            seen.update(program_tokens(e.target))
        for letter in ("T", "F", "C", "S", "N", "B", "D", "I", "r"):
            limit = 16 if letter == "r" else extra_symbols
            seen.update(f"{letter}{i}" for i in range(limit))
        return cls(sorted(seen))


def train_input_tokenizer(texts: list[str], vocab_size: int = 2048,
                          max_length: int = 1024) -> Tokenizer:
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>"],
        show_progress=False,
    )
    tk.train_from_iterator(texts, trainer)
    tk.enable_padding(pad_id=tk.token_to_id("<pad>"), pad_token="<pad>")
    tk.enable_truncation(max_length=max_length)
    return tk


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from corpus import load, COVENANT

    ex = load(COVENANT / "data" / "s5_plain.jsonl", limit=4000)
    ov = OutVocab.build(ex)
    print(f"output vocab: {len(ov)} tokens")
    print("first 20:", ov.itos[:20])

    tk = train_input_tokenizer([e.source for e in ex[:2000]])
    enc = tk.encode(ex[0].source)
    print(f"\ninput vocab: {tk.get_vocab_size()}")
    print(f"source {len(ex[0].source)} chars -> {len(enc.ids)} tokens")
    lens = sorted(len(tk.encode(e.source).ids) for e in ex[:500])
    print(f"input tokens: p50 {lens[len(lens)//2]} p95 {lens[int(len(lens)*.95)]} max {lens[-1]}")

    toks = program_tokens(ex[0].target)
    ids = ov.encode(toks, 64)
    assert ov.decode(ids)[:len(toks)] == toks, "output round-trip failed"
    print("\noutput round-trip ok")
