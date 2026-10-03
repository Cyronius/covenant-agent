"""The tiny planner's request budget, as a check the generators share.

The tiny planner keeps the first 128 request tokens and never reads the rest
(`max_req`, models/tiny/prep.py). A brief longer than that trains on a
request the model cannot see whole, so a generator refuses it. The count
uses the planner's own tokenizer when the local cache is present
(`models/tiny/data_cache_clt/in_tok.json`, not in git) and otherwise
estimates at 2.4 characters a token, which is what that tokenizer averages
on this kind of text (runtime/worlds/rpg.py BRIEF_CHARS).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TOKENIZER = ROOT / "models" / "tiny" / "data_cache_clt" / "in_tok.json"
BUDGET = 128
CHARS_PER_TOKEN = 2.4


@lru_cache(maxsize=1)
def _tokenizer():
    if not TOKENIZER.exists():
        return None
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOKENIZER))
    tk.no_truncation()
    tk.no_padding()
    return tk


def brief_tokens(text: str) -> int:
    tk = _tokenizer()
    if tk is None:
        return int(len(text) / CHARS_PER_TOKEN + 0.999)
    return len(tk.encode(text).ids)


def check_brief(text: str) -> None:
    """Raise if `text` is not a brief the tiny planner reads whole."""
    if "\n" in text:
        raise ValueError(f"brief has a line break: {text[:80]!r}")
    n = brief_tokens(text)
    if n > BUDGET:
        raise ValueError(f"brief is {n} tokens, over the {BUDGET} budget: "
                         f"{text[:80]!r}")
