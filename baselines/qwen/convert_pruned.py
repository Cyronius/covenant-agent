"""Convert a vocab-pruned HF checkpoint (baselines/qwen/prune_vocab.py) to
GGUF with llama.cpp's converter.

The converter identifies the BPE pre-tokenizer by hashing the token ids of a
probe string; pruning renumbers ids, so the hash is unknown and it refuses.
The pre-tokenizer itself (regex + byte-level) is untouched, so we pin the
answer to "qwen2" and run the converter unchanged otherwise.

  python -m baselines.qwen.convert_pruned <hf_dir> --outfile <gguf> [--outtype q8_0]
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

LLAMA_CPP = Path(os.environ.get("LLAMA_CPP", r"c:\code\llama.cpp"))
sys.path.insert(0, str(LLAMA_CPP))
sys.path.insert(1, str(LLAMA_CPP / "gguf-py"))

import conversion.base as base  # noqa: E402

patched = 0
for name, cls in vars(base).items():
    if inspect.isclass(cls) and "get_vocab_base_pre" in vars(cls):
        cls.get_vocab_base_pre = lambda self, tokenizer: "qwen2"
        patched += 1
assert patched, "no class with get_vocab_base_pre found in conversion.base"

import runpy  # noqa: E402

sys.argv[0] = str(LLAMA_CPP / "convert_hf_to_gguf.py")
runpy.run_path(str(LLAMA_CPP / "convert_hf_to_gguf.py"), run_name="__main__")
