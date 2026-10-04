"""The tiny planner behind POST /plan (.claude/plans/tiny-planner-demo.md
step 3): a models/tiny checkpoint, and for a --reader checkpoint the
ELECTRA reader run live (models/tiny/live_reader.py, electra_reader.py).

A GGUF planner reads the prompt text the client built. This one reads the
task itself -- the context JSON /kanban_prompt or /db_prompt returned, the
request, and on a continuation the registers /validate bound and their types
-- and serializes and encodes it exactly as models/tiny/play.py does in the
eval loop, so the demo and the offline suite run the same code path.

Checkpoints are named `tiny:<run>` in GET /models (TINY_MODELS below).
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TINY = ROOT / "models" / "tiny"
if str(TINY) not in sys.path:
    sys.path.insert(0, str(TINY))

# name -> (checkpoint, the cache it trained on: tokenizer, keywords, layout).
# The reader checkpoints read with ELECTRA (results/R29.md); the planners
# trained on Ternlight's vectors (clt_RD, fdc25_RD, brw*_RD, s6off_RD) went
# with it on 2026-10-04.
TINY_MODELS = {
    "tiny:c0_ESC": ("runs/pod_c0e/runs/c0_ESC/best.pt", "data_cache_c0esc"),
    "tiny:c0_EL": ("runs/pod_c0e/runs/c0_EL/best.pt", "data_cache_c0e"),
    "tiny:s6off_A0": ("runs/pod_s6split/runs/s6off_A0/best.pt", "data_cache_s6off"),
}
# Constants reach the pointer as a set of line vectors, so nothing learned
# depends on how many there are; the S6 caches' layout stops at 10 and the
# kanban demo board has 16 (play.py --max-const). Fields likewise
# (play.py --max-field): the service worlds carry up to 51. Both only ever
# widen a checkpoint's own layout, never narrow it.
MAX_CONST = 32
MAX_FIELD = 56


def available() -> list[dict]:
    """GET /models entries for the tiny checkpoints on this machine."""
    out = []
    for name, (ckpt, cache) in TINY_MODELS.items():
        p = TINY / ckpt
        if p.exists() and (TINY / cache / "config.json").exists():
            out.append({"name": name, "template": "tiny", "tuned": True,
                        "size_mb": round(p.stat().st_size / 1e6)})
    return out


class TinyPlanner:
    def __init__(self, name: str, backoff: int = 3):
        import torch
        from tokenizers import Tokenizer
        from canvas import Layout, load_keywords
        from evaluate import load_model
        if name not in TINY_MODELS:
            raise ValueError(f"no such tiny model: {name}")
        ckpt, cache = (TINY / p for p in TINY_MODELS[name])
        cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
        self.name, self.backoff = name, backoff
        self.device = torch.device("cpu")
        self.tk = Tokenizer.from_file(str(cache / "in_tok.json"))
        self.tk.no_truncation()
        self.tk.no_padding()
        self.keywords = load_keywords(cache / "keywords.json")
        base = Layout.from_dict(cfg["layout"])
        self.layout = dataclasses.replace(base, max_const=max(MAX_CONST, base.max_const),
                                          max_field=max(MAX_FIELD, base.max_field))
        self.dims = {"max_line": cfg["max_line"], "max_req": cfg["max_req"],
                     "max_reg": cfg.get("max_reg", 8),
                     **{k: cfg[k] for k in ("names", "split", "desc_chars", "max_sig",
                                            "max_desc", "max_name", "name_words",
                                            "chunk_words", "electra_reader") if k in cfg}}
        self.model = load_model(ckpt, self.device)
        self.model.c.max_const = self.layout.max_const
        self.model.c.max_field = self.layout.max_field
        self.reader = None
        if getattr(self.model.c, "reader", False):
            from live_reader import check_reader, open_reader
            self.reader = open_reader("electra")
            try:
                check_reader(self.reader, cache, self.model.c)
            except SystemExit as exc:
                raise RuntimeError(f"{name}: {exc}") from None
        self._lock = threading.Lock()

    def plan(self, request: str, context: dict, registers: dict | None = None,
             pause_types: dict | None = None) -> dict:
        import torch
        from canvas import TaskCodec, context_symbols
        from prep import encode_one
        from sample import ar_backoff, to_text
        from core.ir import TaskContext, parse_type
        from core.pipeline import build
        from harness.context import serialize_context

        # The serialized context is line-oriented: a line break inside the
        # request would start what the parser reads as a schema line (the
        # dungeon's grid did exactly this). Its request budget is also a hard
        # cut, so say when a request went past it instead of dropping the
        # tail silently.
        request = re.sub(r"\s*\n\s*", " ", request).strip()
        request_tokens = len(self.tk.encode(request).ids)
        ctx = TaskContext.from_json(context)
        if registers and pause_types:
            # a continuation: the registers the sandbox bound, typed the way
            # /validate and harness/run.py re-derive them after a PAUSE
            ctx.initial_registers = {r: parse_type(t) for r, t in pause_types.items()
                                     if r in registers}
        source = serialize_context(request, ctx, registers or None,
                                   names=bool(self.dims.get("names")))
        syms = context_symbols(context)
        with self._lock:
            t0 = time.perf_counter()
            inputs = encode_one(source, syms, self.tk, self.layout, self.dims, self.device)
            if self.reader is not None:
                from live_reader import reader_inputs
                inputs.update(reader_inputs(source, self.dims, self.layout.max_tool,
                                            self.reader, self.device, self.model.c))
            codec = TaskCodec(self.keywords, self.layout, **syms)
            with torch.no_grad():
                canvas, tr = ar_backoff(self.model, inputs, codec,
                                        lambda text: bool(build(text, ctx).compile_ok),
                                        tries=self.backoff)
            text = to_text(canvas, codec)
            gen_ms = (time.perf_counter() - t0) * 1000
        return {"text": text, "tokens_out": len(text.split()), "gen_ms": gen_ms,
                "model": self.name, "tries": tr.tries, "source": source,
                "request_tokens": request_tokens,
                "truncated": request_tokens > self.dims["max_req"]}
