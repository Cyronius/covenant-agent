"""One resident base serving all three roles — IR planning, prose, and
embeddings — instead of the two full GGUFs `server/dev_server.py` loads
today (planner + a second `DEFAULT_WRITER_MODEL` for prose, +775 MB).

The three roles are three toggles on one `llama_context`, not three models:

  plan   adapter attached      grammar-constrained Agent Core generation
  write  adapter detached      the base weights, which still write prose
  embed  embeddings on         per-token final-layer states, for a head

`plan`/`write` ride on `lora_runtime.attach`/`detach` — E1 (results/S2.md
"A8 follow-up") measured the adapter path against the merged checkpoint and
got identical call sequences on 77 of 78 comparable tasks, so the adapter is
not a downgrade from merging.

`embed` is the part that is easy to talk yourself out of. llama.cpp exposes
no intermediate hidden states, so an early-exit head looked like it needed a
separate truncated GGUF — a second file, which defeats the point. It does
not: `llama_set_embeddings` is a *runtime* setter, and on a generative model
`llama_get_embeddings` returns the per-token final-layer states contiguously
(the pooling-NONE branch at `llama_cpp/llama.py:1121`). Verified on this
box 2026-09-15: a context built with `embedding=False` generates, flips to
embeddings and yields n_tokens x 1024 vectors, flips back and generates
byte-identical text. Final-layer, not intermediate — which is the normal
input for a trained pooling/projection head anyway.

Every role switch clears the KV cache — changing the active adapter set
changes the effective weights, and `Llama.embed` clears the cache itself per
batch (`llama.py:1116`) — so a switch throws away the shared system-prompt
prefix and the next planner turn re-prefills from zero. E2 measured what
that costs (results/S2.md "E2 — what a role switch costs"): **+0.6% for a
prose turn, +2.4% for an embedding turn**, and ~4.5 ms for the toggle
itself. It is nearly free because the prefix cache it destroys does not
exist: on this hybrid architecture `kv_cache_seq_rm` refuses partial
removal, so llama.cpp re-evaluates the whole prompt whenever it differs at
all from the last one. Only an exact whole-prompt repeat hits the cache.
That is the argument for one resident model — the second GGUF in
`dev_server.py` buys a per-instance prefix cache that never fires across
distinct requests.

Same-role calls do NOT switch: `_to` is a no-op when already in the role, so
a run of planner turns keeps its prefix exactly as today.
"""
from __future__ import annotations

import time
from pathlib import Path

from baselines.qwen import lora_runtime

PLAN, WRITE, EMBED = "plan", "write", "embed"


class RoleModel:
    """One base model, three roles. `adapter` is a GGUF LoRA (convert with
    llama.cpp's `convert_lora_to_gguf.py`); without one, `plan` and `write`
    are the same weights and switching between them is free."""

    def __init__(self, base: str | Path, adapter: str | Path | None = None,
                 n_ctx: int = 4096, n_threads: int | None = None,
                 adapter_scale: float = 1.0, embed_with_adapter: bool = False,
                 verbose: bool = False):
        import llama_cpp
        from llama_cpp import Llama

        # embedding=False so the context allocates logits and can generate;
        # the embed role flips it at runtime. pooling NONE is what makes
        # llama_get_embeddings hand back per-token rows instead of one
        # pooled vector — the head does its own pooling.
        self.llm = Llama(model_path=str(base), n_ctx=n_ctx,
                         n_threads=n_threads, verbose=verbose,
                         embedding=False,
                         pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE)
        self.adapter_path = str(adapter) if adapter else None
        self.adapter_scale = adapter_scale
        # Embeddings come from the base by default: the adapter is tuned for
        # IR, and a general-purpose embedding should not carry that. Flip
        # this if the head is meant to see planner-space semantics.
        self.embed_with_adapter = embed_with_adapter
        self._adapter = None
        self._attached = False
        self._embeddings = False
        self.role: str | None = None
        self.switches = 0
        self.switch_ms = 0.0

    # ---------------------------------------------------------------- roles

    def _want(self, role: str) -> tuple[bool, bool]:
        """(adapter attached, embeddings on) for a role."""
        if role == PLAN:
            return (self.adapter_path is not None, False)
        if role == WRITE:
            return (False, False)
        if role == EMBED:
            return (self.embed_with_adapter and self.adapter_path is not None,
                    True)
        raise ValueError(f"unknown role: {role}")

    def _to(self, role: str) -> None:
        if role == self.role:
            return  # same role: keep the prefix cache, change nothing
        import llama_cpp
        want_adapter, want_embed = self._want(role)
        t0 = time.perf_counter()
        if want_adapter != self._attached:
            if want_adapter:
                if self._adapter is None:
                    self._adapter = lora_runtime.load_adapter(
                        self.llm, self.adapter_path)
                lora_runtime.attach(self.llm, self._adapter,
                                    self.adapter_scale)
            else:
                lora_runtime.detach(self.llm)
            self._attached = want_adapter
        if want_embed != self._embeddings:
            llama_cpp.llama_set_embeddings(self.llm._ctx.ctx, want_embed)
            # Llama.embed guards on the Python mirror of this flag
            # (llama.py:1096) and nothing else reads it, so keep the two
            # in step rather than reconstructing the model.
            self.llm.context_params.embeddings = want_embed
            self.llm.reset()
            self.llm._ctx.kv_cache_clear()
            self._embeddings = want_embed
        self.switch_ms += (time.perf_counter() - t0) * 1000
        self.switches += 1
        self.role = role

    # ------------------------------------------------------------- the work

    def plan(self, prompt: str, grammar=None, max_tokens: int = 256,
             stop: list | None = None) -> dict:
        self._to(PLAN)
        return self._complete(prompt, grammar, max_tokens, stop)

    def write(self, prompt: str, max_tokens: int = 120,
              stop: list | None = None) -> dict:
        self._to(WRITE)
        return self._complete(prompt, None, max_tokens, stop)

    def embed(self, text: str | list[str]) -> list:
        """Per-token final-layer states: [n_tokens][n_embd] for one string,
        or a list of those for a list. Pool and project downstream — this
        deliberately does not pool for you."""
        self._to(EMBED)
        return self.llm.embed(text, return_count=False)

    def _complete(self, prompt: str, grammar, max_tokens: int,
                  stop: list | None) -> dict:
        t0 = time.perf_counter()
        res = self.llm.create_completion(
            prompt, grammar=grammar, temperature=0.0, max_tokens=max_tokens,
            stop=stop or ["<|im_end|>"])
        ms = (time.perf_counter() - t0) * 1000
        choice = res["choices"][0]
        usage = res.get("usage", {})
        return {"text": choice["text"], "ms": ms,
                "finish_reason": choice["finish_reason"],
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "role": self.role}

    def stats(self) -> dict:
        return {"switches": self.switches,
                "switch_ms_total": round(self.switch_ms, 2),
                "role": self.role, "adapter": self.adapter_path}

    def close(self) -> None:
        if self._adapter is not None:
            lora_runtime.free_adapter(self._adapter)
            self._adapter = None
