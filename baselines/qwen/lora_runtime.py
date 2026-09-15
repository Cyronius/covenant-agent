"""Runtime LoRA attach/detach against an already-loaded llama.cpp model —
the ~10-line low-level call `.claude/plans/writer-adapter-experiment.md`
and `.claude/plans/example-host-and-new-worlds.md` §6 describe.

llama-cpp-python's high-level `Llama` class only wraps this at construction
time (`lora_path=`); this module reaches one layer below the public API
(`llama_cpp.llama_cpp`'s ctypes bindings, via `Llama.model`/`Llama.ctx`) to
toggle an adapter on an already-resident model — and, unlike
`lora_path=`, can attach any adapter to a base that was loaded with none,
which is what lets one resident model serve more than one stage without a
second load.

Changing the active adapter set changes the effective weights, which
invalidates whatever the KV cache holds from before the change — `attach`/
`detach` reset it for exactly that reason (`llm.reset()` +
`llm._ctx.kv_cache_clear()`, the same pair the first version of this module
used). **Earlier revisions of this file (this session, 2026-09-15) dropped
that reset** while re-deriving the mechanism from scratch instead of
reading the version already on disk — the "switching costs nothing extra"
measurement that produced is unsound: without the reset, a later call could
silently reuse a KV prefix computed under the previous adapter state. The
reset is back; the switching-cost number needs to be re-measured with it in
place before anyone trusts it. The fidelity gap this module's docstring
used to report (one example, S5 adapter vs. merged checkpoint, same tool
and argument order, a different constant chosen) is unaffected by this —
that came from two fresh, non-adjacent generations, not two calls on one
object — but still needs the real suite comparison
(`.claude/plans/writer-adapter-experiment.md`'s E1) before it means
anything past "not identical."

**Not wired into `server/dev_server.py`.** Kept as a standalone module,
reusable once someone runs that real comparison with the reset in place.
"""
from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import Llama


def _reset_kv(llm: "Llama") -> None:
    llm.reset()
    llm._ctx.kv_cache_clear()


def attach(llm: "Llama", adapter, scale: float = 1.0) -> int:
    """Attach a loaded adapter (from `load_adapter`) to `llm`'s context and
    reset the KV cache, which no longer matches the weights that produced
    it. Returns llama.cpp's status code; 0 is success (cache is reset
    either way — a failed attach still changes what's active)."""
    import llama_cpp
    adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(adapter)
    scales = (ctypes.c_float * 1)(scale)
    rc = llama_cpp.llama_set_adapters_lora(llm.ctx, adapters, 1, scales)
    _reset_kv(llm)
    return rc


def detach(llm: "Llama") -> int:
    """Clear every active adapter on `llm`'s context and reset the KV
    cache. llama.cpp has no separate remove/clear call in this build; an
    empty set does the job."""
    import llama_cpp
    empty_adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 0)()
    empty_scales = (ctypes.c_float * 0)()
    rc = llama_cpp.llama_set_adapters_lora(llm.ctx, empty_adapters, 0, empty_scales)
    _reset_kv(llm)
    return rc


def load_adapter(llm: "Llama", gguf_path: str):
    """Load a GGUF LoRA adapter against `llm`'s already-resident model — no
    second model load. Free with `free_adapter` when done with it."""
    import llama_cpp
    adapter = llama_cpp.llama_adapter_lora_init(llm.model, gguf_path.encode("utf-8"))
    if not adapter:
        raise RuntimeError(f"failed to load LoRA adapter: {gguf_path}")
    return adapter


def free_adapter(adapter) -> None:
    import llama_cpp
    llama_cpp.llama_adapter_lora_free(adapter)
