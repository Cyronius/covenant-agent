"""Toggle a runtime LoRA adapter on a loaded llama-cpp-python model.

Plan `.claude/plans/writer-adapter-experiment.md` E2. `Llama` attaches an
adapter once at construction (llama_cpp/llama.py:436-458) and exposes no way
to change its scale afterwards, so the planner/writer role switch has to go
through the ctypes layer directly.

    llm = Llama(model_path=BASE, lora_path=ADAPTER, verbose=False)
    set_adapter_scale(llm, 1.0)   # planner: tuned weights
    set_adapter_scale(llm, 0.0)   # writer: base weights

Changing weights invalidates everything already in the KV cache, so this
resets the model. That reset is the cost E2 measures: after a role switch
the next generation re-prefills its whole prompt.
"""
from __future__ import annotations

import ctypes

import llama_cpp


def set_adapter_scale(llm, scale: float) -> None:
    """Set the attached adapter's scale (0.0 = base weights) and drop the
    KV cache, which no longer matches the weights that produced it."""
    adapter = getattr(llm, "_lora_adapter", None)
    if adapter is None:
        raise RuntimeError("model was loaded without lora_path; nothing to scale")
    adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(adapter)
    scales = (ctypes.c_float * 1)(scale)
    if llama_cpp.llama_set_adapters_lora(llm._ctx.ctx, adapters, 1, scales):
        raise RuntimeError(f"llama_set_adapters_lora failed (scale={scale})")
    llm.reset()
    llm._ctx.kv_cache_clear()
