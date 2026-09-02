# B4 prep: merge the S1 LoRA into the base HF weights on CPU (bf16 -> fp32 math, save bf16).
import torch, json, time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
t0=time.time()
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B"); tc = getattr(cfg, "text_config", cfg)
print("vocab", getattr(tc, "vocab_size", None), "tie", getattr(tc, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", None)), "hidden", getattr(tc, "hidden_size", None), "type", cfg.model_type, flush=True)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "baselines/qwen/models/lora_s1_adapter").merge_and_unload()
merged.save_pretrained("baselines/qwen/models/merged_s1_hf")
AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B").save_pretrained("baselines/qwen/models/merged_s1_hf")
print("merged ->", "baselines/qwen/models/merged_s1_hf", f"{time.time()-t0:.0f}s", flush=True)
