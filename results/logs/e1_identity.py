# E1 generation identity (plan .claude/plans/writer-adapter-experiment.md):
# merged S2 checkpoint vs base + runtime LoRA, same 40 prompts the B4 vocab
# comparison used (results/logs/vocab_trim_eval.py), grammar on, temp 0.
import json, sys, time
sys.path.insert(0, '.')
from baselines.qwen.run_a import SYSTEM, build_prompt
from llama_cpp import Llama, LlamaGrammar

M = "baselines/qwen/models"
g = LlamaGrammar.from_string(open("baselines/qwen/agent_core.gbnf").read(), verbose=False)
def chat(u): return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

tasks = [json.loads(l) for l in open("data/holdout/e_demo_requests.jsonl", encoding="utf-8")][::3][:20]
tasks += [json.loads(l) for l in open("data/holdout/e_foreign.jsonl", encoding="utf-8")][:20]

arms = {"merged": dict(model_path=f"{M}/qwen3.5-0.8b-s2-q8.gguf"),
        "adapter": dict(model_path=f"{M}/Qwen3.5-0.8B-Q8_0.gguf",
                        lora_path=f"{M}/lora_s2_f16.gguf", lora_scale=1.0)}
outs = {}
for name, kw in arms.items():
    llm = Llama(n_ctx=8192, verbose=False, **kw)
    t0 = time.time(); res = []
    for t in tasks:
        r = llm.create_completion(chat(build_prompt(t["input_text"], None, [])),
                                  grammar=g, temperature=0.0, max_tokens=120,
                                  stop=["<|im_end|>"])
        res.append(r["choices"][0]["text"].strip())
    outs[name] = res
    print(name, f"{(time.time()-t0)/len(tasks):.1f}s/prompt", flush=True)
    del llm

same = sum(x == y for x, y in zip(outs["merged"], outs["adapter"]))
print(f"IDENTITY merged vs adapter: {same}/{len(tasks)}", flush=True)
json.dump(outs, open("results/logs/e1_identity.json", "w"), indent=1)
for i, (a, b) in enumerate(zip(outs["merged"], outs["adapter"])):
    if a != b:
        print(f"--- prompt {i}\n  merged : {a!r}\n  adapter: {b!r}", flush=True)
