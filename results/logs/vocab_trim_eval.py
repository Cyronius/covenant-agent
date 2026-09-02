# B4 pass/fail: pruned vs re-merged (same merge, same converter) — text identity on
# prompts, then the harness on the demo suite and E-ood (grammar on).
import json, sys, subprocess, time
sys.path.insert(0, '.')
from baselines.qwen.run_a import SYSTEM, build_prompt
from llama_cpp import Llama, LlamaGrammar
g = LlamaGrammar.from_string(open("baselines/qwen/agent_core.gbnf").read(), verbose=False)
def chat(u): return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
tasks = [json.loads(l) for l in open("data/holdout/e_demo_requests.jsonl", encoding="utf-8")][::3][:20]
tasks += [json.loads(l) for l in open("data/holdout/e_foreign.jsonl", encoding="utf-8")][:20]
outs = {}
for name in ("qwen3.5-0.8b-s1-remerged-q8.gguf", "qwen3.5-0.8b-s1-pruned-q8.gguf", "qwen3.5-0.8b-s1-q8.gguf"):
    llm = Llama(model_path=f"baselines/qwen/models/{name}", n_ctx=8192, verbose=False)
    t0 = time.time(); res = []
    for t in tasks:
        r = llm.create_completion(chat(build_prompt(t["input_text"], None, [])), grammar=g, temperature=0.0, max_tokens=120, stop=["<|im_end|>"])
        res.append(r["choices"][0]["text"].strip())
    outs[name] = res; print(name, f"{(time.time()-t0)/len(tasks):.1f}s/prompt", flush=True); del llm
names = list(outs)
for i in range(3):
    for j in range(i + 1, 3):
        same = sum(x == y for x, y in zip(outs[names[i]], outs[names[j]]))
        print(f"IDENTITY {names[i]} vs {names[j]}: {same}/{len(tasks)}", flush=True)
json.dump(outs, open("results/logs/vocab_trim_identity.json", "w"), indent=1)
for name, tag in (("qwen3.5-0.8b-s1-remerged-q8.gguf", "remerged"), ("qwen3.5-0.8b-s1-pruned-q8.gguf", "pruned")):
    for suite, extra in (("data/holdout/e_demo_requests.jsonl", []), ("data/holdout/e_ood_english.jsonl", ["--domains", "data/gen/themes", "--ctx", "16384"])):
        out = f"results/logs/s1_{tag}_{suite.split('/')[-1].replace('.jsonl','')}.jsonl"
        print("RUN", tag, suite, flush=True)
        subprocess.run([sys.executable, "-m", "baselines.qwen.run_a", "--model", f"baselines/qwen/models/{name}", "--tasks", suite, "--out", out] + extra, check=False)
print("=== all done ===", flush=True)
