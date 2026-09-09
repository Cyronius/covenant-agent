"""Per-layer residual-stream influence.

For each layer l: cos(h_l, h_{l+1}) averaged over tokens. A layer whose output
is nearly parallel to its input barely moves the residual stream -- the
signature of a layer that is cheap to share, repeat or drop. Prompts are real
eval prompts so the measurement is on-distribution.

  python results/logs/s4l_layer_influence.py [MODEL_DIR] [--template qwen|chat]
"""
import json, sys, torch
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from baselines.qwen.run_a import SYSTEM, build_prompt

D = sys.argv[1] if len(sys.argv) > 1 else 'baselines/qwen/models/merged_s1_pruned_v2_hf'
TEMPLATE = sys.argv[sys.argv.index('--template') + 1] if '--template' in sys.argv else 'qwen'

tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float32).eval()
cfg = json.load(open(f'{D}/config.json'))
types = cfg.get('layer_types') or ['layer'] * cfg['num_hidden_layers']

def render(u):
    if TEMPLATE == 'chat':
        try:
            return tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True)
        except ValueError:
            # tokenizer shipped without a template; LFM2's is plain ChatML
            return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{u}"
                    f"<|im_end|>\n<|im_start|>assistant\n")
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{u}"
            f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

rows = []
for f, n in (("data/holdout/e_demo_requests.jsonl", 3),
             ("data/holdout/e_crowded.jsonl", 3),
             ("data/holdout/e_foreign.jsonl", 2)):
    rows += [json.loads(l) for l in open(f, encoding="utf-8")][:n]

acc, ntok = None, 0
for t in rows:
    ids = tok(render(build_prompt(t["input_text"], None, [])),
              return_tensors="pt").input_ids[:, -768:]
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    sims = [torch.nn.functional.cosine_similarity(
                hs[l][0].float(), hs[l + 1][0].float(), dim=-1).mean().item()
            for l in range(len(hs) - 1)]
    acc = [x + y for x, y in zip(acc, sims)] if acc else sims
    ntok += ids.shape[1]

sims = [x / len(rows) for x in acc]
print(f"\n{D}\n{len(rows)} prompts, {ntok} tokens, template={TEMPLATE}\n")
print("layer  type              cos(in,out)   change   bar")
for l, (s, ty) in enumerate(zip(sims, types)):
    d = 1 - s
    print(f"  L{l:02d}  {ty:16s}  {s:+.4f}    {d:.4f}   {'#' * min(60, int(d * 400))}")
print("\nlow 'change' = near-identity layer = cheap to share/loop/drop")
order = sorted(range(len(sims)), key=lambda i: 1 - sims[i])
print("most redundant -> least:", ' '.join(f'L{i}' for i in order))
