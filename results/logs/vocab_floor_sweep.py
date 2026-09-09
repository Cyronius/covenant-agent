"""How big should the English floor be?

Build the keep-set at several --floor values using ONLY the synthetic corpus's
used tokens (i.e. pretend the real requests were never seen), then ask how many
real user requests still tokenize identically to the full tokenizer. That is
the honest proxy for "English we haven't seen yet".
"""
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from baselines.qwen.prune_vocab import merge_parts

SRC = Path('baselines/qwen/models/merged_s1_hf')      # unpruned tokenizer
tok = AutoTokenizer.from_pretrained(str(SRC))
tj = json.load(open(SRC / 'tokenizer.json', encoding='utf-8'))
vocab = tj['model']['vocab']; merges = tj['model']['merges']
added = tj.get('added_tokens', [])
parents = {}
for m in merges:
    a, b = merge_parts(m)
    r = vocab.get(a + b)
    if r is not None:
        parents[r] = (vocab[a], vocab[b])

used = set(json.load(open('data/s2_used_token_ids.json', encoding='utf-8')))
print(f"synthetic-corpus used tokens: {len(used)}")

# held-out English: real user request texts (never used to build these keep-sets)
texts = []
seen = set()
for line in open('data/real_sessions/b2_pool.jsonl', encoding='utf-8'):
    t = (json.loads(line).get('request') or '').strip()
    if t and t.lower() not in seen:
        seen.add(t.lower()); texts.append(t)
print(f"held-out real request texts: {len(texts)}\n")

full = [tok(t, add_special_tokens=False).input_ids for t in texts]

def build(floor):
    keep = set(used) | set(range(256)) | {a['id'] for a in added} | set(tok.all_special_ids)
    for m in merges[:floor]:
        a, b = merge_parts(m)
        r = vocab.get(a + b)
        if r is not None:
            keep.add(r)
    stack = list(keep)
    while stack:
        t = stack.pop()
        for p in parents.get(t, ()):
            if p not in keep:
                keep.add(p); stack.append(p)
    return keep

H = 1024
print(f"{'floor':>7} {'vocab':>8} {'emb params':>11} {'identical':>10} {'texts w/ OOV':>13} {'tok inflation':>14}")
for floor in (40_000, 60_000, 80_000, 100_000, 128_000, 160_000):
    keep = build(floor)
    n = ((len(keep) + 63) // 64) * 64
    ident = sum(all(i in keep for i in ids) for ids in full)
    # a text tokenizes identically iff every token the full tokenizer produced survives
    oov = len(texts) - ident
    extra = sum(len(ids) for ids in full)
    print(f"{floor:>7} {n:>8} {n*H/1e6:>10.1f}M {ident/len(texts)*100:>9.1f}% {oov:>13} "
          f"{'(baseline)' if floor==40_000 else '':>14}")
print("\nnote: 'identical' = every token the full tokenizer emitted is in the keep-set.")
print("      Text with a dropped token still decodes correctly, but splits into pieces")
print("      the model never saw in that arrangement.")
