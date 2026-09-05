# Last mile of the S3 corpus: generated SFT rows + the B3 open-data draw
# (reused from S2, cached) + the verified real-turn rows x3, shuffled;
# then the two numbers the pod run needs -- the training cap, set from the
# measured max so the crowded tail is never truncated (S2 capped 6,144
# against a 6,459 max), and the vocab keep-set, built as the S2 set UNION
# every token this corpus actually uses (the S2R lesson: a keep-set that
# predates the corpus silently re-tokenises what it misses).
# Run from the repo root:  bash results/logs/finish_s3_corpus.sh
set -e
cd /c/code/covenant-agent
python -m baselines.qwen.mix_sft --out data/sft_s3.jsonl --seed 20260905 \
    data/sft_s3_gen.jsonl:1 data/open_pairs/b3_draw.jsonl:1 data/sft_real_train.jsonl:3
python - <<'EOF'
import collections, json, random
tasks = [json.loads(l) for l in open('data/s3_tasks.jsonl', encoding='utf-8')]
print("level mix:", sorted(collections.Counter(t['level'] for t in tasks).items()))
print("crowded:", sum('crowded' in t['tags'] for t in tasks), "| abort:", sum(t['expected_status'] == 'aborted' for t in tasks),
      "| worlds:", len({t['world'] for t in tasks}))
rows = [json.loads(l) for l in open('data/sft_s3.jsonl', encoding='utf-8')]
print("ABORT targets:", sum(r['messages'][-1]['content'].strip().startswith('ABORT') for r in rows), "of", len(rows))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('baselines/qwen/models/lora_s2')
def toklen(r):
    return len(tok(tok.apply_chat_template(r['messages'], tokenize=False, enable_thinking=False),
                   add_special_tokens=False)['input_ids'])
# Exact max over the whole corpus (not a sample): the cap must cover it.
used = set(json.load(open('data/s2_used_token_ids.json')))
before = len(used)
lens = []
for r in rows:
    txt = tok.apply_chat_template(r['messages'], tokenize=False, enable_thinking=False)
    ids = tok(txt, add_special_tokens=False)['input_ids']
    lens.append(len(ids))
    used.update(ids)
lens.sort(); n = len(lens)
mx = lens[-1]
cap = ((mx + 511) // 512) * 512
print(f"tokens/row (all {n}): p50 {lens[n//2]} p95 {lens[int(.95*n)]} max {mx}; over 6144: {sum(x > 6144 for x in lens)}")
print(f"TRAIN_CAP={cap}")
open('data/s3_train_cap.txt', 'w').write(str(cap))
json.dump(sorted(used), open('data/s3_used_token_ids.json', 'w'))
print(f"keep-set: {before} -> {len(used)} (+{len(used) - before}) -> data/s3_used_token_ids.json")
EOF
echo
echo "Upload to the pod: data/sft_s3.jsonl data/s3_used_token_ids.json data/s3_train_cap.txt baselines/qwen/train_b.py baselines/qwen/prune_vocab.py baselines/qwen/convert_pruned.py results/logs/train_s3.sh"
