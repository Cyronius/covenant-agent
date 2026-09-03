# Last mile of the S2 corpus: merge generated SFT rows with the B3 draw,
# report the mix and token lengths, dump the vocab keep-set as token ids
# (the only thing besides sft_s2.jsonl that goes to the pod), and list
# what to upload. Run from the repo root:  bash results/logs/finish_s2_corpus.sh
set -e
cd /c/code/covenant-agent
python - <<'EOF'
import collections, json, random
gen = [json.loads(l) for l in open('data/sft_s2_gen.jsonl', encoding='utf-8')]
b3 = [json.loads(l) for l in open('data/open_pairs/b3_draw.jsonl', encoding='utf-8')]
rows = [{"messages": r["messages"]} for r in gen] + [{"messages": r["messages"]} for r in b3]
random.Random(20260903).shuffle(rows)
with open('data/sft_s2.jsonl', 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{len(rows)} SFT rows -> data/sft_s2.jsonl ({len(gen)} generated + {len(b3)} open-data)")
tasks = [json.loads(l) for l in open('data/s2_tasks.jsonl', encoding='utf-8')]
lv = collections.Counter(t['level'] for t in tasks)
print("level mix:", sorted(lv.items()))
print("crowded:", sum('crowded' in t['tags'] for t in tasks), "| abort:", sum(t['expected_status'] == 'aborted' for t in tasks),
      "| worlds:", len({t['world'] for t in tasks}))
print("abort reasons:", collections.Counter(t['reference'].get('abort_reason') for t in tasks if t['expected_status'] == 'aborted'))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('baselines/qwen/models/merged_s1_hf')
samp = random.Random(1).sample(rows, 3000)
lens = sorted(sum(len(tok.encode(m['content'])) for m in r['messages']) for r in samp)
n = len(lens)
print(f"tokens/row (3000 sample): p50 {lens[n//2]} p95 {lens[int(.95*n)]} max {lens[-1]}; over 4096: {sum(x > 4096 for x in lens)/n:.2%}")
EOF
# keep-set ids for the pod (no text leaves the machine)
python -m baselines.qwen.prune_vocab --src baselines/qwen/models/merged_s1_hf --out /dev/null --floor 40000 \
    --dump-used data/s2_used_token_ids.json
python -c "import json; print('keep-set ids:', len(json.load(open('data/s2_used_token_ids.json'))))"
echo
echo "Upload to the pod: data/sft_s2.jsonl data/s2_used_token_ids.json baselines/qwen/train_b.py baselines/qwen/prune_vocab.py baselines/qwen/convert_pruned.py results/logs/train_s2.sh"
