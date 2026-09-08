# Last mile of the S4 corpus (companion to results/logs/gen_s4.sh):
#
#   bash results/logs/finish_s4_corpus.sh [typed|classic]
#
#  1. Rebuild the real-turn rows in the arm's surface. S3 mixed them in at
#     ~4% x3 from the start, which is what stopped S2R's abstain trigger
#     over-firing; they have to carry the same symbols as the generated
#     rows or the model sees two symbol tables.
#  2. Mix: generated : open-domain (B3 draw, reused) : real = 1 : 1 : 3.
#  3. The training cap, from the measured max over the whole corpus (S2
#     capped 6,144 against a 6,459 max and truncated the crowded tail), and
#     the vocab keep-set as the S3 set UNION every token this corpus uses --
#     the typed letters are new token sequences (S0, N0, I0, D0, B0), so a
#     keep-set that predates the corpus would silently re-tokenise them.
set -e
cd /c/code/covenant-agent
ARM="${1:-typed}"
case "$ARM" in
  typed)   REAL_SURFACE="--symbols typed --enums --kinds"; TAG=s4 ;;
  classic) REAL_SURFACE="";                                TAG=s4c ;;
  *) echo "usage: finish_s4_corpus.sh [typed|classic]"; exit 2 ;;
esac
python -m harness.real_train_build --out data/real_train_tasks_${TAG}.jsonl $REAL_SURFACE
python -m baselines.qwen.make_sft --tasks data/real_train_tasks_${TAG}.jsonl \
    --symbols "$ARM" --out data/sft_real_train_${TAG}.jsonl
python -m baselines.qwen.mix_sft --out data/sft_${TAG}.jsonl --seed 20260908 \
    data/sft_${TAG}_gen.jsonl:1 data/open_pairs/b3_draw.jsonl:1 data/sft_real_train_${TAG}.jsonl:3
TAG=$TAG python - <<'EOF'
import collections, json, os
tag = os.environ["TAG"]
tasks = [json.loads(l) for l in open(f'data/{tag}_tasks.jsonl', encoding='utf-8')]
print("level mix:", sorted(collections.Counter(t['level'] for t in tasks).items()))
print("crowded:", sum('crowded' in t['tags'] for t in tasks),
      "| abort:", sum(t['expected_status'] == 'aborted' for t in tasks),
      "| worlds:", len({t['world'] for t in tasks}))
recipes = collections.Counter(t['provenance']['recipe'] for t in tasks)
print("argmax/argmin:", recipes['argmax_count'], recipes['argmin_count'])
progs = [t['reference']['segments'][0] for t in tasks]
print("MOST:", sum('MOST ' in p for p in progs),
      "| LEAST:", sum('LEAST ' in p for p in progs),
      "| EMPTY:", sum('EMPTY ' in p for p in progs),
      "| old count-loop:", sum('LET r8' in p for p in progs))
rows = [json.loads(l) for l in open(f'data/sft_{tag}.jsonl', encoding='utf-8')]
print("ABORT targets:", sum(r['messages'][-1]['content'].strip().startswith('ABORT')
                            for r in rows), "of", len(rows))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('baselines/qwen/models/lora_s2')
used = set(json.load(open('data/s3_used_token_ids.json')))
before = len(used)
lens = []
for r in rows:
    txt = tok.apply_chat_template(r['messages'], tokenize=False, enable_thinking=False)
    ids = tok(txt, add_special_tokens=False)['input_ids']
    lens.append(len(ids))
    used.update(ids)
lens.sort(); n = len(lens)
cap = ((lens[-1] + 511) // 512) * 512
print(f"tokens/row (all {n}): p50 {lens[n//2]} p95 {lens[int(.95*n)]} max {lens[-1]}")
print(f"TRAIN_CAP={cap}")
open(f'data/{tag}_train_cap.txt', 'w').write(str(cap))
json.dump(sorted(used), open(f'data/{tag}_used_token_ids.json', 'w'))
print(f"keep-set: {before} -> {len(used)} (+{len(used) - before}) -> data/{tag}_used_token_ids.json")
EOF
echo
echo "Upload to the pod: data/sft_${TAG}.jsonl data/${TAG}_used_token_ids.json data/${TAG}_train_cap.txt"
echo "  plus baselines/qwen/{train_b.py,prune_vocab.py,convert_pruned.py} results/logs/train_s4.sh"
