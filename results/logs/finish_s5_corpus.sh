# Last mile of the S5 corpus (companion to results/logs/gen_s5.sh):
#
#   bash results/logs/finish_s5_corpus.sh [typed|classic]
#
# Same three jobs as finish_s4_corpus.sh - rebuild the real-turn rows in the
# arm's surface, mix, measure the cap and the keep-set - with the families
# added to the mix at weight 1. That puts them at roughly a fifth of the
# corpus, which is the knob to turn if the run gets too long: append a CAP
# to the family spec (PATH:REPEAT:CAP) rather than dropping a family, so
# each family's own exam stays attributable.
#
# The keep-set starts from S4's, not S3's: S4 is the corpus this one extends.
set -e
cd /c/code/covenant-agent
ARM="${1:-typed}"
case "$ARM" in
  typed)   REAL_SURFACE="--symbols typed --enums --kinds"; TAG=s5; FAM=data/sft_fam.jsonl; PRIOR=s4 ;;
  classic) REAL_SURFACE="";                                TAG=s5c; FAM=data/sft_famc.jsonl; PRIOR=s4c ;;
  *) echo "usage: finish_s5_corpus.sh [typed|classic]"; exit 2 ;;
esac
[ -f "$FAM" ] || { echo "missing $FAM -- run gen_families.sh $ARM first"; exit 2; }
python -m harness.real_train_build --out data/real_train_tasks_${TAG}.jsonl $REAL_SURFACE
python -m baselines.qwen.make_sft --tasks data/real_train_tasks_${TAG}.jsonl \
    --symbols "$ARM" --out data/sft_real_train_${TAG}.jsonl
python -m baselines.qwen.mix_sft --out data/sft_${TAG}.jsonl --seed 20260912 \
    data/sft_${TAG}_gen.jsonl:1 data/open_pairs/b3_draw.jsonl:1 \
    data/sft_real_train_${TAG}.jsonl:3 ${FAM}:1
TAG=$TAG PRIOR=$PRIOR python - <<'EOF'
import collections, json, os
tag, prior = os.environ["TAG"], os.environ["PRIOR"]
tasks = [json.loads(l) for l in open(f'data/{tag}_tasks.jsonl', encoding='utf-8')]
print("level mix:", sorted(collections.Counter(t['level'] for t in tasks).items()))
print("crowded:", sum('crowded' in t['tags'] for t in tasks),
      "| abort:", sum(t['expected_status'] == 'aborted' for t in tasks),
      "| worlds:", len({t['world'] for t in tasks}))
recipes = collections.Counter(t['provenance']['recipe'] for t in tasks)
print("argmax:", recipes['argmax_count'], "| argmax questions:", recipes['argmax_which'],
      "| argmin:", recipes['argmin_count'], "| argmin questions:", recipes['argmin_which'])
print("membership:", recipes['parents_with'], "with,", recipes['parents_without'], "without")
progs = [t['reference']['segments'][0] for t in tasks]
print("MOST:", sum('MOST ' in p for p in progs),
      "| re-fetch after it:", sum(' IN r' not in p and 'MOST ' in p and 'FIRST ' in p for p in progs),
      "| IN:", sum(' IN r' in p for p in progs),
      "| EMPTY:", sum('EMPTY ' in p for p in progs))
bare = sum(p.strip() == 'ABORT NOT_FOUND' for p in progs)
checked = sum('ABORT NOT_FOUND ' in p for p in progs)
print(f"ABORT NOT_FOUND: {bare} bare, {checked} with a referent")
rows = [json.loads(l) for l in open(f'data/sft_{tag}.jsonl', encoding='utf-8')]
print("ABORT targets:", sum(r['messages'][-1]['content'].strip().startswith('ABORT')
                            for r in rows), "of", len(rows))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('baselines/qwen/models/lora_s2')
used = set(json.load(open(f'data/{prior}_used_token_ids.json')))
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
echo "Train:  CORPUS_TAG=${TAG} bash train_s4.sh ${ARM}"
echo "Score:  bash eval_s4.sh ${ARM} <gguf>   (the six R6 §0.1 suites)"
echo "        bash results/logs/eval_families.sh <gguf> ${ARM}   (each family's exam)"
