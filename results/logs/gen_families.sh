# The task-family corpus and its exams (.claude/plans/archive/task-families.md).
#
#   bash results/logs/gen_families.sh            # typed  -> data/fam_tasks.jsonl
#   bash results/logs/gen_families.sh classic    # control -> data/famc_tasks.jsonl
#
# The finding this exists for: the 0.8B transfers across domains (95% on 143
# unseen themes) and not across task families (0/18 dungeon episodes). Every
# row of the 62,841-row corpus is one family - find records, filter them, act
# on them - so this adds the rest of the map:
#
#   A  decision episodes in warehouse / elevator / cards, labelled by their
#      oracles from the state the episode is actually in, including states a
#      deliberately wrong move produced (data/gen/episodes.py)
#   B  decoy tools: every mutating tool in the existing 143 themes gets two
#      to four siblings with its signature and a neighbouring description,
#      so only the description says which one the request means
#   C  the same episode recipe in three page apps (family A in a screen's
#      clothing); the coursebuilder-shaped app is held out
#   D  ask, then act: the NEEDS_INFO turn, the turn after the answer, and
#      the same job stated in full so no question is called for - that last
#      one is what keeps the family from being half abstentions
#   E  the continuation the error code you were just shown calls for
#
# Family G is not here on purpose: `python -m harness.schedule_probe` shows
# it is blocked on the IR, not on data.
#
# Held out throughout: the dungeon, the house, the coursebuilder app and the
# scheduling probe (data/holdout/reserved.json). The exams at the bottom are
# built from those.
#
# Run from the repo root in Git Bash.
set -e
cd /c/code/covenant-agent
ARM="${1:-typed}"
case "$ARM" in
  typed)   SURFACE="--symbols typed --enums --kinds"; TAG=fam ;;
  classic) SURFACE="";                                TAG=famc ;;
  *) echo "usage: gen_families.sh [typed|classic]"; exit 2 ;;
esac
SEED=20260910
mkdir -p data/${TAG}_shards results/logs/fam

# --- families A and C: oracle episodes with off-path restarts -------------
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen.episodes --world all --episodes 20 \
      --seed $((SEED + i * 1000)) --offpath 0.25 --illegal 0.3 $SURFACE \
      --out data/${TAG}_shards/episodes_$i.jsonl \
      > results/logs/fam/gen_${TAG}_episodes_$i.log 2>&1 &
done

# --- family B: the same recipes, with description-only decoys -------------
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8"
for i in 0 1 2 3; do
  python -m data.gen --levels "$LEVELS" --n 1500 --seed $((SEED + 100 + i * 1000)) \
      --drop-noops --decoys 2:4 $SURFACE --domains data/gen/themes \
      --out data/${TAG}_shards/decoy_$i.jsonl \
      > results/logs/fam/gen_${TAG}_decoy_$i.log 2>&1 &
done

# --- family D: ask, then act ---------------------------------------------
for i in 0 1; do
  python -m data.gen.askact --n 500 --seed $((SEED + 200 + i * 1000)) \
      $SURFACE --domains data/gen/themes \
      --out data/${TAG}_shards/askact_$i.jsonl \
      > results/logs/fam/gen_${TAG}_askact_$i.log 2>&1 &
done

# --- family E: recovery by error code ------------------------------------
for i in 0 1; do
  python -m data.gen.recovery --n 1500 --seed $((SEED + 300 + i * 1000)) \
      $SURFACE --domains data/gen/themes \
      --out data/${TAG}_shards/recovery_$i.jsonl \
      > results/logs/fam/gen_${TAG}_recovery_$i.log 2>&1 &
done
wait

cat data/${TAG}_shards/episodes_*.jsonl > data/${TAG}_episodes.jsonl
cat data/${TAG}_shards/decoy_*.jsonl    > data/${TAG}_decoy.jsonl
cat data/${TAG}_shards/askact_*.jsonl   > data/${TAG}_askact.jsonl
cat data/${TAG}_shards/recovery_*.jsonl > data/${TAG}_recovery.jsonl
cat data/${TAG}_episodes.jsonl data/${TAG}_decoy.jsonl \
    data/${TAG}_askact.jsonl data/${TAG}_recovery.jsonl > data/${TAG}_tasks.jsonl
wc -l data/${TAG}_episodes.jsonl data/${TAG}_decoy.jsonl \
      data/${TAG}_askact.jsonl data/${TAG}_recovery.jsonl data/${TAG}_tasks.jsonl

python -m baselines.qwen.make_sft --tasks data/${TAG}_tasks.jsonl \
    --domains data/gen/themes --symbols "$ARM" --out data/sft_${TAG}.jsonl

TAG=$TAG python - <<'EOF'
import collections, json, os
tag = os.environ["TAG"]
tasks = [json.loads(l) for l in open(f'data/{tag}_tasks.jsonl', encoding='utf-8')]
print("levels:", sorted(collections.Counter(t['level'] for t in tasks).items()))
fam = collections.Counter(tag for t in tasks for tag in t['tags']
                          if tag.startswith('family:'))
print("families:", dict(sorted(fam.items())))
print("decision worlds:", dict(sorted(collections.Counter(
    t['world'] for t in tasks if t['level'] in (19, 20)).items())))
print("off-path turns:", sum(bool(t['provenance'].get('off_path_move'))
                             for t in tasks))
print("turns that open with a failure:",
      sum('failed:' in t['request'] for t in tasks))
print("abort targets:", sum(t['expected_status'] == 'aborted' for t in tasks))
print("decoyed rows:", sum('decoyed' in t['tags'] for t in tasks))
EOF

# --- the exams -----------------------------------------------------------
# One set per arm: a typed-trained model scored on classic prompts looks
# broadly broken (R6 section 0), so the two must never share a file.
SFX=""; [ "$ARM" = typed ] && SFX="_typed"
# Family B's number is a gap, so the pair has to be paired: same seed, same
# worlds, same programs, one with decoys and one without.
python -m data.gen --levels "$LEVELS" --n 400 --seed 777001 --drop-noops \
    $SURFACE --domains data/gen/themes --out data/holdout/e_known_plain${SFX}.jsonl
python -m data.gen --levels "$LEVELS" --n 400 --seed 777001 --drop-noops \
    --decoys 2:4 $SURFACE --domains data/gen/themes \
    --out data/holdout/e_known_decoy${SFX}.jsonl
python -m data.gen.askact --n 100 --seed 777002 --holdout $SURFACE \
    --domains data/gen/themes --out data/holdout/e_askact${SFX}.jsonl
python -m data.gen.recovery --n 300 --seed 777003 --holdout $SURFACE \
    --domains data/gen/themes --out data/holdout/e_recovery${SFX}.jsonl

echo
echo "Families A and C are episodes, so their exam is the suite, not a file:"
echo "  python -m harness.rpg_suite --world rpg               --model <gguf> ..."
echo "  python -m harness.rpg_suite --world house             --model <gguf> ..."
echo "  python -m harness.rpg_suite --world app_coursebuilder --model <gguf> ..."
echo "  (results/logs/eval_families.sh runs all three plus the file suites)"
echo "=== gen_families ($ARM) done ==="
