# S6b: R11 §7 item 2 -- close step 3's 2.1-point gap at zero cost.
#
# FAILED, kept as the record (R11 §3b): every shard hit "FATAL: level 9 failed
# 25 consecutive attempts" under 7c3e765's generator, 1,701 of 30,000 rows;
# set -e does not see a backgrounded generator die, so prep and both arms ran
# on that and are void. Superseded by stages_u.sh.
#
#   bash results/logs/gen_s6b.sh    # -> data/s6b_train.jsonl, data_cache_s6b, runs/stages_y1, y2
#
# Only the TRAINING corpus is regenerated: same command and seeds as gen_s6.sh,
# with the generator as of 7c3e765, which rejects flip-slot calls whose request
# never carries the called tool's own templates (14% of S6's, all answered by
# the authored tool). The exam, data/s6_holdout_both.jsonl, is reused
# byte-for-byte so every number compares directly with X4 and the teacher.
#
# Two arms on X4's exact recipe:
#   Y1  3 epochs  -- X4 vs Y1 isolates the data fix
#   Y2  6 epochs  -- Y1 vs Y2 isolates duration. X4's flat final 500 steps
#                    were the cosine schedule reaching zero, so they show
#                    neither saturation nor headroom.
# Each is graded on the full exam by stage_quant (fp only), like X4's
# full_exam.json. Gate: acc_flip_comb >= acc_flip_t_desc (teacher 89.9%).
set -e
cd /c/code/covenant-agent
SURFACE="--symbols typed --enums --kinds"
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
COMMON="--drop-noops --domains data/gen/themes --twin-roles --opaque-names 0.15 $SURFACE"
mkdir -p data/s6b_shards

for i in 0 1 2 3 4 5 6 7; do
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260923 + i*10000000)) \
      $COMMON --decoys 1:2 \
      --out data/s6b_shards/train_$i.jsonl > results/logs/gen_s6b_train_$i.log 2>&1 &
done
wait
cat data/s6b_shards/train_[0-7].jsonl > data/s6b_train.jsonl
wc -l data/s6b_shards/train_[0-7].jsonl data/s6b_train.jsonl
echo "=== gen done $(date) ==="

cd models/tiny
python prep.py --corpus ../../data/s6b_train.jsonl --limit 30000 \
    --holdout-corpus ../../data/s6_holdout_both.jsonl \
    --names --desc-chars 120 --split --max-line 112 --max-sig 64 \
    --teacher unsloth/bge-small-en-v1.5 --out data_cache_s6b \
    > ../../results/logs/prep_s6b.log 2>&1
echo "=== prep done $(date) ==="

RECIPE="--desc-w 128 --pool mean --init-emb teacher --lam-rel 20 --flip-weight 3 --limit-eval 2000 --eval-every 400"
for arm in "y1 3" "y2 6"; do
  set -- $arm
  python -W ignore stage_pretrain.py --cache data_cache_s6b --out runs/stages_$1 $RECIPE --epochs $2 \
      > ../../results/logs/stages_$1.log 2>&1
  python -W ignore stage_quant.py --stages runs/stages_$1/stages.pt --cache data_cache_s6b \
      --limit-eval 100000 --modes fp --json runs/stages_$1/full_exam.json \
      > ../../results/logs/stages_$1_full.log 2>&1
  echo "=== $1 done $(date) ==="
done
echo "=== gen_s6b done ==="
