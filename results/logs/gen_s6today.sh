# S6's training rows regenerated with today's generator (.claude/plans/
# decoy-fraction.md). The same recipe, levels, surface and seeds as
# gen_s6.sh's training half: decoys 1:2 on every mutating tool, twin roles,
# opaque names on 15% of rows.
#
# Two uses:
#   - the whole file is the handoff's generator-drift check (R14 caveat 2):
#     A0 on it against A0 on the original S6 shards (R13, 47.1% plain)
#   - its shards are the decoyed part of the fraction mixes
#     (results/logs/mix_s6d.sh)
#
#   bash results/logs/gen_s6today.sh     # -> data/s6today_shards/train_[0-7].jsonl
set -e
cd /c/code/covenant-agent
SURFACE="--symbols typed --enums --kinds"
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
COMMON="--drop-noops --domains data/gen/themes --twin-roles --opaque-names 0.15 $SURFACE"
mkdir -p data/s6today_shards

for i in 0 1 2 3 4 5 6 7; do
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260923 + i*10000000)) \
      $COMMON --decoys 1:2 \
      --out data/s6today_shards/train_$i.jsonl > results/logs/gen_s6today_train_$i.log 2>&1 &
done
wait
cat data/s6today_shards/train_[0-7].jsonl > data/s6today_train.jsonl
wc -l data/s6today_shards/train_[0-7].jsonl data/s6today_train.jsonl
echo "=== gen_s6today done $(date)"
