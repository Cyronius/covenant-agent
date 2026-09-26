# S6 with its three options off: the rewritten themes and today's generator,
# but no decoys, no twin roles and no opaque names -- S5's recipe on S6's
# themes (.claude/plans/s5-first-s6-cost-split.md, run 3). Same levels,
# surface, shard layout and seeds as gen_s6.sh's training rows; the decoy
# draws shift the random stream, so the tasks are a fresh sample, not S6's
# rows with the decoys stripped.
#
#   bash results/logs/gen_s6off.sh         # -> data/s6off_train.jsonl
#
# --decoys 0 needs --allow-signature-unique, as in gen_s5.sh and every plain
# exam: undecoyed rows are signature-unique by construction.
set -e
cd /c/code/covenant-agent
SURFACE="--symbols typed --enums --kinds"
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
mkdir -p data/s6off_shards

for i in 0 1 2 3 4 5 6 7; do
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260923 + i*10000000)) \
      --drop-noops --domains data/gen/themes $SURFACE --decoys 0 --allow-signature-unique \
      --out data/s6off_shards/train_$i.jsonl > results/logs/gen_s6off_train_$i.log 2>&1 &
done
wait
cat data/s6off_shards/train_[0-7].jsonl > data/s6off_train.jsonl
wc -l data/s6off_shards/train_[0-7].jsonl data/s6off_train.jsonl
echo "=== gen_s6off done $(date) ==="
