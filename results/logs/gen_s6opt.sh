# S6's three options one at a time (.claude/plans/option-split.md): two shards
# each, with S6's seeds for shards 1 and 2, to be mixed 25% / 75% with the
# options-off rows exactly as R17's s6d25 mix was (results/logs/mix_s6opt.sh).
#
#   dec    decoys only            --decoys 1:2
#   dtw    decoys + twin roles    --decoys 1:2 --twin-roles
#   opq    meaningless names only --decoys 0 --opaque-names 1.0 (every row of
#          these shards, so 25% of the mix; S6 had 15% of rows)
#
#   bash results/logs/gen_s6opt.sh    # -> data/s6opt_shards/{dec,dtw,opq}_{1,2}.jsonl
set -e
cd /c/code/covenant-agent
SURFACE="--symbols typed --enums --kinds"
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
BASE="--drop-noops --domains data/gen/themes $SURFACE"
mkdir -p data/s6opt_shards

for i in 1 2; do
  seed=$((20260923 + i*10000000))
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $seed $BASE --decoys 1:2 \
      --out data/s6opt_shards/dec_$i.jsonl > results/logs/gen_s6opt_dec_$i.log 2>&1 &
  # twin roles resample L9/L12 rows whose fixed wording cannot name a
  # swapped-in sibling; at the default 25 attempts both shards died FATAL
  # near 500 rows (2026-09-27), so 200. The rows before that point are the same.
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $seed $BASE --decoys 1:2 --twin-roles \
      --max-attempts 200 \
      --out data/s6opt_shards/dtw_$i.jsonl > results/logs/gen_s6opt_dtw_$i.log 2>&1 &
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $seed $BASE --decoys 0 \
      --allow-signature-unique --opaque-names 1.0 \
      --out data/s6opt_shards/opq_$i.jsonl > results/logs/gen_s6opt_opq_$i.log 2>&1 &
done
wait
wc -l data/s6opt_shards/*.jsonl
echo "=== gen_s6opt done $(date)"
