# S3 corpus (results/S2.md §Lane C "Next"): one fresh rebuild that fixes the
# two things S2/S2R got wrong, instead of stacking another continuation.
#
#  1. E-crowded (S2 81.3 vs S1 94.7, bar 90). S2 kept S1's 15k crowded tasks
#     but spread them over 19 levels instead of 11 -- ~790 per level against
#     S1's ~1,360. S3 draws 26k crowded (8 shards x 3,250) to restore S1's
#     per-level coverage, and finish_s3_corpus.sh sets the training cap from
#     the measured max so no crowded row loses its answer to truncation.
#  2. Real requests. The 900 verified real-turn rows (harness/real_train_build.py)
#     go in from the start at ~4% (x3, no 25%-share continuation), so the
#     abstain trigger is learned alongside the curriculum, not pushed on top
#     of it -- that is what over-fired on S2R (9/65 false abstains on demo).
#
# Same level weights as S2, new seeds. Run from the repo root in Git Bash:
#   bash results/logs/gen_s3.sh
set -e
cd /c/code/covenant-agent
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8"
mkdir -p data/s3_shards
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260905 + i*10000000)) --drop-noops \
      --domains data/gen/themes --out data/s3_shards/plain_$i.jsonl > results/logs/gen_s3_plain_$i.log 2>&1 &
done
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 3250 --seed $((20260906 + i*10000000)) --drop-noops \
      --domains data/gen/themes --crowd 15:60 --out data/s3_shards/crowded_$i.jsonl > results/logs/gen_s3_crowded_$i.log 2>&1 &
done
wait
cat data/s3_shards/plain_*.jsonl > data/s3_plain.jsonl
cat data/s3_shards/crowded_*.jsonl > data/s3_crowded.jsonl
cat data/s3_plain.jsonl data/s3_crowded.jsonl > data/s3_tasks.jsonl
wc -l data/s3_plain.jsonl data/s3_crowded.jsonl data/s3_tasks.jsonl
python -m baselines.qwen.make_sft --tasks data/s3_tasks.jsonl --domains data/gen/themes --out data/sft_s3_gen.jsonl

bash results/logs/finish_s3_corpus.sh
echo "=== gen_s3 done ==="
