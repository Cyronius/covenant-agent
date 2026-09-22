# S5-inject: the plain S5 shards, with imported open schemas injected as
# distractors (.claude/plans/imported-schemas-as-distractors.md).
#
#   bash results/logs/gen_s5_inject.sh            # 42 distractors -> data/s5_inject.jsonl
#   bash results/logs/gen_s5_inject.sh 42 classic # the classic-symbol control
#   N_INJECT=18 bash results/logs/gen_s5_inject.sh
#
# Same levels, same surface and the SAME SEEDS as gen_s5.sh's plain half, so
# data/s5_inject.jsonl pairs row-for-row with data/s5_plain.jsonl: same world,
# same state, same request, same reference frame. --inject-open draws from
# random.Random(seed ^ 0xFEED) rather than the shared rng for exactly that
# reason (decision 4; --crowd does not, and its pairs diverge).
#
# The injected corpus is ~4.4x the plain one on disk (about 2.4 GB at 30,000
# rows) because every row carries 60 tool lines and ~150 field lines instead
# of 18 and 23. Check free space before running. One shard takes ~20 min here.
#
# What the pair is for: step 3 re-baselines both arms x 3 seeds on the
# widened holdout WITHOUT injection, step 4 repeats it WITH. The question is
# whether unseen-world binding transfer improves and whether the control
# arm's 74-95 spread narrows (results/R8.md:227-232).
#
# THE INJECTED CACHE NEEDS A WIDER LINE BUDGET. models/tiny/prep.py defaults
# to --max-line 64, which was tuned to themed lines (longest 56 tokens).
# Foreign vocabulary tokenizes into more pieces in a 4,096-BPE trained on the
# corpus, so the longest injected line runs to 66. prep refuses to truncate
# and stops, so pass the wider budget explicitly:
#
#   cd models/tiny
#   python prep.py --corpus ../../data/s5_plain.jsonl --limit 30000 --holdout-worlds 5 --holdout-seed 0 --out data_cache_wide
#   HO=$(python -c "import json;print(','.join(json.load(open('data_cache_wide/config.json'))['holdout_worlds']))")
#   python prep.py --corpus ../../data/s5_inject.jsonl --limit 30000 --max-line 80 --holdout-world "$HO" --out data_cache_inject
#   bash selftest.sh data_cache_inject
#
# Draw the holdout worlds once and NAME them for the second cache rather than
# re-drawing with the same seed. The draw ranks worlds by row count, and any
# difference in what the two corpora drop would silently move a world across
# the band boundary: step 3's baseline and step 4's injected arm are only
# comparable on one split.
set -e
cd /c/code/covenant-agent
N_INJECT="${N_INJECT:-${1:-42}}"
ARM="${2:-typed}"
case "$ARM" in
  typed)   SURFACE="--symbols typed --enums --kinds"; TAG=s5 ;;
  classic) SURFACE="";                                TAG=s5c ;;
  *) echo "usage: gen_s5_inject.sh [N_INJECT] [typed|classic]"; exit 2 ;;
esac
# --decoys 0 --allow-signature-unique (1f, 2026-09-21): predates the
# mandatory collision ceiling; reproduces this plain (undecoyed) record
# exactly, with the escape stamped into every row's provenance.
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
mkdir -p data/${TAG}_shards
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260912 + i*10000000)) --drop-noops \
      --decoys 0 --allow-signature-unique \
      $SURFACE --domains data/gen/themes --inject-open ${N_INJECT}:${N_INJECT} \
      --out data/${TAG}_shards/inject_$i.jsonl > results/logs/gen_${TAG}_inject_$i.log 2>&1 &
done
wait
cat data/${TAG}_shards/inject_*.jsonl > data/${TAG}_inject.jsonl
wc -l data/${TAG}_inject.jsonl
# the plain half is the comparison, not an input: its absence is not a failure
if [ -f data/${TAG}_plain.jsonl ]; then
  wc -l data/${TAG}_plain.jsonl
else
  echo "note: data/${TAG}_plain.jsonl absent - run gen_s5.sh for the control half"
fi
echo "=== gen_s5_inject ($ARM, $N_INJECT distractors) done ==="
