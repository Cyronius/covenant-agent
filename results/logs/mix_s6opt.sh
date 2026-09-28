# The option-split mixes (.claude/plans/option-split.md): 25% of rows from
# gen_s6opt.sh's two shards for one option, the rest options-off shards 0 and
# 3-7, 30,000 rows, as mix_s6d.sh builds s6d25. No seed appears twice.
#
#   bash results/logs/mix_s6opt.sh   # after gen_s6opt.sh
set -e
cd /c/code/covenant-agent/data
for opt in dec dtw opq; do
  cat s6opt_shards/${opt}_1.jsonl s6opt_shards/${opt}_2.jsonl \
      s6off_shards/train_{0,3,4,5,6,7}.jsonl > ${opt}25_train.jsonl
done
wc -l dec25_train.jsonl dtw25_train.jsonl opq25_train.jsonl
