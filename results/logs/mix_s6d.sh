# The fraction mixes (.claude/plans/decoy-fraction.md): whole shards of S6
# (all three S6 options) and of the options-off rows, 30,000 rows in all.
# Both corpora use the same eight shard seeds, so each mix takes different
# shard numbers from each and no seed appears twice. S6's shard 0 stops at
# 2,489 rows (as the regenerated one does, gen_s6today.sh), so it is left out
# and every shard used has 3,750 rows.
# prep.py shuffles before splitting (models/tiny/corpus.py:261).
#
#   bash results/logs/mix_s6d.sh   # after gen_s6.sh and gen_s6off.sh
set -e
cd /c/code/covenant-agent/data
cat s6_shards/train_{1,2}.jsonl s6off_shards/train_{0,3,4,5,6,7}.jsonl > s6d25_train.jsonl
cat s6_shards/train_{1,2,3,4}.jsonl s6off_shards/train_{0,5,6,7}.jsonl > s6d50_train.jsonl
wc -l s6d25_train.jsonl s6d50_train.jsonl
