# Borrowed-worlds retrain caches (.claude/plans/borrowed-worlds.md step 6).
# Three mixes of R23's clt_train with the new kinds at ~10/20/30%
# (results/logs/mix_brw.py), prepped with gen_clt.sh's flags and the same
# regression exam (clt_holdout_both: plain, decoy, flip, cluttered) so a
# drop on the old exams is visible beside any gain on the new ones.
# The new-kind exams are scored separately after training:
#   data/holdout/e_brief_{boatyard,rooms_after,house}.jsonl   (play.py)
#   data/holdout/e_service_telecom.jsonl                       (play.py)
#   harness.rpg_suite --planner tiny --exits --world <w>       (live episodes)
#
#   TERNLIGHT=models/tiny/reader bash results/logs/gen_brw.sh
set -e
cd /c/code/covenant-agent
export HF_HUB_OFFLINE=1
python results/logs/mix_brw.py
python data/borrowed/canary_check.py data/brw10_train.jsonl data/brw20_train.jsonl data/brw30_train.jsonl
PREP="--limit 30000 --holdout-corpus ../../data/clt_holdout_both.jsonl --names --desc-chars 120 --split \
      --max-line 112 --max-sig 64 --name-words --in-tok data_cache_s6g/in_tok.json --teacher unsloth/bge-small-en-v1.5"
cd models/tiny
for m in brw10 brw20 brw30; do
  python -W ignore prep.py --corpus ../../data/${m}_train.jsonl $PREP --out data_cache_$m \
      > ../../results/logs/prep_$m.log 2>&1
  python -W ignore reader_table.py --cache data_cache_$m --ternlight "${TERNLIGHT:-reader}" --tier mini \
      >> ../../results/logs/prep_$m.log 2>&1
  grep -E "layout|longest|kept|dropped" ../../results/logs/prep_$m.log | head -8
done
echo "=== gen_brw done $(date)"
