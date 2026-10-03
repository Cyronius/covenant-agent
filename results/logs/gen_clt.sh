# R23: fdc25's corpus with the demo host's kind of constants list
# (data.gen --clutter, data/gen/clutter.py). Same seeds, levels, flags and
# 25% / 75% mix as gen_fdc.sh + gen_s6off.sh; clutter draws from its own RNG,
# so every row is the fdc25 row with the same request and program and a
# different constants list.
#
#   TERNLIGHT=models/tiny/reader bash results/logs/gen_clt.sh
#
# Also a cluttered copy of the plain exam (s6_holdout.jsonl's seeds and flags,
# ids suffixed +clut), appended to the S6 exam as clt_holdout_both.jsonl, so
# the pod scores plain / decoy / flip as R22 did plus the cluttered half.
set -e
cd /c/code/covenant-agent
SURFACE="--symbols typed --enums --kinds"
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
BASE="--drop-noops --domains data/gen/themes $SURFACE --clutter"
PREP="--limit 30000 --holdout-corpus ../../data/clt_holdout_both.jsonl --names --desc-chars 120 --split \
      --max-line 112 --max-sig 64 --name-words --in-tok data_cache_s6g/in_tok.json --teacher unsloth/bge-small-en-v1.5"
mkdir -p data/clt_shards
export HF_HUB_OFFLINE=1

for i in 1 2; do
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260923 + i*10000000)) $BASE \
      --decoys 1:2 --twin-roles --decoy-slots flip --max-attempts 200 --allow-signature-unique \
      --out data/clt_shards/fdc_$i.jsonl > results/logs/gen_clt_fdc_$i.log 2>&1 &
done
for i in 0 3 4 5 6 7; do
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260923 + i*10000000)) $BASE \
      --decoys 0 --allow-signature-unique \
      --out data/clt_shards/off_$i.jsonl > results/logs/gen_clt_off_$i.log 2>&1 &
done
for i in 0 1 2 3 4 5 6 7; do
  python -W ignore -m data.gen --levels "$LEVELS" --n 500 --seed $((20260924 + i*10000000)) \
      --holdout $BASE --twin-roles --opaque-names 0.15 --decoys 0 --allow-signature-unique \
      --out data/clt_shards/holdout_$i.jsonl > results/logs/gen_clt_holdout_$i.log 2>&1 &
done
wait
cat data/clt_shards/fdc_1.jsonl data/clt_shards/fdc_2.jsonl \
    data/clt_shards/off_{0,3,4,5,6,7}.jsonl > data/clt_train.jsonl
cat data/clt_shards/holdout_[0-7].jsonl > data/clt_holdout.jsonl
python - <<'EOF'
import json
from pathlib import Path
with Path("data/clt_holdout_both.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
    for line in Path("data/s6_holdout_both.jsonl").open(encoding="utf-8"):
        fh.write(line)
    for line in Path("data/clt_holdout.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        r["id"] += "+clut"
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF
wc -l data/clt_train.jsonl data/clt_holdout.jsonl data/clt_holdout_both.jsonl

cd models/tiny
python -W ignore prep.py --corpus ../../data/clt_train.jsonl $PREP --out data_cache_clt \
    > ../../results/logs/prep_clt.log 2>&1
python -W ignore reader_table.py --cache data_cache_clt --ternlight "$TERNLIGHT" --tier mini
echo "=== gen_clt done $(date)"
