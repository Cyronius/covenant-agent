# Phase 2 corpora and reader tables (.claude/plans/npu-planner.md).
#
#   TERNLIGHT=<dir with node_modules/@ternlight> bash results/logs/gen_fdc.sh
#
# 1. fdc: decoys only on the four flip slots, and only where the request
#    tells the working tool from them (data.gen --decoy-slots flip), with
#    twin roles; two shards on S6's seeds for shards 1 and 2, mixed 25% /
#    75% with the options-off rows as R17/R19's mixes were.
# 2. The s6off and dtw25 caches rebuilt with today's prep.py, which also
#    saves the teacher table's texts. The rebuilt tensors must equal the old
#    ones; only then is the old teacher.pt replaced.
# 3. A Ternlight-mini reader table (reader.pt) for s6off, dtw25 and fdc25.
set -e
cd /c/code/covenant-agent
SURFACE="--symbols typed --enums --kinds"
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
BASE="--drop-noops --domains data/gen/themes $SURFACE"
PREP="--limit 30000 --holdout-corpus ../../data/s6_holdout_both.jsonl --names --desc-chars 120 --split \
      --max-line 112 --max-sig 64 --name-words --in-tok data_cache_s6g/in_tok.json --teacher unsloth/bge-small-en-v1.5"
mkdir -p data/s6opt_shards
export HF_HUB_OFFLINE=1

for i in 1 2; do
  seed=$((20260923 + i*10000000))
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $seed $BASE --decoys 1:2 --twin-roles \
      --decoy-slots flip --max-attempts 200 --allow-signature-unique \
      --out data/s6opt_shards/fdc_$i.jsonl > results/logs/gen_fdc_$i.log 2>&1 &
done
(cd models/tiny && for c in s6off dtw25; do
   python -W ignore prep.py --corpus ../../data/${c}_train.jsonl $PREP --out data_cache_${c}_rebuild \
       > ../../results/logs/prep_${c}_rebuild.log 2>&1
 done) &
wait
cat data/s6opt_shards/fdc_1.jsonl data/s6opt_shards/fdc_2.jsonl \
    data/s6off_shards/train_{0,3,4,5,6,7}.jsonl > data/fdc25_train.jsonl
wc -l data/s6opt_shards/fdc_*.jsonl data/fdc25_train.jsonl
python -W ignore results/logs/decoy_decidable.py data/s6opt_shards/fdc_1.jsonl data/s6opt_shards/fdc_2.jsonl

cd models/tiny
python -W ignore prep.py --corpus ../../data/fdc25_train.jsonl $PREP --out data_cache_fdc25 \
    > ../../results/logs/prep_fdc25.log 2>&1
for c in s6off dtw25; do
  python -W ignore - "$c" <<'EOF'
import sys, torch
c = sys.argv[1]
old, new = f"data_cache_{c}", f"data_cache_{c}_rebuild"
for split in ("train", "val", "test", "holdout"):
    a, b = torch.load(f"{old}/{split}.pt"), torch.load(f"{new}/{split}.pt")
    for k in a:
        if torch.is_tensor(a[k]) and not torch.equal(a[k], b[k]):
            raise SystemExit(f"{c} {split} {k}: rebuilt cache differs; old teacher.pt kept")
ta, tb = torch.load(f"{old}/teacher.pt"), torch.load(f"{new}/teacher.pt")
if not torch.equal(ta["table"], tb["table"]):
    raise SystemExit(f"{c}: rebuilt teacher table differs; old teacher.pt kept")
torch.save(tb, f"{old}/teacher.pt")
print(f"{c}: rebuilt cache identical on every tensor; teacher.pt now carries its {len(tb['texts'])} texts")
EOF
done
for c in s6off dtw25 fdc25; do
  python -W ignore reader_table.py --cache data_cache_$c --ternlight "$TERNLIGHT" --tier mini
done
echo "=== gen_fdc done $(date)"
