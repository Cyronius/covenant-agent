# S6: the description-reading corpus (.claude/plans/description-reading.md
# step 1). Same levels and typed surface as S5; what changes is what a twin
# decision can be passed by.
#
#   bash results/logs/gen_s6.sh            # -> data/s6_train.jsonl, data/s6_holdout_both.jsonl
#
# Every theme authors its decoys and its nine generic tools in its own voice
# (data/gen/THEME_SCHEMA.md "Decoys"); --decoys 1:2 draws from those, never
# the template bank, whose style a description-only classifier separated at
# AUC 1.000 (results/R10.md section 8). --twin-roles makes a decoy's text as
# likely to be the answer as the authored tool's on the four flip slots, so
# "the canonical action is always right" is not a shortcut either.
# --opaque-names 0.15 renames every tool on 15% of rows, so reading the
# description stays measurable once names are model input (spec 0.8.0).
#
# Training is decoyed on every row (1f's default rate). R10 trained on 0
# decoyed rows and was examined on decoys only; a planner that has never had
# to choose between twins was being asked to do it for the first time on the
# exam.
#
# The exam is three things over the 42 reserved worlds, assembled into ONE
# holdout corpus with suffixed ids (prep refuses repeated ids):
#   plain     undecoyed; continuity with the plain-exam goal line
#   +decoy    the same tasks, decoyed -- the grounding measurement
#   +flip     every row's answer is a sibling the theme did not author as the
#             tool (data/gen/flip_probe.py)
# Gate before anything trains on it: python -m harness.decoy_audit
set -e
cd /c/code/covenant-agent
SURFACE="--symbols typed --enums --kinds"
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
COMMON="--drop-noops --domains data/gen/themes --twin-roles --opaque-names 0.15 $SURFACE"
mkdir -p data/s6_shards

for i in 0 1 2 3 4 5 6 7; do
  python -W ignore -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260923 + i*10000000)) \
      $COMMON --decoys 1:2 \
      --out data/s6_shards/train_$i.jsonl > results/logs/gen_s6_train_$i.log 2>&1 &
done
for i in 0 1 2 3 4 5 6 7; do
  python -W ignore -m data.gen --levels "$LEVELS" --n 500 --seed $((20260924 + i*10000000)) \
      --holdout $COMMON --decoys 0 --allow-signature-unique \
      --out data/s6_shards/holdout_$i.jsonl > results/logs/gen_s6_holdout_$i.log 2>&1 &
  python -W ignore -m data.gen --levels "$LEVELS" --n 500 --seed $((20260924 + i*10000000)) \
      --holdout $COMMON --decoys 1:2 \
      --out data/s6_shards/holdout_decoy_$i.jsonl > results/logs/gen_s6_holdout_decoy_$i.log 2>&1 &
done
python -W ignore -m data.gen.flip_probe --n 400 --opaque-names 0.15 \
    --out data/s6_shards/flip.jsonl > results/logs/gen_s6_flip.log 2>&1 &
wait
cat data/s6_shards/train_[0-7].jsonl > data/s6_train.jsonl
cat data/s6_shards/holdout_[0-7].jsonl > data/s6_holdout.jsonl
cat data/s6_shards/holdout_decoy_[0-7].jsonl > data/s6_holdout_decoy.jsonl
cp data/s6_shards/flip.jsonl data/s6_flip.jsonl
python - <<'EOF'
import json
from pathlib import Path
with Path("data/s6_holdout_both.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
    for src, suf in (("data/s6_holdout.jsonl", ""), ("data/s6_holdout_decoy.jsonl", "+decoy"),
                     ("data/s6_flip.jsonl", "")):
        for line in Path(src).open(encoding="utf-8"):
            r = json.loads(line)
            if suf:
                r["id"] += suf
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF
wc -l data/s6_train.jsonl data/s6_holdout.jsonl data/s6_holdout_decoy.jsonl data/s6_flip.jsonl data/s6_holdout_both.jsonl
python -m harness.signature_uniqueness data/s6_holdout_decoy.jsonl
echo "=== gen_s6 done ==="
