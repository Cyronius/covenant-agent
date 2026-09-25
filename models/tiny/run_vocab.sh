# The vocabulary push on one GPU (.claude/plans/vocab-push.md).
#
#   bash run_vocab.sh            # from tiny/, after unpacking the bundle
#
# 1. general.pt for both caches (the teacher embeds the general corpus once,
#    on the GPU, and both caches reuse its vectors)
# 2. four arms side by side, 3 epochs of S6 each, U2's recipe otherwise:
#      V1      general-English tokenizer only              (data_cache_s6g)
#      V2      + general text taught by the teacher         (data_cache_s6g)
#      V3      + dictionary look-alikes                     (data_cache_s6g)
#      V2-30k  V2 on the teacher's 30,522-piece vocabulary  (data_cache_s6t)
# 3. the best of V1-V3 (by the full exam, last checkpoint) again at 6 epochs
# 4. vocab_report.py on every arm: all / seen-word / unseen-word decisions
#
# Reported numbers come from each arm's LAST checkpoint: stages.pt is picked
# on the exam's first 2,000 rows, which would put a selection on the test.
set -e
cd "$(dirname "$0")"
GENERAL="${GENERAL:-../general}"
GEN_STEPS="${GEN_STEPS:-3000}"
THREADS="${THREADS:-2}"
mkdir -p runs/vocab logs

for c in data_cache_s6g data_cache_s6t; do
  python -W ignore general_prep.py --cache $c --general "$GENERAL" > logs/general_prep_$c.log 2>&1
  tail -1 logs/general_prep_$c.log
done

BASE="--desc-w 128 --pool mean --init-emb teacher --lam-rel 20 --flip-weight 3 --drop-unreadable \
      --limit-eval 2000 --eval-every 800 --threads $THREADS"
GEN="--general --gen-steps $GEN_STEPS --gen-every 2"
arm() {  # name cache epochs extra...
  local name=$1 cache=$2 ep=$3; shift 3
  python -W ignore stage_pretrain.py --cache $cache --out runs/vocab/$name $BASE --epochs $ep "$@" \
      > logs/stages_$name.log 2>&1
  python -W ignore vocab_report.py --cache $cache --words train_words_s6.json \
      --stages runs/vocab/$name/stages_last.pt runs/vocab/$name/stages.pt \
      --json runs/vocab/$name/report.json > logs/report_$name.log 2>&1
  echo "=== $name done $(date)"; cat logs/report_$name.log
}
arm v1 data_cache_s6g 3 &
arm v2 data_cache_s6g 3 $GEN &
arm v3 data_cache_s6g 3 $GEN --siblings &
arm v2_30k data_cache_s6t 3 $GEN &
wait

BEST=$(python - <<'EOF'
import json
best = max(("v1", "v2", "v3"), key=lambda a: json.load(open(f"runs/vocab/{a}/report.json"))
           ["stages"][f"runs/vocab/{a}/stages_last.pt"]["all"]["acc"])
print(best)
EOF
)
echo "best of V1-V3 at 3 epochs: $BEST"
case $BEST in
  v1) arm ${BEST}_6ep data_cache_s6g 6 ;;
  v2) arm ${BEST}_6ep data_cache_s6g 6 $GEN ;;
  v3) arm ${BEST}_6ep data_cache_s6g 6 $GEN --siblings ;;
esac
tar czf vocab_results.tar.gz runs/vocab logs
echo "=== run_vocab done $(date)"
