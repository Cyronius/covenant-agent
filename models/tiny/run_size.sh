# Vocabulary size and stage width, on top of R12's V2 recipe (general English,
# 6 epochs of S6), all four arms side by side on one GPU:
#
#   base_s1   4,096 pieces, 128 wide, seed 1  (V2 again: the noise between runs)
#   v8k       8,192 pieces, 128 wide
#   w256      4,096 pieces, 256 wide (both stages)
#   v8k_w256  8,192 pieces, 256 wide
#
#   bash run_size.sh     # from tiny/, after unpacking the bundle
#
# Graded like R12: vocab_report.py on each LAST checkpoint.
set -e
cd "$(dirname "$0")"
GENERAL="${GENERAL:-../general}"
mkdir -p runs/size logs
for c in data_cache_s6g data_cache_s6g8k; do
  python -W ignore general_prep.py --cache $c --general "$GENERAL" > logs/general_prep_$c.log 2>&1
  tail -1 logs/general_prep_$c.log
done

BASE="--pool mean --init-emb teacher --lam-rel 20 --flip-weight 3 --drop-unreadable \
      --limit-eval 2000 --eval-every 800 --threads 2 --general --gen-steps 3000 --gen-every 2 --epochs 6"
arm() {  # name cache extra...
  local name=$1 cache=$2; shift 2
  python -W ignore stage_pretrain.py --cache $cache --out runs/size/$name $BASE "$@" \
      > logs/stages_$name.log 2>&1
  python -W ignore vocab_report.py --cache $cache --words train_words_s6.json \
      --stages runs/size/$name/stages_last.pt --json runs/size/$name/report.json \
      > logs/report_$name.log 2>&1
  echo "=== $name done $(date)"; grep -v -E "Loading|general English" logs/report_$name.log
}
arm base_s1  data_cache_s6g   --desc-w 128 --name-w 128 --seed 1 &
arm v8k      data_cache_s6g8k --desc-w 128 --name-w 128 &
arm w256     data_cache_s6g   --desc-w 256 --name-w 256 &
arm v8k_w256 data_cache_s6g8k --desc-w 256 --name-w 256 &
wait
tar czf size_results.tar.gz runs/size logs
echo "=== run_size done $(date)"
