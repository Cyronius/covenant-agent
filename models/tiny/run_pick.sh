# Step 2d of .claude/plans/staged-decoder-experts.md: the constant picker.
# About 90% of the staged decoder's wrong constants are decoys, constants in
# the context the program doesn't use (R28 point 11). The picker scores each
# constant once, before decoding, and the head adds that score to the
# constant pointers (model.py, Config.pick). One GPU:
#
#   d3_DRp_s{0,1}     DR + picker      vs d3_DR
#   d3_D2Rp_s{0,1}    D2R + picker     vs d3_D2R
#   (d3_ctrl6p_s{0,1}, ctrl6 + picker, dropped by the owner: a perfect picker
#   buys left to right only +2.1, R28 point 13; RUNS= can still name it)
#
# Recipe, generations and archive as run_staged.sh. A slot is held only while
# a run TRAINS (generation is light), so the next run starts as soon as one
# finishes training (R28's runners held slots through generation and left the
# GPU idle). Scoring runs on the laptop (results/logs/score_staged.sh).
#
#   bash run_pick.sh      # from tiny/
set -uo pipefail
cd "$(dirname "$0")"
RUNS="${RUNS:-d3_DRp_s0 d3_D2Rp_s0 d3_DRp_s1 d3_D2Rp_s1}"
SMOKE="${SMOKE-d3_DRp_s0}"
JOBS="${JOBS:-3}"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

REC="--arm ar --cache data_cache_d3 --epochs 12 --batch 64 --pad-weight 0.5 --pack --pick"
flags() {  # run
  local s="--seed ${1##*_s}"
  case "$1" in
    *_DRp_*)    echo "$s --stages draft:4x4,refine:2x1" ;;
    *_D2Rp_*)   echo "$s --stages draft:2x4,draft:2x4,refine:2x1" ;;
    *_ctrl6p_*) echo "$s --dec-layers 6" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
gen() {  # run split tag selector...
  local run=$1 split=$2 tag=$3; shift 3
  [ -f out/${run}_${tag}.jsonl ] && return 0
  python evaluate.py --generate --ckpt runs/$run/best.pt --cache data_cache_d3 --split $split \
      --limit 2000 "$@" --gen-out out/${run}_${tag}.jsonl
}
arm() {  # run
  local run=$1
  if [ ! -f runs/$run/done ]; then
    # shellcheck disable=SC2086,SC2046
    python train.py $REC $(flags $run) --out runs/$run || { echo "train $run failed"; return 1; }
    touch runs/$run/done
  fi
  cp runs/$run/config.json out/${run}_config.json
  cp runs/$run/log.jsonl out/${run}_curve.jsonl
  gen $run holdout plain --id-not-contains +
  gen $run holdout decoy --id-contains +decoy
  gen $run holdout flip --id-contains +flip
  gen $run holdout clut --id-contains +clut
  gen $run test test
  case $run in *ctrl6p*) ;; *) gen $run val val ;; esac
}
training() { pgrep -fc "python train.py" || true; }

echo "=== smoke: every arm builds, steps and generates ==="
for r in $SMOKE; do
  # shellcheck disable=SC2086,SC2046
  python train.py $REC $(flags $r) --epochs 1 --limit-train 128 --limit-val 32 --batch 16 \
      --eval-every 4 --out runs/_smoke_$r || { echo "smoke $r failed"; exit 1; }
  python evaluate.py --generate --ckpt runs/_smoke_$r/best.pt --cache data_cache_d3 --split test \
      --limit 2 --gen-out runs/_smoke_$r.jsonl || { echo "smoke $r generate failed"; exit 1; }
  rm -rf runs/_smoke_$r runs/_smoke_$r*.jsonl
done

for r in $RUNS; do
  while [ "$(training)" -ge "$JOBS" ]; do sleep 30; done
  arm $r > out/$r.log 2>&1 &
  sleep 90                     # let it reach train.py before the next count
  [ "$r" = d3_D2Rp_s0 ] && nvidia-smi > out/nvidia_smi_pick.txt 2>&1
done
wait
TAR="${TAR:-pick_all.tar.gz}"
tar czf "$TAR" runs/d3_*p_s* out
ls -l "$TAR"
echo "=== run_pick done $(date)"
