# Step 2c of .claude/plans/staged-decoder-experts.md: the draft commits its
# surest slots after every loop, so the slots it is unsure of -- the
# constants that must agree with each other -- are decided last, knowing the
# rest (R28 point 11). One GPU, three runs at a time:
#
#   d3_DRc_s{0,1}    draft 4x4 committing (4 rounds) -> refiner 2x1    vs d3_DR
#   d3_D2Rc_s{0,1}   draft 2x4 -> 2x4 committing (8 rounds) -> 2x1     vs d3_D2R
#
# Recipe, generations and archive as run_staged.sh, so each compares with its
# twin row for row. Scoring runs on the laptop (results/logs/score_staged.sh).
#
#   bash run_commit.sh      # from tiny/
set -uo pipefail
cd "$(dirname "$0")"
RUNS="${RUNS:-d3_DRc_s0 d3_D2Rc_s0 d3_DRc_s1 d3_D2Rc_s1}"
SMOKE="${SMOKE-d3_DRc_s0 d3_D2Rc_s0}"
JOBS="${JOBS:-3}"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

REC="--arm ar --cache data_cache_d3 --epochs 12 --batch 64 --pad-weight 0.5 --pack --commit"
flags() {  # run
  local s="--seed ${1##*_s}"
  case "$1" in
    *_DRc_*)  echo "$s --stages draft:4x4,refine:2x1" ;;
    *_D2Rc_*) echo "$s --stages draft:2x4,draft:2x4,refine:2x1" ;;
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
  if [ ! -f runs/$run/best.pt ]; then
    # shellcheck disable=SC2086,SC2046
    python train.py $REC $(flags $run) --out runs/$run || { echo "train $run failed"; return 1; }
  fi
  cp runs/$run/config.json out/${run}_config.json
  cp runs/$run/log.jsonl out/${run}_curve.jsonl
  gen $run holdout plain --id-not-contains +
  gen $run holdout decoy --id-contains +decoy
  gen $run holdout flip --id-contains +flip
  gen $run holdout clut --id-contains +clut
  gen $run test test
  gen $run val val
}

echo "=== smoke: every arm builds, steps and generates ==="
for r in $SMOKE; do
  # shellcheck disable=SC2086,SC2046
  python train.py $REC $(flags $r) --epochs 1 --limit-train 128 --limit-val 32 --batch 16 \
      --eval-every 4 --out runs/_smoke_$r || { echo "smoke $r failed"; exit 1; }
  python evaluate.py --generate --ckpt runs/_smoke_$r/best.pt --cache data_cache_d3 --split test \
      --limit 2 --gen-out runs/_smoke_$r.jsonl || { echo "smoke $r generate failed"; exit 1; }
  grep -q commit_round runs/_smoke_$r.draft.jsonl || { echo "smoke $r: no commit_round"; exit 1; }
  rm -rf runs/_smoke_$r runs/_smoke_$r*.jsonl
done

for r in $RUNS; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 30; done
  arm $r > out/$r.log 2>&1 &
  [ "$r" = d3_D2Rc_s0 ] && { sleep 300; nvidia-smi > out/nvidia_smi_commit.txt 2>&1; }
done
wait
TAR="${TAR:-commit_all.tar.gz}"
tar czf "$TAR" runs/d3_DRc_* runs/d3_D2Rc_* out
ls -l "$TAR"
echo "=== run_commit done $(date)"
