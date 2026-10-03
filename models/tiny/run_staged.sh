# Step 2 of .claude/plans/staged-decoder-experts.md, and step 0b's retrain,
# on one GPU. Twelve runs, three at a time; R10's recipe (ar, 12 epochs,
# batch 64, pad-weight 0.5, packed) without a reader (the owner took
# Ternlight out, 2026-10-02):
#
#   clt_base_s{0,1}   data_cache_clt, plain left to right, 4 layers  (D3's "before")
#   d3_base_s{0,1}    data_cache_d3 (every old row + every new row), same model
#   d3_ctrl6_s{0,1}   left to right, 6 layers: the parameter control
#   d3_draft_s{0,1}   one diffusion draft stage, 4 layers x 4 loops, no refiner
#   d3_DR_s{0,1}      draft 4x4 -> refiner 2x1                    (the design)
#   d3_D2R_s{0,1}     draft 2x4 -> draft 2x4 -> refiner 2x1      (two draft stages)
#
# Greedy generation on the holdout's four halves (plain, decoy, flip,
# cluttered) and the test split. Staged runs also write each program's draft
# readout (<gen>.draft.jsonl) and confidence, so the early-exit threshold is
# chosen offline. Scoring, the new-kind exams and the demo requests run on the
# laptop (results/logs/score_staged.sh).
#
#   bash run_staged.sh      # from tiny/
#
# Everything is archived into staged_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
RUNS="${RUNS:-d3_DR_s0 d3_D2R_s0 d3_base_s0 d3_ctrl6_s0 d3_draft_s0 clt_base_s0 \
      d3_DR_s1 d3_D2R_s1 d3_base_s1 d3_ctrl6_s1 d3_draft_s1 clt_base_s1}"
SMOKE="${SMOKE-d3_base_s0 d3_ctrl6_s0 d3_draft_s0 d3_DR_s0 d3_D2R_s0 clt_base_s0}"
JOBS="${JOBS:-3}"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

cache() { case $1 in clt_*) echo data_cache_clt ;; *) echo data_cache_d3 ;; esac; }
flags() {  # run
  local s="--pack --seed ${1##*_s}"
  case "$1" in
    *_base_*)  echo "$s" ;;
    *_ctrl6_*) echo "$s --dec-layers 6" ;;
    *_draft_*) echo "$s --stages draft:4x4" ;;
    *_DR_*)    echo "$s --stages draft:4x4,refine:2x1" ;;
    # R27's fix arm: the refiner sees same-kind swaps at about the draft's
    # held-out error rate, so copying the draft stops paying in training
    *_DRx_*)   echo "$s --stages draft:4x4,refine:2x1 --draft-noise 0.05 --draft-swap 0.25" ;;
    *_D2R_*)   echo "$s --stages draft:2x4,draft:2x4,refine:2x1" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
gen() {  # run split tag selector...
  local run=$1 split=$2 tag=$3; shift 3
  [ -f out/${run}_${tag}.jsonl ] && return 0      # evaluate.py writes it whole, at the end
  python evaluate.py --generate --ckpt runs/$run/best.pt --cache $(cache $run) --split $split \
      --limit 2000 "$@" --gen-out out/${run}_${tag}.jsonl
}
arm() {  # run
  local run=$1
  if [ ! -f "runs/$run/best.pt" ]; then
    # shellcheck disable=SC2046
    python train.py --arm ar --cache $(cache $run) --epochs 12 --batch 64 --pad-weight 0.5 \
        --out runs/$run $(flags "$run") || { echo "train $run failed"; return 1; }
  fi
  cp runs/$run/config.json out/${run}_config.json
  cp runs/$run/log.jsonl out/${run}_curve.jsonl
  gen $run holdout plain --id-not-contains +
  gen $run holdout decoy --id-contains +decoy
  gen $run holdout flip --id-contains +flip
  gen $run holdout clut --id-contains +clut
  gen $run test test
  # the early-exit threshold is chosen on val (calibrate.py), not on an exam
  case $run in *_DR_*|*_D2R_*|*_DRx_*) gen $run val val ;; esac
}
smoke() {  # run
  # shellcheck disable=SC2046
  python train.py --arm ar --cache $(cache $1) --epochs 1 --limit-train 128 \
    --limit-val 32 --batch 16 --eval-every 4 --out runs/_smoke_$1 $(flags $1) || return 1
  python evaluate.py --generate --ckpt runs/_smoke_$1/best.pt --cache $(cache $1) --split test \
    --limit 2 --gen-out runs/_smoke_$1.jsonl || return 1
  rm -rf runs/_smoke_$1 runs/_smoke_$1*.jsonl
}
echo "=== smoke: every arm builds, steps and generates ==="
for r in $SMOKE; do
  smoke $r || { echo "smoke $r failed"; exit 1; }
done

for r in $RUNS; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 30; done
  arm $r > out/$r.log 2>&1 &
  [ "$r" = d3_DR_s0 ] && { sleep 240; nvidia-smi > out/nvidia_smi_first.txt 2>&1; }
done
sleep 300
nvidia-smi > out/nvidia_smi_full.txt 2>&1
wait
TAR="${TAR:-staged_all.tar.gz}"
tar czf "$TAR" runs/d3_* runs/clt_* out
ls -l "$TAR"
echo "=== run_staged done $(date)"
