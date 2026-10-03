# Borrowed worlds (.claude/plans/borrowed-worlds.md step 6) on one GPU:
# clt_RD's recipe (A0 + frozen Ternlight-mini reader, R10's ar / 12 epochs /
# batch 64 / pad-weight 0.5 / packed) on three mixes of clt_train with the new
# kinds (observe-then-act briefs, workshop, rooms, tau2 service) at ~10, 20
# and 30% (results/logs/mix_brw.py, gen_brw.sh). One seed each; the sweep is
# over the mix, and R23's two seeds put single-seed noise at ~2.5 points.
#
#   brw10_RD   data_cache_brw10, seed 0
#   brw20_RD   data_cache_brw20, seed 0
#   brw30_RD   data_cache_brw30, seed 0
#
# Greedy generation on the holdout's four halves (plain, decoy, flip, and the
# cluttered plain exam, ids +clut) and the test split; backoff, scoring, the
# new-kind exams (play.py / rpg_suite need the sandbox) and the demo requests
# run on the laptop.
#
#   bash run_brw.sh      # from tiny/
#
# Everything is archived into brw_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
RUNS="brw10_RD brw20_RD brw30_RD"
cache() { echo "data_cache_${1%%_*}"; }
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

flags() {  # run
  case "$1" in
    *_RD) echo "--pack --seed 0 --reader" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
gen() {  # run split tag selector...
  local run=$1 split=$2 tag=$3; shift 3
  local lim=2000; [ "$split" = test ] && lim=1000
  python evaluate.py --generate --ckpt runs/$run/best.pt --cache $(cache $run) --split $split \
      --limit $lim "$@" --gen-out out/${run}_${tag}.jsonl
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
}
smoke() {  # run
  # shellcheck disable=SC2046
  python train.py --arm ar --cache $(cache $1) --epochs 1 --limit-train 128 \
    --limit-val 32 --batch 16 --eval-every 4 --out runs/_smoke_$1 $(flags $1) || return 1
  rm -rf runs/_smoke_$1
}
echo "=== smoke: every run builds and steps ==="
for r in $RUNS; do smoke $r || exit 1; done

for r in $RUNS; do
  arm $r > out/$r.log 2>&1 &
done
sleep 300
nvidia-smi > out/nvidia_smi_5min.txt 2>&1
wait
tar czf brw_all.tar.gz $(for r in $RUNS; do echo runs/$r; done) out
ls -l brw_all.tar.gz
echo "=== run_brw done $(date)"
