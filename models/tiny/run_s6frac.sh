# Three planners on one GPU (.claude/plans/decoy-fraction.md): S6's three
# options put back on 25% and 50% of the rows (results/logs/mix_s6d.sh).
# R10's recipe (ar, seed 0, 12 epochs, batch 64, pad-weight 0.5, packed), as
# run_s6off2.sh.
#
#   s6d25_SPt    SPt on data_cache_s6d25 (25% of rows are S6's)
#   s6d50_SPt    SPt on data_cache_s6d50 (50%)
#   s6d25_A0     A0 on data_cache_s6d25: does the plain planner keep its
#                plain score with a quarter of S6's rows
#
# Greedy generation only; backoff and scoring run on the laptop.
#
#   bash run_s6frac.sh      # from tiny/
#
# Everything is archived into s6frac_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
STAGES="${STAGES:-runs/pod_vocab/runs/vocab/v2_6ep/stages_last.pt}"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

flags() {  # run
  case "$1" in
    s6d25_A0) echo "--pack --seed 0" ;;
    s6d25_SPt|s6d50_SPt) echo "--pack --seed 0 --split --desc-w 128 --desc-layers 4 --stage-pool mean --stages-from $STAGES --stage-lr-mult 0.1 --lam-nce 0.5 --lam-rel 20" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
cache_of() {
  case "$1" in
    s6d25_*) echo data_cache_s6d25 ;;
    s6d50_*) echo data_cache_s6d50 ;;
  esac
}
gen() {  # run cache split tag selector...
  local run=$1 cache=$2 split=$3 tag=$4; shift 4
  local lim=2000; [ "$split" = test ] && lim=1000
  python evaluate.py --generate --ckpt runs/$run/best.pt --cache $cache --split $split \
      --limit $lim "$@" --gen-out out/${run}_${tag}.jsonl
}
arm() {  # run
  local run=$1 cache
  cache=$(cache_of "$run")
  if [ ! -f "runs/$run/best.pt" ]; then
    # shellcheck disable=SC2046
    python train.py --arm ar --cache $cache --epochs 12 --batch 64 --pad-weight 0.5 \
        --out runs/$run $(flags "$run") || { echo "train $run failed"; return 1; }
  fi
  cp runs/$run/config.json out/${run}_config.json
  cp runs/$run/log.jsonl out/${run}_curve.jsonl
  gen $run $cache holdout plain --id-not-contains +
  gen $run $cache holdout decoy --id-contains +decoy
  gen $run $cache holdout flip --id-contains +flip
  gen $run $cache test test
}
smoke() {  # run
  # shellcheck disable=SC2046
  python train.py --arm ar --cache $(cache_of $1) --epochs 1 --limit-train 128 \
    --limit-val 32 --batch 16 --eval-every 4 --out runs/_smoke_$1 $(flags $1) || return 1
  rm -rf runs/_smoke_$1
}
echo "=== smoke: every run builds and steps ==="
for r in s6d25_SPt s6d50_SPt s6d25_A0; do smoke $r || exit 1; done

for r in s6d25_SPt s6d50_SPt s6d25_A0; do
  arm $r > out/$r.log 2>&1 &
done
sleep 300
nvidia-smi > out/nvidia_smi_5min.txt 2>&1
wait
tar czf s6frac_all.tar.gz runs/s6d25_SPt runs/s6d50_SPt runs/s6d25_A0 out
ls -l s6frac_all.tar.gz
echo "=== run_s6frac done $(date)"
