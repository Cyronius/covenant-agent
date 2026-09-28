# Four planners on one GPU (.claude/plans/option-split.md): S6's options one
# at a time, each on 25% of the rows (results/logs/mix_s6opt.sh). R10's recipe
# (ar, seed 0, 12 epochs, batch 64, pad-weight 0.5, packed), as run_s6frac.sh.
#
#   dec25_A0     A0 on data_cache_dec25 (decoys only)
#   dtw25_A0     A0 on data_cache_dtw25 (decoys + twin roles)
#   opq25_A0     A0 on data_cache_opq25 (meaningless names only)
#   opq25_SPt    SPt on data_cache_opq25
#
# Greedy generation only; backoff, scoring and R18's probes run on the laptop.
#
#   bash run_s6opt.sh      # from tiny/
#
# Everything is archived into s6opt_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
STAGES="${STAGES:-runs/pod_vocab/runs/vocab/v2_6ep/stages_last.pt}"
RUNS="dec25_A0 dtw25_A0 opq25_A0 opq25_SPt"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

flags() {  # run
  case "$1" in
    *_A0) echo "--pack --seed 0" ;;
    *_SPt) echo "--pack --seed 0 --split --desc-w 128 --desc-layers 4 --stage-pool mean --stages-from $STAGES --stage-lr-mult 0.1 --lam-nce 0.5 --lam-rel 20" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
cache_of() { echo "data_cache_${1%%_*}"; }
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
for r in $RUNS; do smoke $r || exit 1; done

for r in $RUNS; do
  arm $r > out/$r.log 2>&1 &
done
sleep 300
nvidia-smi > out/nvidia_smi_5min.txt 2>&1
wait
tar czf s6opt_all.tar.gz $(for r in $RUNS; do echo runs/$r; done) out
ls -l s6opt_all.tar.gz
echo "=== run_s6opt done $(date)"
