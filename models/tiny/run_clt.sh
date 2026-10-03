# R23 on one GPU: fdc25_RD's recipe (A0 + frozen Ternlight-mini reader) on
# the cluttered corpus (results/logs/gen_clt.sh, data.gen --clutter), two
# seeds, since single-seed plain differences under ~2.5 points are noise.
# R10's recipe (ar, 12 epochs, batch 64, pad-weight 0.5, packed), as
# run_rd.sh.
#
#   clt_RD     reader, cluttered 25% flip decoys + 75% options-off, seed 0
#   clt_RDs1   the same, seed 1
#
# Greedy generation on the holdout's four halves (plain, decoy, flip, and the
# cluttered plain exam, ids +clut) and the test split; backoff, scoring and
# the demo requests run on the laptop.
#
#   bash run_clt.sh      # from tiny/
#
# Everything is archived into clt_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
RUNS="clt_RD clt_RDs1"
CACHE=data_cache_clt
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

flags() {  # run
  case "$1" in
    *_RD) echo "--pack --seed 0 --reader" ;;
    *_RDs1) echo "--pack --seed 1 --reader" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
gen() {  # run split tag selector...
  local run=$1 split=$2 tag=$3; shift 3
  local lim=2000; [ "$split" = test ] && lim=1000
  python evaluate.py --generate --ckpt runs/$run/best.pt --cache $CACHE --split $split \
      --limit $lim "$@" --gen-out out/${run}_${tag}.jsonl
}
arm() {  # run
  local run=$1
  if [ ! -f "runs/$run/best.pt" ]; then
    # shellcheck disable=SC2046
    python train.py --arm ar --cache $CACHE --epochs 12 --batch 64 --pad-weight 0.5 \
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
  python train.py --arm ar --cache $CACHE --epochs 1 --limit-train 128 \
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
tar czf clt_all.tar.gz $(for r in $RUNS; do echo runs/$r; done) out
ls -l clt_all.tar.gz
echo "=== run_clt done $(date)"
