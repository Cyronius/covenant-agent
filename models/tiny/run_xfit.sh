# Step 2b of .claude/plans/staged-decoder-experts.md: cross-fitted drafts, on
# one GPU. R28 found the staged refiner copies its draft, because on the rows
# it trains on the draft stage has seen the answer. Here the refiner trains on
# drafts from a model that never saw the row:
#
#   d3_half{0,1}_s{0,1}  draft only (draft:4x4) on one fold of D3's training
#                        rows (staged.fold_of, by episode); four at once
#   xdraft               each half drafts the other fold's rows
#                        -> data_cache_d3/train_xdraft_s{0,1}.pt
#   d3_DRxf_s{0,1}       DR (draft:4x4,refine:2x1) with --xdraft: the draft
#                        stage trains on every row, the refiner reads the
#                        stored drafts
#
# Recipe, generations and archive as run_staged.sh, so d3_DRxf compares with
# d3_DR row for row. Scoring runs on the laptop (results/logs/score_staged.sh).
#
#   bash run_xfit.sh      # from tiny/
set -uo pipefail
cd "$(dirname "$0")"
SEEDS="${SEEDS:-0 1}"
SMOKE="${SMOKE-1}"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

REC="--arm ar --cache data_cache_d3 --epochs 12 --batch 64 --pad-weight 0.5 --pack"
gen() {  # run split tag selector...
  local run=$1 split=$2 tag=$3; shift 3
  [ -f out/${run}_${tag}.jsonl ] && return 0
  python evaluate.py --generate --ckpt runs/$run/best.pt --cache data_cache_d3 --split $split \
      --limit 2000 "$@" --gen-out out/${run}_${tag}.jsonl
}
half() {  # seed fold
  local run=d3_half$2_s$1
  if [ ! -f runs/$run/best.pt ]; then
    # shellcheck disable=SC2086
    python train.py $REC --seed $1 --stages draft:4x4 --fold $2/2 --out runs/$run \
      || { echo "train $run failed"; return 1; }
  fi
  cp runs/$run/config.json out/${run}_config.json
  cp runs/$run/log.jsonl out/${run}_curve.jsonl
}
xd() {  # seed
  local f=data_cache_d3/train_xdraft_s$1.pt
  [ -f $f ] || python xdraft.py --cache data_cache_d3 --ckpts runs/d3_half0_s$1/best.pt \
      runs/d3_half1_s$1/best.pt --out $f || return 1
  cp data_cache_d3/train_xdraft_s$1.json out/
}
arm() {  # seed
  local run=d3_DRxf_s$1
  if [ ! -f runs/$run/best.pt ]; then
    # shellcheck disable=SC2086
    python train.py $REC --seed $1 --stages draft:4x4,refine:2x1 \
        --xdraft data_cache_d3/train_xdraft_s$1.pt --out runs/$run \
      || { echo "train $run failed"; return 1; }
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

if [ -n "$SMOKE" ]; then
  echo "=== smoke: halves, stored drafts, refiner on them, generation ==="
  S="--epochs 1 --limit-train 128 --limit-val 32 --batch 16 --eval-every 4"
  for k in 0 1; do
    # shellcheck disable=SC2086
    python train.py $REC $S --stages draft:4x4 --fold $k/2 --out runs/_smoke_half$k \
      || { echo "smoke half $k failed"; exit 1; }
  done
  python xdraft.py --cache data_cache_d3 --ckpts runs/_smoke_half0/best.pt runs/_smoke_half1/best.pt \
      --out runs/_smoke_x.pt || { echo "smoke xdraft failed"; exit 1; }
  # shellcheck disable=SC2086
  python train.py $REC $S --stages draft:4x4,refine:2x1 --xdraft runs/_smoke_x.pt \
      --out runs/_smoke_drxf || { echo "smoke drxf failed"; exit 1; }
  python evaluate.py --generate --ckpt runs/_smoke_drxf/best.pt --cache data_cache_d3 --split test \
      --limit 2 --gen-out runs/_smoke_drxf.jsonl || { echo "smoke generate failed"; exit 1; }
  rm -rf runs/_smoke_*
fi

for s in $SEEDS; do
  for k in 0 1; do half $s $k > out/d3_half${k}_s$s.log 2>&1 & done
done
sleep 300
nvidia-smi > out/nvidia_smi_xfit_halves.txt 2>&1
wait
for s in $SEEDS; do xd $s > out/xdraft_s$s.log 2>&1 || echo "xdraft s$s failed"; done
for s in $SEEDS; do arm $s > out/d3_DRxf_s$s.log 2>&1 & done
wait
TAR="${TAR:-xfit_all.tar.gz}"
tar czf "$TAR" runs/d3_half* runs/d3_DRxf* out data_cache_d3/train_xdraft_s*
ls -l "$TAR"
echo "=== run_xfit done $(date)"
