# Three planners on one GPU (.claude/plans/s5-first-s6-cost-split.md): what
# R13's 23 points come from, and the reading stages on S5's rows. R10's
# recipe throughout (ar, seed 0, 12 epochs, batch 64, pad-weight 0.5, packed).
#
#   s5g_A0     A0 on S5's rows, S6's cache settings (names, 120-character
#              descriptions, R12's tokenizer, 112-token lines). Against
#              diag_s5fix only the cache settings differ; against s6_A0 only
#              the rows.
#   s5g_SPt    SPt (R12's V2 stages, fine-tuned) on the same cache.
#   s6off_A0   A0 on S6's themes with decoys, twin roles and opaque names off
#              (results/logs/gen_s6off.sh). Against s5g_A0 only the themes
#              differ; against s6_A0 only the three options.
#
# data_cache_s5g's exam is data/s5s6_holdout.jsonl: S5 plain (no suffix),
# S5 decoyed (+decoy), S6 plain (+s6, the same 4,000 tasks as
# data/s6_holdout.jsonl). data_cache_s6off's is s6_holdout_both, as s6_A0's.
#
#   bash run_s6split.sh      # from tiny/
#
# Everything is archived into s6split_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
STAGES="${STAGES:-runs/pod_vocab/runs/vocab/v2_6ep/stages_last.pt}"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

flags() {  # run
  case "$1" in
    s5g_A0|s6off_A0) echo "--pack" ;;
    s5g_SPt) echo "--pack --split --desc-w 128 --desc-layers 4 --stage-pool mean --stages-from $STAGES --stage-lr-mult 0.1 --lam-nce 0.5 --lam-rel 20" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
cache_of() {
  case "$1" in
    s5g_*) echo data_cache_s5g ;;
    s6off_*) echo data_cache_s6off ;;
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
    python train.py --arm ar --seed 0 --cache $cache --epochs 12 --batch 64 --pad-weight 0.5 \
        --out runs/$run $(flags "$run") || { echo "train $run failed"; return 1; }
  fi
  cp runs/$run/config.json out/${run}_config.json
  cp runs/$run/log.jsonl out/${run}_curve.jsonl
  gen $run $cache holdout plain --id-not-contains +
  gen $run $cache holdout decoy --id-contains +decoy
  if [ "$cache" = data_cache_s5g ]; then
    gen $run $cache holdout s6plain --id-contains +s6
    if [ "$run" = s5g_SPt ]; then
      gen $run $cache holdout s6plain_bo4 --id-contains +s6 --best-of 4
    fi
  else
    gen $run $cache holdout flip --id-contains +flip
  fi
  gen $run $cache test test
}

smoke() {  # run
  # shellcheck disable=SC2046
  python train.py --arm ar --cache $(cache_of $1) --epochs 1 --limit-train 128 \
    --limit-val 32 --batch 16 --eval-every 4 --out runs/_smoke_$1 $(flags $1) || return 1
  rm -rf runs/_smoke_$1
}
echo "=== smoke: both S5 runs build and step ==="
smoke s5g_A0 || exit 1
smoke s5g_SPt || exit 1

# Two at a time, as R13 ran A0 and SPt on the same card; the third starts
# when either finishes. The per-process VRAM for these caches was never
# recorded, so take it once both are running.
arm s5g_SPt > out/s5g_SPt.log 2>&1 &
arm s5g_A0 > out/s5g_A0.log 2>&1 &
sleep 240
nvidia-smi > out/nvidia_smi_4min.txt 2>&1
wait -n
# data_cache_s6off may still be on its way up: it is unpacked elsewhere and
# moved into place, then data_cache_s6off.ready is touched.
while [ ! -f data_cache_s6off.ready ]; do sleep 30; done
if smoke s6off_A0 > out/smoke_s6off_A0.log 2>&1; then
  arm s6off_A0 > out/s6off_A0.log 2>&1 &
else
  echo "smoke s6off_A0 failed: out/smoke_s6off_A0.log"
fi
wait
tar czf s6split_all.tar.gz runs/s5g_* runs/s6off_* out
ls -l s6split_all.tar.gz
echo "=== run_s6split done $(date)"
