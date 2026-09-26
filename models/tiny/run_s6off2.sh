# Two more planners on the S6 options-off rows (results/R15.md "Next" 2),
# side by side on one GPU. R10's recipe (ar, 12 epochs, batch 64,
# pad-weight 0.5, packed), as run_s6split.sh.
#
#   s6off_SPt    SPt (R12's V2 stages, fine-tuned) on data_cache_s6off: R14's
#                best plain score with the stages' decoy reading. Against
#                s6off_A0 only the model differs.
#   s6off_A0s1   s6off_A0 again with seed 1: a seed spread for the best model,
#                and whether R15's seven failing worlds follow the seed.
#
# Greedy generation only. Tool backoff (results/logs/tool_backoff_r15.py) and
# best-of need the sandbox's typechecker, so they run on the laptop.
#
#   bash run_s6off2.sh      # from tiny/
#
# Everything is archived into s6off2_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
STAGES="${STAGES:-runs/pod_vocab/runs/vocab/v2_6ep/stages_last.pt}"
CACHE=data_cache_s6off
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

flags() {  # run
  case "$1" in
    s6off_A0s1) echo "--pack --seed 1" ;;
    s6off_SPt) echo "--pack --seed 0 --split --desc-w 128 --desc-layers 4 --stage-pool mean --stages-from $STAGES --stage-lr-mult 0.1 --lam-nce 0.5 --lam-rel 20" ;;
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
  gen $run test test
}
smoke() {  # run
  # shellcheck disable=SC2046
  python train.py --arm ar --cache $CACHE --epochs 1 --limit-train 128 \
    --limit-val 32 --batch 16 --eval-every 4 --out runs/_smoke_$1 $(flags $1) || return 1
  rm -rf runs/_smoke_$1
}
echo "=== smoke: both runs build and step ==="
smoke s6off_SPt || exit 1
smoke s6off_A0s1 || exit 1

arm s6off_SPt > out/s6off_SPt.log 2>&1 &
arm s6off_A0s1 > out/s6off_A0s1.log 2>&1 &
sleep 240
nvidia-smi > out/nvidia_smi_4min.txt 2>&1
wait
tar czf s6off2_all.tar.gz runs/s6off_SPt runs/s6off_A0s1 out
ls -l s6off2_all.tar.gz
echo "=== run_s6off2 done $(date)"
