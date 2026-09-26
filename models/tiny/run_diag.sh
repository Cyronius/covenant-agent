# R13's two diagnostics, side by side on one GPU. Both are A0 (one line
# encoder), R10's recipe: ar, seed 0, 12 epochs, batch 64, pad-weight 0.5,
# packed (bit-equal to padded, R11 §2).
#
#   s5fix     R10's corpus (s5_plain) and exam, the current code (schema-graph
#             fix). Its exam also carries S6's plain half (ids +s6), so one
#             model trained on S5 is scored on both exams.
#   s6nonames S6 as in R13's A0, the same tokenizer, tool names NOT on the line
#
#   bash run_diag.sh      # from tiny/
#
# Everything is archived into diag_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

gen() {  # run cache split tag selector...
  local run=$1 cache=$2 split=$3 tag=$4; shift 4
  local lim=2000; [ "$split" = test ] && lim=1000
  python evaluate.py --generate --ckpt runs/$run/best.pt --cache $cache --split $split \
      --limit $lim "$@" --gen-out out/${run}_${tag}.jsonl
}
arm() {  # run cache
  local run=$1 cache=$2
  python train.py --arm ar --seed 0 --cache $cache --epochs 12 --batch 64 --pad-weight 0.5 \
      --pack --out runs/$run
  cp runs/$run/config.json out/${run}_config.json
  cp runs/$run/log.jsonl out/${run}_curve.jsonl
  gen $run $cache holdout plain --id-not-contains +
  gen $run $cache holdout decoy --id-contains +decoy
  if [ "$run" = diag_s5fix ]; then
    gen $run $cache holdout s6plain --id-contains +s6
  else
    gen $run $cache holdout flip --id-contains +flip
  fi
  gen $run $cache test test
}
arm diag_s5fix data_cache_ho42_fix > out/diag_s5fix.log 2>&1 &
arm diag_s6nonames data_cache_s6g_nonames > out/diag_s6nonames.log 2>&1 &
wait
tar czf diag_all.tar.gz runs/diag_* out
ls -l diag_all.tar.gz
echo "=== run_diag done $(date)"
