# The ELECTRA reader on one GPU (.claude/plans/electra-only-reader.md step 3):
# R23's recipe (A0 + a frozen reader, results/R23.md) on the same cluttered
# corpus as C0, with every reader vector ELECTRA's (electra_cache.py: head n,
# one pass per request). Three arms, two seeds each, since single-seed plain
# differences under ~2.5 points are noise:
#
#   c0_EL  / c0_ELs1   the request as ELECTRA's 4-word chunks (data_cache_c0e),
#                      the planner's own request words kept
#   c0_ESC / c0_ESCs1  7 role slots (tagged by ternary ELECTRA), then the chunks
#                      (data_cache_c0esc), words kept: do roles help?
#   c0_ESCV/ c0_ESCVs1 slots + chunks, the request's own words dropped: can the
#                      request reach the planner as vectors only?
#
# Both caches share their rows, targets, constants, fields and reader table;
# only t_chunk differs. Greedy generation on the holdout's four halves (plain,
# decoy, flip, clut) and the test split here; backoff 3, sandbox scoring and
# the demo run on the laptop (results/logs/score_c0e.sh): backoff's compile
# check replays paused tasks through harness.run, the Node sandbox, which the
# pod doesn't have (2026-10-04: every backoff run died on the first paused task).
#
#   bash run_c0e.sh      # from tiny/
#
# Everything is archived into c0e_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
RUNS="c0_EL c0_ELs1 c0_ESC c0_ESCs1 c0_ESCV c0_ESCVs1"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"
for c in data_cache_c0e data_cache_c0esc; do
  [ -f $c/reader.pt ] || { echo "no $c/reader.pt (electra_cache.py apply)"; exit 2; }
done

cache() {  # run
  case "$1" in c0_EL*) echo data_cache_c0e ;; *) echo data_cache_c0esc ;; esac
}
flags() {  # run
  case "$1" in
    c0_EL) echo "--pack --seed 0 --reader --reader-lines" ;;
    c0_ELs1) echo "--pack --seed 1 --reader --reader-lines" ;;
    c0_ESC) echo "--pack --seed 0 --reader --reader-lines" ;;
    c0_ESCs1) echo "--pack --seed 1 --reader --reader-lines" ;;
    c0_ESCV) echo "--pack --seed 0 --reader --reader-lines --no-req-words" ;;
    c0_ESCVs1) echo "--pack --seed 1 --reader --reader-lines --no-req-words" ;;
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
  python evaluate.py --generate --ckpt runs/_smoke_$1/best.pt --cache $(cache $1) --split holdout     --limit 4 --id-not-contains + --gen-out runs/_smoke_$1/gen.jsonl || return 1
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
tar czf c0e_all.tar.gz $(for r in $RUNS; do echo runs/$r; done) out
ls -l c0e_all.tar.gz
echo "=== run_c0e done $(date)"
