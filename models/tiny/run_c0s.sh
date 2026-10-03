# The slot reader on one GPU (.claude/plans/electra-slot-reader.md step 4):
# R23's recipe (A0 + frozen Ternlight-mini reader, results/R23.md) on the same
# cluttered corpus, with C0's reader on every constant and field. What differs
# is how the request reaches the planner. Three arms, two seeds each, since
# single-seed plain differences under ~2.5 points are noise:
#
#   c0_RL / c0_RLs1   C0: the request in 4-word chunks (data_cache_c0), the
#                     planner's own request words kept
#   c0_SC  / c0_SCs1  the request as 7 role slots (tagged by ternary ELECTRA,
#                     read by Ternlight) followed by C0's 4-word chunks
#                     (data_cache_c0sc, slot_cache.py --keep-chunks), words kept.
#                     Slots alone match constants 3-6 probe points worse than
#                     chunks; slots + chunks beat chunks on plain and cluttered
#                     (results/logs/slots/tagcheck_t_tern_e1.json), so this asks
#                     whether adding roles helps the planner
#   c0_SCV / c0_SCVs1 slots + chunks, the request's own words dropped: the
#                     request reaches the planner only as reader vectors
#
# Both caches share their rows, targets, constants and fields; only t_chunk and
# the reader table's tail differ. Greedy generation on the holdout's four
# halves (plain, decoy, flip, clut) and the test split; backoff, scoring and
# the demo run on the laptop (results/logs/score_c0s.sh).
#
#   bash run_c0s.sh      # from tiny/
#
# Everything is archived into c0s_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
RUNS="c0_RL c0_RLs1 c0_SC c0_SCs1 c0_SCV c0_SCVs1"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"
for c in data_cache_c0 data_cache_c0sc; do
  [ -f $c/reader.pt ] || { echo "no $c/reader.pt (reader_table.py)"; exit 2; }
done

cache() {  # run
  case "$1" in c0_RL*) echo data_cache_c0 ;; *) echo data_cache_c0sc ;; esac
}
flags() {  # run
  case "$1" in
    c0_RL) echo "--pack --seed 0 --reader --reader-lines" ;;
    c0_RLs1) echo "--pack --seed 1 --reader --reader-lines" ;;
    c0_SC) echo "--pack --seed 0 --reader --reader-lines" ;;
    c0_SCs1) echo "--pack --seed 1 --reader --reader-lines" ;;
    c0_SCV) echo "--pack --seed 0 --reader --reader-lines --no-req-words" ;;
    c0_SCVs1) echo "--pack --seed 1 --reader --reader-lines --no-req-words" ;;
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
tar czf c0s_all.tar.gz $(for r in $RUNS; do echo runs/$r; done) out
ls -l c0s_all.tar.gz
echo "=== run_c0s done $(date)"
