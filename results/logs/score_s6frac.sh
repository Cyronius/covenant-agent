# Score run_s6frac.sh's generations on the laptop (.claude/plans/decoy-fraction.md).
#
#   bash results/logs/score_s6frac.sh   # after unpacking s6frac_all.tar.gz into
#                                       # models/tiny/runs/pod_s6frac/
#
# Every exam half is S6 themes, so everything scores in today's tree. Tool
# backoff (evaluate.py --backoff 3) needs the typechecker, so it runs here:
# on the plain exam for all three runs, and on the decoyed exam for SPt.
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_s6frac/out
CKPT=$TINY/runs/pod_s6frac/runs
LOGS=$ROOT/results/logs/s6frac
mkdir -p "$LOGS"

cache_of() {
  case "$1" in
    s6d25_*) echo data_cache_s6d25 ;;
    s6d50_*) echo data_cache_s6d50 ;;
  esac
}
score() {  # run name split
  (cd "$TINY" && python -W ignore evaluate.py --score --cache $(cache_of $1) --split $3 \
      --gen-out "$OUT/$2.jsonl" > "$OUT/$2.score.log" 2>&1)
}
backoff() {  # run tag selector...
  local run=$1 tag=$2; shift 2
  (cd "$TINY" && HF_HUB_OFFLINE=1 python -W ignore evaluate.py --generate --backoff 3 \
      --ckpt "$CKPT/$run/best.pt" --cache $(cache_of $run) --split holdout --limit 2000 \
      --device cpu "$@" --gen-out "$OUT/${run}_${tag}_bk3.jsonl" > "$OUT/${run}_${tag}_bk3.gen.log" 2>&1)
  score $run ${run}_${tag}_bk3 holdout
}

RUNS="s6d25_SPt s6d50_SPt s6d25_A0"
for run in $RUNS; do
  for half in plain decoy flip; do score $run ${run}_$half holdout & done
  score $run ${run}_test test &
done
wait
for run in $RUNS; do backoff $run plain --id-not-contains + & done
backoff s6d25_SPt decoy --id-contains +decoy &
backoff s6d50_SPt decoy --id-contains +decoy &
wait
for run in $RUNS; do
  (cd "$TINY" && python -W ignore fail_kinds.py --cache $(cache_of $run) "$OUT/${run}_plain" \
      --json "$LOGS/fail_kinds_${run}_plain.json" > "$LOGS/fail_kinds_${run}_plain.log" 2>&1)
done
cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$LOGS"/ 2>/dev/null
for run in $RUNS; do cp "$OUT/$run.log" "$LOGS"/; done
grep -H -E "^  goal " "$OUT"/*.score.log
echo "=== score_s6frac done $(date)"
