# Score run_clt.sh's generations on the laptop (R23, .claude/plans/demo-clutter-retrain.md).
#
#   bash results/logs/score_clt.sh   # after unpacking clt_all.tar.gz into models/tiny/runs/pod_clt/
#
# Sandbox scores for every half (plain, decoy, flip, clut) and test;
# evaluate.py --backoff 3 on plain and clut; and play.py (the demo's own loop,
# live reader, backoff 3) on the 70 demo requests and on the 500-task
# plain/cluttered exam samples fdc25_RD was measured on.
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_clt/out
LOGS=$ROOT/results/logs/clt
RUNS="clt_RD clt_RDs1"
CACHE=data_cache_clt
READER=$ROOT/models/tiny/reader
export HF_HUB_OFFLINE=1
mkdir -p "$LOGS"

score() {  # name split
  (cd "$TINY" && python -W ignore evaluate.py --score --cache $CACHE --split $2 \
      --gen-out "$OUT/$1.jsonl" > "$OUT/$1.score.log" 2>&1)
}
for run in $RUNS; do
  for half in plain decoy flip clut; do score ${run}_$half holdout & done
  score ${run}_test test &
done
wait
for run in $RUNS; do
  for half in plain clut; do
    case $half in plain) sel="--id-not-contains +";; clut) sel="--id-contains +clut";; esac
    (cd "$TINY" && python -W ignore evaluate.py --generate --backoff 3 \
        --ckpt "runs/pod_clt/runs/$run/best.pt" --cache $CACHE --split holdout --limit 2000 \
        --device cpu $sel --gen-out "$OUT/${run}_${half}_bk3.jsonl" \
        > "$OUT/${run}_${half}_bk3.gen.log" 2>&1; score ${run}_${half}_bk3 holdout) &
  done
done
wait
for run in $RUNS; do
  (cd "$TINY" && python -W ignore play.py --ckpt runs/pod_clt/runs/$run/best.pt --cache $CACHE \
      --tasks ../../data/holdout/e_demo_requests.jsonl --retype --reader-live $READER --backoff 3 \
      --show 0 --out "$LOGS/demo_$run.jsonl" > "$LOGS/demo_$run.log" 2>&1) &
  for v in s6 clt; do
    (cd "$TINY" && python -W ignore play.py --ckpt runs/pod_clt/runs/$run/best.pt --cache $CACHE \
        --tasks ../../data/${v}_holdout_s500.jsonl --reader-live $READER --backoff 3 \
        --show 0 --out "$LOGS/s500_${v}_$run.jsonl" > "$LOGS/s500_${v}_$run.log" 2>&1) &
  done
done
wait
cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$OUT"/*.log "$LOGS"/ 2>/dev/null
grep -H -E "^  goal " "$OUT"/*.score.log
grep -H "goal" "$LOGS"/demo_*.log "$LOGS"/s500_*.log
echo "=== score_clt done $(date)"
