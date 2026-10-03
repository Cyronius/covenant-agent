# Score run_c0s.sh's generations on the laptop (the slot reader,
# .claude/plans/electra-slot-reader.md step 4), exactly as score_c0.sh scores
# C0's, so c0_S* sits beside c0_RL* and R23's clt_RD.
#
#   bash results/logs/score_c0s.sh   # after unpacking c0s_all.tar.gz into models/tiny/runs/pod_c0s/
#
# Sandbox scores for every half (plain, decoy, flip, clut) and test;
# evaluate.py --backoff 3 on plain and clut; and play.py (the demo's own loop,
# live reader, backoff 3) on the 70 demo requests and the 500-task
# plain/cluttered exam samples. The slot runs' live path tags each request
# with the reader named in data_cache_c0sc/config.json (live_reader.request_pieces).
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_c0s/out
LOGS=$ROOT/results/logs/c0s
RUNS="c0_RL c0_RLs1 c0_SC c0_SCs1 c0_SCV c0_SCVs1"
READER=$ROOT/models/tiny/reader
cache_for() { case $1 in c0_RL*) echo data_cache_c0 ;; *) echo data_cache_c0sc ;; esac; }
export HF_HUB_OFFLINE=1
mkdir -p "$LOGS"

score() {  # name split run
  (cd "$TINY" && python -W ignore evaluate.py --score --cache $(cache_for $3) --split $2 \
      --gen-out "$OUT/$1.jsonl" > "$OUT/$1.score.log" 2>&1)
}
for run in $RUNS; do
  for half in plain decoy flip clut; do score ${run}_$half holdout $run & done
  score ${run}_test test $run &
done
wait
for run in $RUNS; do
  for half in plain clut; do
    case $half in plain) sel="--id-not-contains +";; clut) sel="--id-contains +clut";; esac
    (cd "$TINY" && python -W ignore evaluate.py --generate --backoff 3 \
        --ckpt "runs/pod_c0s/runs/$run/best.pt" --cache $(cache_for $run) --split holdout --limit 2000 \
        --device cpu $sel --gen-out "$OUT/${run}_${half}_bk3.jsonl" \
        > "$OUT/${run}_${half}_bk3.gen.log" 2>&1; score ${run}_${half}_bk3 holdout $run) &
  done
  wait
done
for run in $RUNS; do
  (cd "$TINY" && python -W ignore play.py --ckpt runs/pod_c0s/runs/$run/best.pt --cache $(cache_for $run) \
      --tasks ../../data/holdout/e_demo_requests.jsonl --retype --reader-live $READER --backoff 3 \
      --show 0 --out "$LOGS/demo_$run.jsonl" > "$LOGS/demo_$run.log" 2>&1) &
  for v in s6 clt; do
    (cd "$TINY" && python -W ignore play.py --ckpt runs/pod_c0s/runs/$run/best.pt --cache $(cache_for $run) \
        --tasks ../../data/${v}_holdout_s500.jsonl --reader-live $READER --backoff 3 \
        --show 0 --out "$LOGS/s500_${v}_$run.jsonl" > "$LOGS/s500_${v}_$run.log" 2>&1) &
  done
  wait
done
cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$OUT"/*.log "$LOGS"/ 2>/dev/null
grep -H -E "^  goal " "$OUT"/*.score.log
grep -H "goal" "$LOGS"/demo_*.log "$LOGS"/s500_*.log
echo "=== score_c0s done $(date)"
