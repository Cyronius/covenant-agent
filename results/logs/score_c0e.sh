# Score run_c0e.sh's generations on the laptop (the ELECTRA reader,
# .claude/plans/electra-only-reader.md step 3), as score_c0s.sh scores the
# slot reader's. Greedy generation ran on the pod; this runs what needs the
# Node sandbox, on the CPU: evaluate.py --generate --backoff 3 on plain and clut
# (its compile check replays paused tasks through harness.run), evaluate.py
# --score on every generation, and play.py (the demo's own loop, the ELECTRA
# reader live, backoff 3) on the 70 demo requests. S500=1 adds the 500-task
# plain and cluttered samples through play.py (much longer).
#
#   bash results/logs/score_c0e.sh   # after unpacking c0e_all.tar.gz into models/tiny/runs/pod_c0e/
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_c0e/out
LOGS=$ROOT/results/logs/c0e
RUNS=${RUNS:-"c0_EL c0_ELs1 c0_ESC c0_ESCs1 c0_ESCV c0_ESCVs1"}
PHASE=${PHASE:-all}      # score | bk3 | demo | all
cache_for() { case $1 in c0_EL*) echo data_cache_c0e ;; *) echo data_cache_c0esc ;; esac; }
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
mkdir -p "$LOGS"

score() {  # name split run
  (cd "$TINY" && python -W ignore evaluate.py --score --cache $(cache_for $3) --split $2 \
      --gen-out "$OUT/$1.jsonl" > "$OUT/$1.score.log" 2>&1)
}
[ $PHASE = all ] || [ $PHASE = score ] && for run in $RUNS; do
  for half in plain decoy flip clut; do score ${run}_$half holdout $run & done
  score ${run}_test test $run &
  wait
done
[ $PHASE = all ] || [ $PHASE = bk3 ] && for run in $RUNS; do
  for half in plain clut; do
    case $half in plain) sel="--id-not-contains +";; clut) sel="--id-contains +clut";; esac
    (cd "$TINY" && python -W ignore evaluate.py --generate --backoff 3         --ckpt "runs/pod_c0e/runs/$run/best.pt" --cache $(cache_for $run) --split holdout --limit 2000         --device cpu $sel --gen-out "$OUT/${run}_${half}_bk3.jsonl"         > "$OUT/${run}_${half}_bk3.gen.log" 2>&1; score ${run}_${half}_bk3 holdout $run) &
  done
  wait
done
[ $PHASE = all ] || [ $PHASE = demo ] && for run in $RUNS; do
  (cd "$TINY" && python -W ignore play.py --ckpt runs/pod_c0e/runs/$run/best.pt --cache $(cache_for $run) \
      --tasks ../../data/holdout/e_demo_requests.jsonl --retype --reader-live electra --backoff 3 \
      --show 0 --out "$LOGS/demo_$run.jsonl" > "$LOGS/demo_$run.log" 2>&1)
  if [ "${S500:-0}" = 1 ]; then
    for v in s6 clt; do
      (cd "$TINY" && python -W ignore play.py --ckpt runs/pod_c0e/runs/$run/best.pt --cache $(cache_for $run) \
          --tasks ../../data/${v}_holdout_s500.jsonl --reader-live electra --backoff 3 \
          --show 0 --out "$LOGS/s500_${v}_$run.jsonl" > "$LOGS/s500_${v}_$run.log" 2>&1)
    done
  fi
done
cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$OUT"/*.log "$LOGS"/ 2>/dev/null
grep -H -E "^  goal " "$OUT"/*.score.log
grep -H "goal" "$LOGS"/demo_*.log "$LOGS"/s500_*.log 2>/dev/null
echo "=== score_c0e done $(date)"
