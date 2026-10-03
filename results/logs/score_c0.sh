# Score run_c0.sh's generations on the laptop (C0, .claude/plans/tiny-general-agent-menu.md),
# exactly as score_clt.sh scored R23's, so c0_* sits beside clt_RD / clt_RDs1.
#
#   bash results/logs/score_c0.sh   # after unpacking c0_all.tar.gz into models/tiny/runs/pod_c0/
#
# Sandbox scores for every half (plain, decoy, flip, clut) and test;
# evaluate.py --backoff 3 on plain and clut; and play.py (the demo's own loop,
# live reader, backoff 3) on the 70 demo requests and the 500-task
# plain/cluttered exam samples.
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_c0/out
LOGS=$ROOT/results/logs/c0
RUNS="c0_RL c0_RLs1 c0_RLr c0_RLrs1 c0_RVr c0_RVrs1"
CACHE=data_cache_c0
READER=$ROOT/models/tiny/reader
# the role-aware reader the c0_*r runs' table (reader_role.pt) was built from;
# play.py checks its sha256 against the table's
ROLE_READER=${ROLE_READER:-$ROOT/models/tiny/reader/role/r5/reader.pt}
reader_for() { case $1 in *r|*rs1) echo "$ROLE_READER" ;; *) echo "$READER" ;; esac; }
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
        --ckpt "runs/pod_c0/runs/$run/best.pt" --cache $CACHE --split holdout --limit 2000 \
        --device cpu $sel --gen-out "$OUT/${run}_${half}_bk3.jsonl" \
        > "$OUT/${run}_${half}_bk3.gen.log" 2>&1; score ${run}_${half}_bk3 holdout) &
  done
  wait
done
for run in $RUNS; do
  (cd "$TINY" && python -W ignore play.py --ckpt runs/pod_c0/runs/$run/best.pt --cache $CACHE \
      --tasks ../../data/holdout/e_demo_requests.jsonl --retype --reader-live $(reader_for $run) --backoff 3 \
      --show 0 --out "$LOGS/demo_$run.jsonl" > "$LOGS/demo_$run.log" 2>&1) &
  for v in s6 clt; do
    (cd "$TINY" && python -W ignore play.py --ckpt runs/pod_c0/runs/$run/best.pt --cache $CACHE \
        --tasks ../../data/${v}_holdout_s500.jsonl --reader-live $(reader_for $run) --backoff 3 \
        --show 0 --out "$LOGS/s500_${v}_$run.jsonl" > "$LOGS/s500_${v}_$run.log" 2>&1) &
  done
  wait
done
cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$OUT"/*.log "$LOGS"/ 2>/dev/null
grep -H -E "^  goal " "$OUT"/*.score.log
grep -H "goal" "$LOGS"/demo_*.log "$LOGS"/s500_*.log
echo "=== score_c0 done $(date)"
