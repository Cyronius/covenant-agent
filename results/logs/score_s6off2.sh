# Score run_s6off2.sh's generations on the laptop (results/R15.md "Next" 2).
#
#   bash results/logs/score_s6off2.sh   # after unpacking s6off2_all.tar.gz into
#                                       # models/tiny/runs/pod_s6off2/
#
# All four halves are S6 themes, so everything scores in today's tree (no
# pre-a03610e worktree, unlike R14's S5 halves). Tool backoff
# (tool_backoff_r15.py) needs the typechecker, so it runs here, on the plain
# exam for both runs and on the decoyed exam for SPt.
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_s6off2/out
CKPT=$TINY/runs/pod_s6off2/runs
LOGS=$ROOT/results/logs/s6off2
mkdir -p "$LOGS"

score() {  # name
  (cd "$TINY" && python -W ignore evaluate.py --score --cache data_cache_s6off --split holdout \
      --gen-out "$OUT/$1.jsonl" > "$OUT/$1.score.log" 2>&1)
}
backoff() {  # run tag selector...
  local run=$1 tag=$2; shift 2
  (cd "$TINY" && HF_HUB_OFFLINE=1 python -W ignore "$ROOT/results/logs/tool_backoff_r15.py" \
      --ckpt "$CKPT/$run/best.pt" --cache data_cache_s6off "$@" --limit 2000 \
      --gen-out "$OUT/${run}_${tag}_tb3.jsonl" > "$OUT/${run}_${tag}_tb3.gen.log" 2>&1)
  score ${run}_${tag}_tb3
}

for run in s6off_SPt s6off_A0s1; do
  for half in plain decoy flip; do score ${run}_$half & done
  (cd "$TINY" && python -W ignore evaluate.py --score --cache data_cache_s6off --split test \
      --gen-out "$OUT/${run}_test.jsonl" > "$OUT/${run}_test.score.log" 2>&1) &
done
wait
backoff s6off_SPt plain &
backoff s6off_A0s1 plain &
backoff s6off_SPt decoy --id-contains +decoy --id-not-contains "" &
wait
for run in s6off_SPt s6off_A0s1; do
  (cd "$TINY" && python -W ignore fail_kinds.py --cache data_cache_s6off "$OUT/${run}_plain" \
      --json "$LOGS/fail_kinds_${run}_plain.json" > "$LOGS/fail_kinds_${run}_plain.log" 2>&1)
done
cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$OUT"/s6off_*.log "$LOGS"/ 2>/dev/null
grep -H -E "^  goal " "$OUT"/*.score.log
echo "=== score_s6off2 done $(date)"
