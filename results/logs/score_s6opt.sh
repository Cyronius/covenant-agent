# Score run_s6opt.sh's generations on the laptop (.claude/plans/option-split.md).
#
#   bash results/logs/score_s6opt.sh   # after unpacking s6opt_all.tar.gz into
#                                      # models/tiny/runs/pod_s6opt/
#
# Sandbox scores for every half, `evaluate.py --backoff 3` on the plain exam,
# failure kinds, and R18's tool-choice probe on the four new checkpoints --
# the direct answer to which option breaks lookalike choices.
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_s6opt/out
CKPT=$TINY/runs/pod_s6opt/runs
LOGS=$ROOT/results/logs/s6opt
RUNS="dec25_A0 dtw25_A0 opq25_A0 opq25_SPt"
mkdir -p "$LOGS"

cache_of() { echo "data_cache_${1%%_*}"; }
score() {  # run name split
  (cd "$TINY" && python -W ignore evaluate.py --score --cache $(cache_of $1) --split $3 \
      --gen-out "$OUT/$2.jsonl" > "$OUT/$2.score.log" 2>&1)
}
for run in $RUNS; do
  for half in plain decoy flip; do score $run ${run}_$half holdout & done
  score $run ${run}_test test &
done
wait
for run in $RUNS; do
  (cd "$TINY" && HF_HUB_OFFLINE=1 python -W ignore evaluate.py --generate --backoff 3 \
      --ckpt "runs/pod_s6opt/runs/$run/best.pt" --cache $(cache_of $run) --split holdout --limit 2000 \
      --device cpu --id-not-contains + --gen-out "$OUT/${run}_plain_bk3.jsonl" \
      > "$OUT/${run}_plain_bk3.gen.log" 2>&1; score $run ${run}_plain_bk3 holdout) &
done
(cd "$TINY" && HF_HUB_OFFLINE=1 python -W ignore "$ROOT/results/logs/tool_choice_r18.py" --models \
    "A0 decoys only 25%|runs/pod_s6opt/runs/dec25_A0/best.pt|data_cache_dec25" \
    "A0 decoys+twin 25%|runs/pod_s6opt/runs/dtw25_A0/best.pt|data_cache_dtw25" \
    "A0 opaque names 25%|runs/pod_s6opt/runs/opq25_A0/best.pt|data_cache_opq25" \
    "SPt opaque names 25%|runs/pod_s6opt/runs/opq25_SPt/best.pt|data_cache_opq25" \
    2>&1 | grep -v Warning > "$LOGS/tool_choice.log") &
wait
for run in $RUNS; do
  (cd "$TINY" && python -W ignore fail_kinds.py --cache $(cache_of $run) "$OUT/${run}_plain" \
      --json "$LOGS/fail_kinds_${run}_plain.json" > "$LOGS/fail_kinds_${run}_plain.log" 2>&1)
  cp "$OUT/$run.log" "$LOGS"/
done
cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$LOGS"/ 2>/dev/null
grep -H -E "^  goal " "$OUT"/*.score.log
cat "$LOGS/tool_choice.log"
echo "=== score_s6opt done $(date)"
