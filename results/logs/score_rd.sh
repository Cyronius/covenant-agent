# Score run_rd.sh's generations on the laptop (.claude/plans/npu-planner.md
# phase 2).
#
#   bash results/logs/score_rd.sh   # after unpacking rd_all.tar.gz into models/tiny/runs/pod_rd/
#
# Sandbox scores for every half, `evaluate.py --backoff 3` on the plain exam,
# R18's tool-choice probe, and the R20/R21 comparisons (llm_baseline.py
# report, reader_bakeoff.py score) with the four new models in them.
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_rd/out
LOGS=$ROOT/results/logs/rd
RUNS="s6off_RD dtw25_RD fdc25_RD fdc25_A0"
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
      --ckpt "runs/pod_rd/runs/$run/best.pt" --cache $(cache_of $run) --split holdout --limit 2000 \
      --device cpu --id-not-contains + --gen-out "$OUT/${run}_plain_bk3.jsonl" \
      > "$OUT/${run}_plain_bk3.gen.log" 2>&1; score $run ${run}_plain_bk3 holdout) &
done
(cd "$TINY" && HF_HUB_OFFLINE=1 python -W ignore "$ROOT/results/logs/tool_choice_r18.py" --models \
    "A0+reader, options-off|runs/pod_rd/runs/s6off_RD/best.pt|data_cache_s6off" \
    "A0+reader, 25% decoys+twin|runs/pod_rd/runs/dtw25_RD/best.pt|data_cache_dtw25" \
    "A0+reader, 25% flip decoys|runs/pod_rd/runs/fdc25_RD/best.pt|data_cache_fdc25" \
    "A0, 25% flip decoys|runs/pod_rd/runs/fdc25_A0/best.pt|data_cache_fdc25" \
    2>&1 | grep -v Warning > "$LOGS/tool_choice.log") &
wait
for run in $RUNS; do
  (cd "$TINY" && python -W ignore fail_kinds.py --cache $(cache_of $run) "$OUT/${run}_plain" \
      --json "$LOGS/fail_kinds_${run}_plain.json" > "$LOGS/fail_kinds_${run}_plain.log" 2>&1)
  cp "$OUT/$run.log" "$LOGS"/
done
cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$LOGS"/ 2>/dev/null
(cd "$ROOT" && python -W ignore results/logs/llm_baseline.py report > /dev/null && \
 python -W ignore results/logs/reader_bakeoff.py score > "$LOGS/reader_bakeoff.log")
grep -H -E "^  goal " "$OUT"/*.score.log
cat "$LOGS/tool_choice.log"
echo "=== score_rd done $(date)"
