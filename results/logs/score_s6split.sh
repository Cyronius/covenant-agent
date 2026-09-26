# Score run_s6split.sh's and run_s6dec.sh's generations on the laptop, as done
# for results/R14.md (.claude/plans/s5-first-s6-cost-split.md).
#
#   bash results/logs/score_s6split.sh    # after unpacking s6split_all.tar.gz and
#                                          # s6dec_all.tar.gz into models/tiny/runs/pod_s6split/
#
# S5-era tasks cannot run in today's sandbox (the theme rewrite a03610e renamed
# their tools), so data_cache_s5g's S5 plain half and its test split are scored
# from the worktree at 212699a, as R13 did. Its decoyed rows carry their own
# sandbox and its +s6 rows are today's themes: both score here, as do S6's
# decoyed and flip rows in data_cache_s5g_s6dec (the evaluation-only cache).
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OLD=$ROOT/../covenant-pre-a03610e/models/tiny
OUT=$TINY/runs/pod_s6split/out
LOGS=$ROOT/results/logs/s6split
mkdir -p "$LOGS"

here() {  # cache split name
  (cd "$TINY" && python -W ignore evaluate.py --score --cache $1 --split $2 \
      --gen-out "$OUT/$3.jsonl" > "$OUT/$3.score.log" 2>&1)
}
old() {   # cache split name  (old themes, absolute paths)
  (cd "$OLD" && python -W ignore evaluate.py --score --cache "C:/code/covenant-agent/models/tiny/$1" \
      --split $2 --gen-out "C:/code/covenant-agent/models/tiny/runs/pod_s6split/out/$3.jsonl" \
      > "$OUT/$3.score.log" 2>&1)
}

for run in s5g_A0 s5g_SPt; do
  old  data_cache_s5g holdout ${run}_plain &
  here data_cache_s5g holdout ${run}_decoy &
  here data_cache_s5g holdout ${run}_s6plain &
  old  data_cache_s5g test    ${run}_test &
done
wait
for run in s5g_A0 s5g_SPt; do
  here data_cache_s5g_s6dec holdout ${run}_s6decoy &
  here data_cache_s5g_s6dec holdout ${run}_s6flip &
done
for half in plain decoy flip; do
  here data_cache_s6off holdout s6off_A0_$half &
done
here data_cache_s6off test s6off_A0_test &
wait

for run in s5g_A0 s5g_SPt; do
  (cd "$TINY" && python -W ignore fail_kinds.py --cache data_cache_s5g "$OUT/${run}_s6plain" \
      --json "$LOGS/fail_kinds_${run}_s6plain.json" > "$LOGS/fail_kinds_${run}_s6plain.log" 2>&1)
done
(cd "$TINY" && python -W ignore fail_kinds.py --cache data_cache_s6off "$OUT/s6off_A0_plain" \
    --json "$LOGS/fail_kinds_s6off_plain.json" > "$LOGS/fail_kinds_s6off_plain.log" 2>&1)
(cd "$ROOT" && python results/logs/s6split_matched.py)

cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$OUT"/nvidia_smi_4min.txt "$LOGS"/ 2>/dev/null
grep -H " goal" "$OUT"/*.score.log
echo "=== score_s6split done $(date)"
