# S3 under the 0.4.0 surface, unchanged weights (plan spec-0.4.0.md §5):
# what does moving the surface cost a model trained on the old one? The
# tuned 0.8B has never seen an `S0`, an `IF EMPTY`, or a `MOST` — it was
# trained on C symbols and the 14-line count loop. This is the floor the S4
# retrain has to beat, and the number that says how much of any S4 gain is
# the corpus rather than the surface.
#
# Same decoding as every S3 score (per-task grammar, single shot); the only
# thing that moves between arms is the symbol table, re-rendered from the
# same stored suites by harness/retype.py. So `base` here reproduces the
# published S3 rows and the rest are that model on a surface it has never
# been trained for.
#
#   base    the S3 surface: C symbols, no MOST/LEAST/EMPTY in prompt or grammar
#   stdlib  + the composites (the corpus teaches these in S4)
#   kinds   + typed letters, string kinds, schema enum constants
#
# Pod, not the dev box: 805 tasks x 3 arms, and the crowded prompts are
# 20k+ tokens each.
#
#   nohup bash rescore_s3_040.sh > rescore_s3_040.log 2>&1 &
set -e
cd /workspace/repo
M=/workspace/models/qwen3.5-0.8b-s3-pruned-q8.gguf
TAG=s3-040
run() {  # arm slice ctx extra...
  OUT="results/logs/${TAG}_$1__$2"
  python -m baselines.qwen.run_a --model "$M" --template qwen --gpu-layers -1 \
      --tasks "data/holdout/$2.jsonl" --ctx "$3" --domains data/gen/themes ${@:4} \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 $2: $(grep -E '"(goal_success_rate|compile_ok_rate)"' "${OUT}.log" | tr -d ' \n')"
}
for S in "e_s2_levels 16384" "e_crowded_v2 24576" "e_demo_requests 16384" "e_ood_english 16384"; do
  set -- $S; SL=$1; CTX=$2
  run base    $SL $CTX --no-stdlib
  run stdlib  $SL $CTX
  run kinds   $SL $CTX --symbols typed --enums --kinds
done
# routing, not goal: the real-session slice is scored by harness.real_suite
for A in "base --no-stdlib" "kinds --symbols typed --enums --kinds"; do
  set -- $A; ARM=$1; shift
  run $ARM e_real_sessions 16384 "$@"
  python -m harness.real_suite score "results/logs/${TAG}_${ARM}__e_real_sessions.jsonl" \
      > "results/logs/${TAG}_${ARM}__e_real_sessions.score" 2>&1
done
echo "=== rescore_s3_040 done ==="
