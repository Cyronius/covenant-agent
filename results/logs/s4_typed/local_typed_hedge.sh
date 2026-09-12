# local CPU hedge for the typed-surface re-score: short suites first, outputs
# suffixed _typed so they never collide with the pod's files
set -e
M=baselines/qwen/models/qwen3.5-0.8b-s4-pruned-q8-fixed.gguf
TAG=qwen3.5-0.8b-s4-pruned-q8-fixed
S="--template qwen --symbols typed --enums --kinds --gpu-layers 0"
run() {
  python -m baselines.qwen.run_a --model "$M" --tasks "$1" --ctx "$2" $S ${@:3} \
    --out "results/logs/s4_typed/${TAG}_$(basename "$1" .jsonl)_typed.jsonl" \
    > "results/logs/s4_typed/${TAG}_$(basename "$1" .jsonl)_typed.log" 2>&1
  echo "done $(basename "$1" .jsonl)"
}
run data/holdout/e_demo_requests.jsonl 4096
run data/holdout/e_real_sessions.jsonl 6144
python -m harness.real_suite score "results/logs/s4_typed/${TAG}_e_real_sessions_typed.jsonl" > "results/logs/s4_typed/${TAG}_e_real_sessions_typed.score" 2>&1 || true
run data/holdout/e_s2_levels.jsonl 16384 --domains data/gen/themes
run data/holdout/e_ood_english.jsonl 16384 --domains data/gen/themes
run data/holdout/e_foreign.jsonl 8192 --domains data/gen/themes
run data/holdout/e_crowded.jsonl 16384 --domains data/gen/themes
run data/holdout/e_crowded_v2.jsonl 16384 --domains data/gen/themes
run data/r1_tasks.jsonl 4096
echo "=== local hedge done ==="
