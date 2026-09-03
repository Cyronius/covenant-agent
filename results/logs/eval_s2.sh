# S2 eval (plan .claude/plans/lane-c-retrain.md §5): every suite on the
# pruned S2 Q8 GGUF, then the routing score on the real-session suite.
# Run from the repo root in Git Bash after the pod artefacts are back in
# baselines/qwen/models/:  bash results/logs/eval_s2.sh [model.gguf]
set -e
cd /c/code/covenant-agent
M=${1:-baselines/qwen/models/qwen3.5-0.8b-s2-pruned-q8.gguf}
TAG=$(basename "$M" .gguf)
run() {  # suite ctx extra...
  python -m baselines.qwen.run_a --model "$M" --tasks "$1" --ctx "$2" ${@:3} \
      --out "results/logs/${TAG}_$(basename "$1" .jsonl).jsonl" \
      > "results/logs/${TAG}_$(basename "$1" .jsonl).log" 2>&1
  tail -c 600 "results/logs/${TAG}_$(basename "$1" .jsonl).log"
}
run data/holdout/e_real_sessions.jsonl 6144
python -m harness.real_suite score "results/logs/${TAG}_e_real_sessions.jsonl"
run data/holdout/e_demo_requests.jsonl 4096
run data/holdout/e_ood_english.jsonl 16384 --domains data/gen/themes
run data/holdout/e_crowded.jsonl 16384 --domains data/gen/themes
run data/holdout/e_foreign.jsonl 8192 --domains data/gen/themes
run data/r1_tasks.jsonl 4096            # E-known (first 1000 = the S1 suite)
echo "=== eval_s2 done ==="
