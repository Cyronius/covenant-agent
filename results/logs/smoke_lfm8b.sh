# Feasibility smoke test: LFM2.5-8B-A1B (untuned, in-context) as a candidate
# teacher. 25 tasks, run twice — one-shot (the SYSTEM prompt's single example,
# the arm the owner asked about) and 4-shot. NOT a scored arm: the slices are
# small random samples, kept small on purpose.
#
#   bash results/logs/smoke_lfm8b.sh [model.gguf]
#
# Condition A with the per-task grammar (harness/task_grammar.py) and the
# model's own chat template, the same path the LFM2.5-350M arm used.
set -e
cd /c/code/covenant-agent
M=${1:-baselines/qwen/models/LFM2.5-8B-A1B-Q8_0.gguf}
TAG=$(basename "$M" .gguf)
run() {  # suite ctx shots
  OUT="results/logs/${TAG}_$1_${3}shot"
  python -m baselines.qwen.run_a --model "$M" --template chat --shots "$3" \
      --tasks "data/holdout/$1.jsonl" --ctx "$2" --domains data/gen/themes \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 ${3}-shot: $(grep -E '\"(goal_success_rate|compile_ok_rate|parse_ok_rate|gen_ms_p50)\"' "${OUT}.log" | tr -d ' \n')"
}
for S in 0 4; do
  run _smoke8b_known 16384 $S
  run _smoke8b_demo 16384 $S
  run _smoke8b_crowded 24576 $S
done
echo "=== smoke_lfm8b done ==="
