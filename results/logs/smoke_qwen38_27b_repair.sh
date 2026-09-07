# Qwen3.8-27B under PLAN.md §6 condition K2: the checker's diagnostics go back
# to the model as input, up to 2 repair rounds (`run_a.py --repair 2`).
#
#   bash results/logs/smoke_qwen38_27b_repair.sh
#
# This runs only the 8 tasks of the 25-task slice that failed to COMPILE at K0
# (results/logs/qwen38-27b__smoke8b_*_0shot.jsonl). That is not a shortcut in
# the result: decoding is greedy at temperature 0 and a repair round only fires
# on a compile failure, so the other 17 rows are bit-identical to their K0 run
# by construction. It is a shortcut in the bill — crowded is ~20 min/task on
# this CPU. `_merge` below reassembles the full 25 for scoring.
#
# The one K0 failure not included is kanban_L4_20261095: it ABORTed, which
# compiles, so no diagnostic exists to feed back. K2 cannot reach a wrong
# decision, only a malformed program — worth remembering when reading the delta.
#
# Context windows are sized to the measured prompts (known/demo max 1,320
# tokens, crowded max 5,634) rather than the 16k/24k the K0 run allocated; the
# KV cache was costing more than the tasks needed.
set -e
cd /c/code/covenant-agent
M="C:/Users/josha/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
TAG=qwen38-27b
run() {  # slice ctx
  OUT="results/logs/${TAG}_repair_$1"
  python -m baselines.qwen.run_a --model "$M" --template qwen --repair 2 \
      --tasks "data/holdout/_repair38_$1.jsonl" --ctx "$2" \
      --domains data/gen/themes \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1: $(grep -E '\"(goal_success_rate|compile_ok_rate)\"' "${OUT}.log" | tr -d ' \n')"
}
run small 4096
run crowded 8192
echo "=== smoke_qwen38_27b_repair done ==="
