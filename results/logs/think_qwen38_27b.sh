# Design-tax ladder rung 1 (reactive-execution.md §4): our IR, one-shot,
# + free reasoning. Two-phase decode: up to 1024 tokens in the model's own
# <think> block, unconstrained, then the grammar-constrained program with
# that reasoning in context. Same 25 tasks, same K0 prompt, so think - k0
# is what the <think> ban costs the 27B. Then the same with K2 repair.
#   nohup bash think_qwen38_27b.sh > think.log 2>&1 &
set -e
cd /workspace/repo
M=/workspace/models/Qwen3.8-27B-UD-IQ3_S.gguf
TAG=qwen38-27b
run() {  # arm slice ctx extra...
  OUT="results/logs/${TAG}_$1_$2"
  python -m baselines.qwen.run_a --model "$M" --template qwen --gpu-layers -1 \
      --tasks "data/holdout/$2.jsonl" --ctx "$3" --domains data/gen/themes ${@:4} \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 $2: $(grep -E '"(goal_success_rate|compile_ok_rate|gen_ms_p50|tokens_out_p50)"' "${OUT}.log" | tr -d ' \n')"
}
for ARM in think thinkr; do
  case $ARM in
    think)  X="--think 1024" ;;
    thinkr) X="--think 1024 --repair 2" ;;
  esac
  run $ARM _smoke8b_known 16384 $X
  run $ARM _smoke8b_demo 16384 $X
  run $ARM _smoke8b_crowded 24576 $X
done
echo "=== think_qwen38_27b done ==="
