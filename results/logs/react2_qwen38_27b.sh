# Reactive execution, second pass: every failure in react_qwen38_27b.sh's
# three arms was a STATIC error (compile_ok == goal_success on all slices),
# so no segment ever raised and the react rule had nothing to act on. Pair
# it with K2 repair (--repair 2): the checker fixes what it can, programs
# run, runtime errors can surface, and react gets its turn.
#   nohup bash react2_qwen38_27b.sh > react2.log 2>&1 &
set -e
cd /workspace/repo
M=/workspace/models/Qwen3.8-27B-UD-IQ3_S.gguf
TAG=qwen38-27b
run() {  # arm slice ctx extra...
  OUT="results/logs/${TAG}_$1_$2"
  python -m baselines.qwen.run_a --model "$M" --template qwen --gpu-layers -1 \
      --tasks "data/holdout/$2.jsonl" --ctx "$3" --domains data/gen/themes ${@:4} \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 $2: $(grep -E '"(goal_success_rate|compile_ok_rate|pauses_mean|tasks_with_pause|error_turns_total|gen_ms_p50)"' "${OUT}.log" | tr -d ' \n')"
}
for ARM in k0r promptr reactr; do
  case $ARM in
    k0r)     X="--repair 2" ;;
    promptr) X="--repair 2 --reactive-prompt" ;;
    reactr)  X="--repair 2 --reactive-prompt --react" ;;
  esac
  run $ARM _smoke8b_known 16384 $X
  run $ARM _smoke8b_demo 16384 $X
  run $ARM _smoke8b_crowded 24576 $X
done
echo "=== react2_qwen38_27b done ==="
