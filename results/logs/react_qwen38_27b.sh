# Reactive execution on Qwen3.8-27B (reactive-execution.md §7 step 1), pod
# side. Same 25-task slice as smoke_qwen38_27b.sh (K0 = 16/25 on the CPU).
#
#   nohup bash react_qwen38_27b.sh > react.log 2>&1 &
#
# Three arms, in order, every one under the per-task grammar, 0-shot, greedy:
#   k0      the K0 prompt, one-shot          -- does the GPU build reproduce the CPU's 16/25?
#   prompt  + observe-then-decide paragraph  -- what teaching PAUSE does on its own
#   react   + runtime error -> planner turn  -- the harness rule's own delta
# `prompt` and `react` share a prompt byte for byte; only the harness rule
# differs, so react - prompt is the number the plan asks for. k0 - prompt is
# the prompt's cost or gain, confounded by greedy flips as always (S2.md).
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
for ARM in k0 prompt react; do
  case $ARM in
    k0)     X="" ;;
    prompt) X="--reactive-prompt" ;;
    react)  X="--reactive-prompt --react" ;;
  esac
  run $ARM _smoke8b_known 16384 $X
  run $ARM _smoke8b_demo 16384 $X
  run $ARM _smoke8b_crowded 24576 $X
done
echo "=== react_qwen38_27b done ==="
