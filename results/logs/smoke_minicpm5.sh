# MiniCPM5 (openbmb), untuned, gate 2 of the base-model bake-off: the same
# 25-task slice as smoke_qwen38_27b.sh, per-task grammar, 0-shot, 4-shot,
# and 0-shot + 512 think tokens. --template qwen: MiniCPM5's own template
# is ChatML and opens a <think> block unless enable_thinking=false, which
# create_chat_completion never passes -- the Qwen3.6 trap -- and our
# hand-rolled template with the explicit empty think block is exactly the
# thinking-off form. Runs on a pod (ROOT=/workspace/repo, all layers on the
# GPU); on the dev box the 2B took 13-18 s per small task at 4 threads.
#   bash results/logs/smoke_minicpm5.sh <model.gguf> <tag>
set -e
cd "${ROOT:-/workspace/repo}"
M="$1"; TAG="$2"
run() {  # arm slice ctx extra...
  OUT="results/logs/${TAG}_$1_$2"
  python -m baselines.qwen.run_a --model "$M" --template qwen --gpu-layers -1 \
      --tasks "data/holdout/$2.jsonl" --ctx "$3" --domains data/gen/themes ${@:4} \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 $2: $(grep -E '"(goal_success_rate|compile_ok_rate|parse_ok_rate|gen_ms_p50)"' "${OUT}.log" | tr -d ' \n')"
}
for ARM in 0shot 4shot think; do
  case $ARM in
    0shot) X="" ;;
    4shot) X="--shots 4" ;;
    think) X="--think 512" ;;
  esac
  run $ARM _smoke8b_known 16384 $X
  run $ARM _smoke8b_demo 16384 $X
  run $ARM _smoke8b_crowded 24576 $X
done
echo "=== smoke_minicpm5 $TAG done ==="
