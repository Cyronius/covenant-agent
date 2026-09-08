# Spec 0.4.0 step-0 A/Bs on Qwen3.8-27B (.claude/plans/spec-0.4.0.md §5),
# pod side. Same 25-task slice, GPU, K0 prompt, per-task grammar, greedy,
# 0-shot; each arm's control ran under the identical prompt (R4's rule).
#
#   nohup bash spec040_qwen38_27b.sh > spec040.log 2>&1 &
#
#   base      0.4.0 code, 0.3.x surface: C symbols, no stdlib lines   (== k0 modulo code)
#   stdlib    + MOST/LEAST/EMPTY in prompt and grammar                 (0b)
#   letters   stdlib + typed constant letters                         (0a)
#   kinds     letters + schema enum constants + kind-aware grammar    (0c)
#   think*    the same four arms with --think 1024 (R4: +3 on K0)
#
# Read goal_success per task (compare_react.py); 0b is decided on
# kanban_L4 / crm_L4, 0c on golf_club_L10 / kanban_L5, 0a on compile.
set -e
cd /workspace/repo
M=/workspace/models/Qwen3.8-27B-UD-IQ3_S.gguf
TAG=qwen38-27b
run() {  # arm slice ctx extra...
  OUT="results/logs/${TAG}_$1_$2"
  python -m baselines.qwen.run_a --model "$M" --template qwen --gpu-layers -1 \
      --tasks "data/holdout/$2.jsonl" --ctx "$3" --domains data/gen/themes ${@:4} \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 $2: $(grep -E '"(goal_success_rate|compile_ok_rate|gen_ms_p50)"' "${OUT}.log" | tr -d ' \n')"
}
arms() {  # prefix extra...
  P=$1; shift
  run ${P}base    _smoke8b_known 16384 --no-stdlib "$@"
  run ${P}base    _smoke8b_demo 16384 --no-stdlib "$@"
  run ${P}base    _smoke8b_crowded 24576 --no-stdlib "$@"
  run ${P}stdlib  _smoke8b_known 16384 "$@"
  run ${P}stdlib  _smoke8b_demo 16384 "$@"
  run ${P}stdlib  _smoke8b_crowded 24576 "$@"
  run ${P}letters _smoke8b_known 16384 --symbols typed "$@"
  run ${P}letters _smoke8b_demo 16384 --symbols typed "$@"
  run ${P}letters _smoke8b_crowded 24576 --symbols typed "$@"
  run ${P}kinds   _smoke8b_known 16384 --symbols typed --enums --kinds "$@"
  run ${P}kinds   _smoke8b_demo 16384 --symbols typed --enums --kinds "$@"
  run ${P}kinds   _smoke8b_crowded 24576 --symbols typed --enums --kinds "$@"
}
arms ""
arms think --think 1024
echo "=== spec040_qwen38_27b done ==="
