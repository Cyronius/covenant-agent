# Spec 0.4.0 confirmation (plan §5: anything that moves >=2 tasks on the
# 25-slice is confirmed on a bigger slice): e_demo_requests (65) and a seeded
# 65-task sample of e_crowded_v2, the same 8 arms as spec040_qwen38_27b.sh.
#   nohup bash spec040_conf_qwen38_27b.sh > spec040_conf.log 2>&1 &
set -e
cd /workspace/repo
M=/workspace/models/Qwen3.8-27B-UD-IQ3_S.gguf
TAG=qwen38-27b
run() {  # arm slice ctx extra...
  OUT="results/logs/${TAG}_$1_$2"
  python -m baselines.qwen.run_a --model "$M" --template qwen --gpu-layers -1 \
      --tasks "data/holdout/$2.jsonl" --ctx "$3" --domains data/gen/themes ${@:4} \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 $2: $(grep -E '"(goal_success_rate|compile_ok_rate)"' "${OUT}.log" | tr -d ' \n')"
}
arms() {  # prefix extra...
  P=$1; shift
  for S in "e_demo_requests 16384" "_conf_crowded65 24576"; do
    set -- $S "$@"; SL=$1; CTX=$2; shift 2
    run ${P}base    $SL $CTX --no-stdlib "$@"
    run ${P}stdlib  $SL $CTX "$@"
    run ${P}letters $SL $CTX --symbols typed "$@"
    run ${P}kinds   $SL $CTX --symbols typed --enums --kinds "$@"
  done
}
arms ""
arms think --think 1024
echo "=== spec040_conf done ==="
