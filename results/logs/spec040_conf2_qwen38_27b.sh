# Re-check after rewording the EMPTY legend line (R5 §3: the first wording
# made the 27B open crowded programs with `IF NOT EMPTY r0` before anything
# bound r0 — UNBOUND 9/65 stdlib, 17/65 kinds, 1/65 base). Same crowded65,
# same three 0.4.0 arms, no think; base needs no re-run (no EMPTY line).
set -e
cd /workspace/repo
M=/workspace/models/Qwen3.8-27B-UD-IQ3_S.gguf
run() {
  OUT="results/logs/qwen38-27b_$1_$2"
  python -m baselines.qwen.run_a --model "$M" --template qwen --gpu-layers -1 \
      --tasks "data/holdout/$2.jsonl" --ctx "$3" --domains data/gen/themes ${@:4} \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 $2: $(grep -E '"(goal_success_rate|compile_ok_rate)"' "${OUT}.log" | tr -d ' \n')"
}
run v2stdlib  _conf_crowded65 24576
run v2letters _conf_crowded65 24576 --symbols typed
run v2kinds   _conf_crowded65 24576 --symbols typed --enums --kinds
run v2stdlib  e_demo_requests 16384
run v2kinds   e_demo_requests 16384 --symbols typed --enums --kinds
echo "=== spec040_conf2 done ==="
