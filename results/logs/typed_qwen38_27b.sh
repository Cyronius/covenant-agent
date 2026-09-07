# C4 typed-slot grammar on Qwen3.8-27B, with its control. The 8 tasks that
# failed to compile at K0 (results/logs/qwen38-27b__smoke8b_*_0shot), run
# twice under the CURRENT prompt: plain task grammar (control) and typed
# slots. Same weights, same prompt, K0 — the delta between the two arms is
# the grammar's alone.
#
# Why a control and not the K0 rows: the SYSTEM prompt gained the
# `ABORT reason sym` line after K0 ran. A first attempt compared typed-under-
# new-prompt against K0-under-old-prompt and was stopped as confounded; a
# check-then-decline paragraph added at the same time cost 2/10 on the demo
# slice without producing the pattern once, and was reverted.
#
#   bash results/logs/typed_qwen38_27b.sh
#
# Read: goal_success, not compile — a typed slot forbids the wrong symbol and
# the sampler picks the best legal one, which can be right or merely legal.
set -e
cd /c/code/covenant-agent
M="C:/Users/josha/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
run() {  # slice ctx mode
  OUT="results/logs/qwen38-27b_$3_$1"
  python -m baselines.qwen.run_a --model "$M" --template qwen --grammar-mode "$3" \
      --tasks "data/holdout/_repair38_$1.jsonl" --ctx "$2" --domains data/gen/themes \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 $3: $(grep -E '\"(goal_success_rate|compile_ok_rate)\"' "${OUT}.log" | tr -d ' \n')"
}
for MODE in task typed; do
  run small 4096 $MODE
  run crowded 8192 $MODE
done
echo "=== typed_qwen38_27b done ==="
