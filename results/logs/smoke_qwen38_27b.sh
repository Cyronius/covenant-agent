# Qwen3.8-27B (dense, UD-IQ3_S), untuned, in-context, per-task grammar.
# Same 25-task slice as smoke_lfm8b.sh and smoke_qwen36_27b.sh.
#
#   bash results/logs/smoke_qwen38_27b.sh [0|4]
#
# Dense 27B, not MoE: ~10x the active parameters of the 3.6-A2.8B at the same
# file size, and ~264 s/task on this CPU against the 3.6's 32 s. 0-shot only by
# default — on the first task it reproduced the reference program exactly, so
# the shots arm is only worth paying for if 0-shot stalls.
#
# --template qwen (hand-rolled ChatML + explicit empty think block), not chat:
# see smoke_qwen36_27b.sh for what the thinking prefix does under a grammar.
set -e
cd /c/code/covenant-agent
M="C:/Users/josha/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
S=${1:-0}
TAG=qwen38-27b
run() {  # suite ctx
  OUT="results/logs/${TAG}_$1_${S}shot"
  python -m baselines.qwen.run_a --model "$M" --template qwen --shots "$S" \
      --tasks "data/holdout/$1.jsonl" --ctx "$2" --domains data/gen/themes \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 ${S}-shot: $(grep -E '\"(goal_success_rate|compile_ok_rate|parse_ok_rate|gen_ms_p50)\"' "${OUT}.log" | tr -d ' \n')"
}
run _smoke8b_known 16384
run _smoke8b_demo 16384
run _smoke8b_crowded 24576
echo "=== smoke_qwen38_27b done ==="
