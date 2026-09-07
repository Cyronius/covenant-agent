# Step 0 of .claude/plans/abort-referent.md: does a model that can do the
# task produce USEFUL referents when merely asked? Qwen3.8-27B, K0, with the
# SYSTEM prompt's new `ABORT reason sym` line. Two slices: the 10 demo
# requests from the smoke slice (S3 aborts 4 of them) and 15 real-session
# abstain rows. Reads: for each abort, which symbol it named and whether
# harness/abort_check.py calls it founded. Not a scored arm.
#
#   bash results/logs/probe_qwen38_referent.sh
set -e
cd /c/code/covenant-agent
M="C:/Users/josha/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
run() {  # slice ctx
  OUT="results/logs/qwen38-27b_referent_$1"
  python -m baselines.qwen.run_a --model "$M" --template qwen \
      --tasks "data/holdout/$2" --ctx "$3" --domains data/gen/themes \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1: $(grep -E '\"(goal_success_rate|compile_ok_rate)\"' "${OUT}.log" | tr -d ' \n')"
}
run demo _smoke8b_demo.jsonl 4096
run real _probe38_abstain.jsonl 8192
echo "=== probe_qwen38_referent done ==="
