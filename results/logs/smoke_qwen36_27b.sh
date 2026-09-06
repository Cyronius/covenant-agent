# Feasibility smoke test: Qwen3.6-27B-A2.8B (IQ3 experts / Q4 head), untuned,
# in-context. Same 25-task slice and the same two arms as smoke_lfm8b.sh, so
# the two are directly comparable.
#
#   bash results/logs/smoke_qwen36_27b.sh
#
# The GGUF is LM Studio's copy, loaded through our own llama-cpp-python: LM
# Studio's OpenAI-compatible endpoint silently ignores a GBNF grammar (a
# grammar admitting only "HELLO" still returned "STOP"), so running through it
# would drop the per-task grammar and break comparability with every other
# arm. Same weights either way; `qwen35moe` loads in llama-cpp-python 0.3.35.
#
# --template qwen, not chat. This model's own chat template ends the
# generation prompt with an opening think tag unless enable_thinking=false,
# so under chat the model is primed to open a reasoning block while the grammar
# forbids every token that block would contain. It then wrote `IF r0 ...`
# without ever binding r0: 24 of 25 rows UNBOUND, 0 compiles, and four worked
# examples did not move it. The hand-rolled ChatML with an explicit empty
# think block is what the tuned Qwen arms use, and it is what this model
# needs. Same slice, same weights, template only: compile 0% -> 43%,
# goal 0/7 -> 2/7 on E-known at 4 shots.
set -e
cd /c/code/covenant-agent
M=${1:-"C:/Users/josha/.lmstudio/models/cyronius/qwen36-27b-a2.8b-mtp-iq3exp-q4head/qwen36-27b-a2.8b-mtp-iq3exp-q4head.gguf"}
TAG=qwen36-27b
run() {  # suite ctx shots
  OUT="results/logs/${TAG}_$1_${3}shot"
  python -m baselines.qwen.run_a --model "$M" --template qwen --shots "$3" \
      --tasks "data/holdout/$1.jsonl" --ctx "$2" --domains data/gen/themes \
      --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done $1 ${3}-shot: $(grep -E '\"(goal_success_rate|compile_ok_rate|parse_ok_rate|gen_ms_p50)\"' "${OUT}.log" | tr -d ' \n')"
}
for S in 0 4; do
  run _smoke8b_known 16384 $S
  run _smoke8b_demo 16384 $S
  run _smoke8b_crowded 24576 $S
done
echo "=== smoke_qwen36_27b done ==="
