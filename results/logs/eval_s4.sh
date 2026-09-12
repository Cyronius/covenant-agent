# S4 typed 0.8B eval, pod side (companion to train_s4.sh; run against the
# pruned checkpoint, matching how every S1-S3 headline table was scored).
# Covers the full S3-comparable suite set from results/S2.md's table plus
# the abstain-in-distribution control, so the S4 row lines up cell for cell.
#
#   bash eval_s4.sh [typed|classic] [model.gguf]
#
# the arm has to match the checkpoint - run_a defaults to classic, and the
# first typed eval (R6 §1) forgot the flags, so the typed model was scored
# on C-symbol prompts it never trained on
#
# Run from /workspace/eval on the pod, after:
#   - the eval tarball (git archive + gitignored holdout suites) extracted there
#   - the pruned GGUF copied in
#   - llama-cpp-python built with CUDA (matches the training pod's setup)
#   - node installed (harness.run shells out to runtime/sandbox.js)
set -e
cd /workspace/eval
ARM="${1:-typed}"
case "$ARM" in
  typed)   SURFACE="--symbols typed --enums --kinds"; DEF=qwen3.5-0.8b-s4-pruned-q8.gguf ;;
  classic) SURFACE="";                                 DEF=qwen3.5-0.8b-s4c-pruned-q8.gguf ;;
  *) echo "usage: eval_s4.sh [typed|classic] [model.gguf]"; exit 2 ;;
esac
M=${2:-$DEF}
TAG=$(basename "$M" .gguf)
TPL="--template qwen $SURFACE"
echo "arm: $ARM | model: $M | surface flags: '${SURFACE:-classic}'"
run() {  # suite ctx extra...
  python -m baselines.qwen.run_a --model "$M" --tasks "$1" --ctx "$2" $TPL --gpu-layers -1 ${@:3} \
      --out "results/logs/${TAG}_$(basename "$1" .jsonl).jsonl" \
      > "results/logs/${TAG}_$(basename "$1" .jsonl).log" 2>&1
  echo "done $(basename "$1" .jsonl): $(tail -c 400 "results/logs/${TAG}_$(basename "$1" .jsonl).log" | tr '\n' ' ')"
}
mkdir -p results/logs
run data/holdout/e_real_sessions.jsonl 6144
python -m harness.real_suite score "results/logs/${TAG}_e_real_sessions.jsonl" \
    > "results/logs/${TAG}_e_real_sessions.score" 2>&1
cat "results/logs/${TAG}_e_real_sessions.score"
run data/holdout/e_demo_requests.jsonl 4096
run data/holdout/e_ood_english.jsonl 16384 --domains data/gen/themes
run data/holdout/e_crowded.jsonl 16384 --domains data/gen/themes
run data/holdout/e_crowded_v2.jsonl 16384 --domains data/gen/themes
run data/holdout/e_foreign.jsonl 8192 --domains data/gen/themes
run data/r1_tasks.jsonl 4096                       # E-known (first 1000 rows)
run data/holdout/e_s2_levels.jsonl 16384 --domains data/gen/themes  # abstain control
echo "=== eval_s4 done ==="
