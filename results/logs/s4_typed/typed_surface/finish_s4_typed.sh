set -e
cd /workspace/eval
M=qwen3.5-0.8b-s4-pruned-q8-fixed.gguf; TAG=qwen3.5-0.8b-s4-pruned-q8-fixed
TPL="--template qwen --symbols typed --enums --kinds"
run() {
  python -m baselines.qwen.run_a --model "$M" --tasks "$1" --ctx "$2" $TPL --gpu-layers -1 ${@:3} --out "results/logs/${TAG}_$(basename "$1" .jsonl).jsonl" > "results/logs/${TAG}_$(basename "$1" .jsonl).log" 2>&1
  echo "done $(basename "$1" .jsonl): $(tail -c 400 "results/logs/${TAG}_$(basename "$1" .jsonl).log" | tr "\n" " ")"
}
run data/r1_tasks.jsonl 4096
run data/holdout/e_s2_levels.jsonl 16384 --domains data/gen/themes
echo "=== eval_s4 done ==="
