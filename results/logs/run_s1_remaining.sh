set -e
cd /c/code/covenant-agent
MODEL=baselines/qwen/models/qwen3.5-0.8b-s1-q8.gguf

echo "=== crowded $(date) ==="
python -m baselines.qwen.run_a --model "$MODEL" \
  --tasks data/holdout/e_crowded.jsonl \
  --out results/s1_0.8b_q8_crowded.jsonl \
  --no-grammar --domains data/gen/themes --ctx 16384

echo "=== known $(date) ==="
python -m baselines.qwen.run_a --model "$MODEL" \
  --tasks data/r1_tasks.jsonl \
  --out results/s1_0.8b_q8_known.jsonl \
  --no-grammar

echo "=== ood $(date) ==="
python -m baselines.qwen.run_a --model "$MODEL" \
  --tasks data/holdout/e_ood_english.jsonl \
  --out results/s1_0.8b_q8_ood.jsonl \
  --no-grammar --domains data/gen/themes --ctx 16384

echo "=== all done $(date) ==="
