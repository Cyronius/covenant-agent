# Re-score the two crowded suites under the per-task grammar (condition
# A-grammar-task). Everything before 2026-09-06 ran under the static grammar,
# whose two-digit `num` made F>=100 undecodable — see results/S2.md §S3 and
# harness/task_grammar.py.
#
#   bash results/logs/rescore_crowded_taskgrammar.sh [model.gguf]
#
# Sequential on purpose (two run_a processes on one box thrash the CPU and
# two writing one --out file corrupt it).
set -e
cd /c/code/covenant-agent
M=${1:-baselines/qwen/models/qwen3.5-0.8b-s3-pruned-q8.gguf}
TAG=$(basename "$M" .gguf)
for SUITE in e_crowded e_crowded_v2; do
  OUT="results/logs/${TAG}_${SUITE}_taskgrammar"
  python -m baselines.qwen.run_a --model "$M" \
      --tasks "data/holdout/${SUITE}.jsonl" --ctx 16384 \
      --domains data/gen/themes --out "${OUT}.jsonl" > "${OUT}.log" 2>&1
  echo "done ${SUITE}: $(tail -c 300 "${OUT}.log" | tr '\n' ' ')"
done
echo "=== rescore done ==="
