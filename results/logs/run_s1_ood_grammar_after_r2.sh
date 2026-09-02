# Queued behind the R2 condB grammar-arm rerun: S1 OOD suite WITH the fixed grammar
# (the existing s1_0.8b_q8_ood.jsonl is --no-grammar). Launched 2026-09-01.
cd /c/code/covenant-agent
for i in $(seq 1 720); do
  n=$(wc -l < results/r2_b_0.8b_grammar_fixed.jsonl 2>/dev/null || echo 0)
  [ "$n" -ge 1000 ] && break
  sleep 30
done
echo "=== r2 done ($n rows), starting s1 ood grammar $(date) ==="
python -m baselines.qwen.run_a --model baselines/qwen/models/qwen3.5-0.8b-s1-q8.gguf \
  --tasks data/holdout/e_ood_english.jsonl \
  --out results/s1_0.8b_q8_ood_grammar.jsonl \
  --domains data/gen/themes --ctx 16384
echo "=== all done $(date) ==="
