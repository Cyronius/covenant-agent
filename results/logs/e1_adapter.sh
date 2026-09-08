# E1 (plan .claude/plans/writer-adapter-experiment.md): merged S2 checkpoint
# vs base + runtime LoRA adapter, same suites, grammar on, temp 0.
# Sequential on purpose — both arms are CPU-bound and must not contend.
set -e
cd /c/code/covenant-agent
BASE=baselines/qwen/models/Qwen3.5-0.8B-Q8_0.gguf
MERGED=baselines/qwen/models/qwen3.5-0.8b-s2-q8.gguf
LORA=baselines/qwen/models/lora_s2_f16.gguf
run() {  # tag suite ctx extra...
  python -m baselines.qwen.run_a --tasks "$2" --ctx "$3" "${@:4}" \
      --out "results/logs/e1_$1.jsonl" > "results/logs/e1_$1.log" 2>&1
  echo "=== $1"; tail -c 700 "results/logs/e1_$1.log"
}
run merged_ood  data/holdout/e_ood_english.jsonl 16384 --domains data/gen/themes --model $MERGED
run adapter_ood data/holdout/e_ood_english.jsonl 16384 --domains data/gen/themes --model $BASE --lora $LORA
run merged_demo  data/holdout/e_demo_requests.jsonl 4096 --model $MERGED
run adapter_demo data/holdout/e_demo_requests.jsonl 4096 --model $BASE --lora $LORA
echo "=== e1 done ==="
