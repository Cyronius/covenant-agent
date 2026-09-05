# The four E-suites (+ demo requests) for one model, sequentially.
# eval_s2.sh also runs e_real_sessions and its routing score; this is the
# rest, for when the real suite has already been run separately.
#
#   bash results/logs/eval_esuites.sh baselines/qwen/models/<model>.gguf
#
# Sequential on purpose: two run_a processes writing the same --out file
# corrupt it, and the reboot on 2026-09-04 left four overlapping runs behind
# (two of them on the same output). Check nothing else is mid-run first:
#   powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -like '*run_a*'} | Select ProcessId,CommandLine"
set -e
cd /c/code/covenant-agent
M=${1:-baselines/qwen/models/qwen3.5-0.8b-s2r-pruned-q8.gguf}
TPL="--template ${TEMPLATE:-qwen}"
TAG=$(basename "$M" .gguf)
run() {  # suite ctx extra...
  python -m baselines.qwen.run_a --model "$M" --tasks "$1" --ctx "$2" $TPL ${@:3} \
      --out "results/logs/${TAG}_$(basename "$1" .jsonl).jsonl" \
      > "results/logs/${TAG}_$(basename "$1" .jsonl).log" 2>&1
  echo "done $(basename "$1" .jsonl): $(tail -c 200 "results/logs/${TAG}_$(basename "$1" .jsonl).log" | tr '\n' ' ')"
}
run data/holdout/e_demo_requests.jsonl 4096
run data/holdout/e_ood_english.jsonl 16384 --domains data/gen/themes
run data/holdout/e_crowded.jsonl 16384 --domains data/gen/themes
run data/holdout/e_foreign.jsonl 8192 --domains data/gen/themes
run data/r1_tasks.jsonl 4096
echo "=== eval_esuites done ==="
