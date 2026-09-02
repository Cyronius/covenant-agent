# Impact of the ABORT line in SYSTEM on the current S1 checkpoint (no retrain):
# demo suite (55) + first 100 R1 tasks. Compare to results/demo_s1_after_a2.jsonl
# and the first 100 rows of results/s1_0.8b_q8_known.jsonl.
cd /c/code/covenant-agent
export PYTHONIOENCODING=utf-8
python -m baselines.qwen.run_a --model baselines/qwen/models/qwen3.5-0.8b-s1-q8.gguf --tasks data/holdout/e_demo_requests.jsonl --out results/demo_s1_after_abort_prompt.jsonl
python -m baselines.qwen.run_a --model baselines/qwen/models/qwen3.5-0.8b-s1-q8.gguf --tasks "C:/Users/josha/AppData/Local/Temp/claude/c--code-covenant-agent/55380b5c-782f-4cb1-839a-899096fe98e2/scratchpad/r1_first100.jsonl" --out results/logs/s1_known100_abort_prompt.jsonl
echo "=== all done ==="
