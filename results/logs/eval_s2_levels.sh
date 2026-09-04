# Abstain-in-distribution control (2026-09-04). Separates "the model is too
# small to learn abstention" from "the model never saw abstention phrased
# like a real request".
#
# Builds a holdout at the seven NEW S2 levels on RESERVED worlds (--holdout
# draws only from data/holdout/reserved.json + reserved_domains.json, so
# none of these worlds appear in the training corpus) and scores it. L11 and
# L16 are the abort levels; L12-L15, L17, L18 must NOT abort.
#
# Result on the S2 pruned Q8 (first 106 rows): 24/24 correct abstains,
# 82/82 correct actions, 0 false abstains, and all four abort reasons used
# (NOT_FOUND 11 / UNSUPPORTED 6 / AMBIGUOUS 6 / NEEDS_INFO 1) -- against
# 0/73 correct abstains on e_real_sessions. The capability is present and
# well-calibrated; only the trigger phrasing is out of distribution.
set -e
cd /c/code/covenant-agent
M=${1:-baselines/qwen/models/qwen3.5-0.8b-s2-pruned-q8.gguf}
TAG=$(basename "$M" .gguf)
python -m data.gen --levels "11:1,12:1,13:1,14:1,15:1,16:1,17:1,18:1" --n 240 \
    --seed 20260905 --holdout --drop-noops --domains data/gen/themes \
    --out data/holdout/e_s2_levels.jsonl
python -m baselines.qwen.run_a --model "$M" --tasks data/holdout/e_s2_levels.jsonl \
    --ctx 16384 --domains data/gen/themes \
    --out "results/logs/${TAG}_e_s2_levels.jsonl" \
    > "results/logs/${TAG}_e_s2_levels.log" 2>&1
python - "$TAG" <<'PY'
import json, sys, collections
tag = sys.argv[1]
rows = [json.loads(l) for l in open(f"results/logs/{tag}_e_s2_levels.jsonl", encoding="utf-8")]
tasks = {t["id"]: t for t in (json.loads(l) for l in
         open("data/holdout/e_s2_levels.jsonl", encoding="utf-8"))}
ab = [r for r in rows if tasks[r["task_id"]]["expected_status"] == "aborted"]
act = [r for r in rows if tasks[r["task_id"]]["expected_status"] == "ok"]
print(f"n={len(rows)}")
print(f"  should abstain {len(ab):3d}  aborted {sum(r['status']=='aborted' for r in ab):3d}")
print(f"  should act     {len(act):3d}  goal    {sum(r['goal_success'] for r in act):3d}"
      f"  false-abstains {sum(r['status']=='aborted' for r in act)}")
print("  reasons:", dict(collections.Counter(
    r.get("abort_reason") for r in ab if r["status"] == "aborted")))
PY
echo "=== eval_s2_levels done ==="
