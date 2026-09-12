# E-rpg on the S3 and S4 checkpoints, each on the surface it was trained on
# (results/R6.md §0: a typed model scored on classic prompts looks broken).
# Unpruned GGUFs only - RPG.md excludes the pruned ones, whose vocabulary trim
# never saw the map glyphs or words like "goblin".
set -u
cd "$(dirname "$0")/../.."
EP=${1:-3}; TURNS=${2:-20}; TH=${THREADS:-16}
M=baselines/qwen/models
python -m harness.rpg_suite --planner oracle --episodes "$EP" --max-turns "$TURNS" \
  --out results/logs/oracle_e_rpg.jsonl 2>&1 | tail -12
python -m harness.rpg_suite --model $M/qwen3.5-0.8b-s4-q8-fixed.gguf --template qwen \
  --symbols typed --enums --kinds --episodes "$EP" --max-turns "$TURNS" --threads "$TH" \
  --out results/logs/qwen3.5-0.8b-s4-q8-fixed_e_rpg.jsonl 2>&1 \
  | tee results/logs/qwen3.5-0.8b-s4-q8-fixed_e_rpg.log | tail -14
python -m harness.rpg_suite --model $M/qwen3.5-0.8b-s3-q8.gguf --template qwen \
  --episodes "$EP" --max-turns "$TURNS" --threads "$TH" \
  --out results/logs/qwen3.5-0.8b-s3-q8_e_rpg.jsonl 2>&1 \
  | tee results/logs/qwen3.5-0.8b-s3-q8_e_rpg.log | tail -14
echo "=== rpg_s3_s4 done ==="
