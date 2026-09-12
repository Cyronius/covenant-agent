#!/usr/bin/env bash
# E-rpg: play the held-out grid dungeon with each candidate checkpoint, same
# scenario, same seeds, same turn cap. Sequential on purpose — these are
# CPU-bound llama.cpp runs and overlapping them only makes both slower and
# the latency columns meaningless.
#
#   bash results/logs/eval_rpg.sh [episodes] [max_turns]
#
# Notes on the model set (plan .claude/plans/rpg-demo-app.md):
#  - the *pruned* checkpoints are deliberately excluded: their vocabulary was
#    trimmed to the SFT corpus plus a general-English floor, so map glyphs and
#    words like "goblin" fall back to segmentations they never saw. Judging
#    them on this world would measure the trim, not the planner.
#  - our tuned checkpoints use the hand-rolled ChatML markup they were trained
#    on (--template qwen); anything else gets its own chat template
#    (--template chat), which is the only fair way to prompt it.
#  - S4 was trained on the spec 0.4.0 surface, so its arm passes
#    --symbols typed --enums --kinds. Every arm has to be run on the surface
#    its checkpoint was trained on: scoring a typed model on classic prompts
#    makes it look broadly broken (results/R6.md §0).
set -u
cd "$(dirname "$0")/../.."

EPISODES=${1:-3}
TURNS=${2:-20}
THREADS=${THREADS:-6}
M=baselines/qwen/models

run () {  # run <gguf> <template> [surface flags...]
  local model="$1" template="$2"
  shift 2
  local tag
  tag=$(basename "$model" .gguf)
  echo "=== $tag ($template ${*:-classic}) ==="
  python -m harness.rpg_suite --model "$model" --template "$template" "$@" \
    --episodes "$EPISODES" --max-turns "$TURNS" --threads "$THREADS" \
    --out "results/logs/${tag}_e_rpg.jsonl" \
    2>&1 | tee "results/logs/${tag}_e_rpg.log"
}

echo "=== oracle (ceiling) ==="
python -m harness.rpg_suite --planner oracle --episodes "$EPISODES" \
  --max-turns "$TURNS" --out results/logs/oracle_e_rpg.jsonl \
  2>&1 | tee results/logs/oracle_e_rpg.log

run "$M/qwen3.5-0.8b-s4-q8-fixed.gguf" qwen --symbols typed --enums --kinds
run "$M/qwen3.5-0.8b-s3-q8.gguf"  qwen    # last classic-surface planner
run "$M/qwen3.5-0.8b-s2r-q8.gguf" qwen
run "$M/qwen3.5-0.8b-s2-q8.gguf"  qwen
run "$M/Qwen3.5-0.8B-Q8_0.gguf"   chat    # same size, untuned
run "$M/Qwen3.5-2B-Q8_0.gguf"     chat    # bigger, untuned

echo
echo "=== summary ==="
for f in results/logs/*_e_rpg.jsonl; do
  python -m harness.rpg_suite --report "$f"
done
