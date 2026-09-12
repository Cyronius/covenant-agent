# Each family's own exam, for one model (.claude/plans/archive/task-families.md §5).
#
#   bash results/logs/eval_families.sh baselines/qwen/models/<model>.gguf [typed]
#
# A regression here is attributable to a family, which is what the classic
# control problem in R6 §0 cost us last time. The six suites in R6 §0.1 are
# the other half of the bar and are not repeated here - run eval_esuites.sh
# and eval_s2.sh for those.
#
# Held out throughout: the dungeon and the house (family A), the
# coursebuilder-shaped app (family C), the reserved domains (B, D, E).
set -e
cd /c/code/covenant-agent
M=${1:-baselines/qwen/models/qwen3.5-0.8b-s4-pruned-q8-fixed.gguf}
ARM=${2:-classic}
TAG=$(basename "$M" .gguf)
case "$ARM" in
  typed)   SURFACE="--symbols typed --enums --kinds" ;;
  classic) SURFACE="" ;;
  *) echo "usage: eval_families.sh <gguf> [typed|classic]"; exit 2 ;;
esac
SFX=""; [ "$ARM" = typed ] && SFX="_typed"
# --- families A and C: episodes, scored per turn against the oracle -------
for W in rpg house app_coursebuilder; do
  python -m harness.rpg_suite --world $W --model "$M" --episodes 6 \
      $SURFACE --quiet --out "results/logs/${TAG}_e_${W}.jsonl" \
      > "results/logs/${TAG}_e_${W}.log" 2>&1
  echo "--- $W"
  tail -12 "results/logs/${TAG}_e_${W}.log"
done

# --- family B: the gap between the paired suites is the number -----------
run() {
  python -m baselines.qwen.run_a --model "$M" --tasks "$1" --ctx "$2" $SURFACE \
      --domains data/gen/themes \
      --out "results/logs/${TAG}_$(basename "$1" .jsonl).jsonl" \
      > "results/logs/${TAG}_$(basename "$1" .jsonl).log" 2>&1
  echo "$(basename "$1" .jsonl): $(tail -c 220 "results/logs/${TAG}_$(basename "$1" .jsonl).log" | tr '\n' ' ')"
}
run data/holdout/e_known_plain${SFX}.jsonl 16384
run data/holdout/e_known_decoy${SFX}.jsonl 24576
run data/holdout/e_askact${SFX}.jsonl 8192
run data/holdout/e_recovery${SFX}.jsonl 8192
echo
echo "family B's number is e_known_plain minus e_known_decoy on the same tasks"
echo "=== eval_families done ==="
