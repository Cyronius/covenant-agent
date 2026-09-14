# S5 corpus: S4's mix plus the task families and the two 0.5.0/0.6.0 FILTER
# forms (results/FAMILIES.md §6 step 3).
#
#   bash results/logs/gen_s5.sh            # typed  -> data/s5_tasks.jsonl
#   bash results/logs/gen_s5.sh classic    # control -> data/s5c_tasks.jsonl
#
# What changes from S4 (results/logs/gen_s4.sh):
#
#  1. Two recipe gaps from R6 §0.2 are closed in the generator, so this is
#     not only a bigger corpus. L4 argmax now asks a question 40% of the
#     time ("which supplier has the most late deliveries?"), whose answer is
#     a *field* of the winner, so the corpus finally teaches the re-fetch
#     that MOST's id makes necessary - 2,016 of S4's 2,024 MOST rows passed
#     the id straight to a tool and E-known's argmax misses were all this
#     idiom. L11 leaves the record's name unextracted 40% of the time, which
#     is the L11 exam's condition: no constant to filter on, no referent to
#     name, so the bare ABORT NOT_FOUND is the only faithful program (S4:
#     195 bare against 2,597 checked, and the model reached for AMBIGUOUS F4).
#  2. L19 teaches spec 0.6.0's IN: every parent with (or without) a child
#     matching a predicate, as a set. The FOREACH-over-children form acts
#     once per child, so a parent with three matching children is messaged
#     three times.
#  3. The families (A-E) ride the mix rather than a separate retrain - see
#     finish_s5_corpus.sh. Generate them first:
#       bash results/logs/gen_families.sh typed
#
# Not here: the 0.5.0 field-vs-field clause has no recipe, because no theme
# world can host one. Every generated theme entity is STR/BOOL/TIME/ID with
# a single event time and a single creation time, and "departed before it
# was planned" is not a request anyone makes. It needs a second comparable
# field on the child (a due/actual pair), which is a theme-surface change.
#
# Same shard layout and volume as S4; new seeds. Run from the repo root in
# Git Bash. Train with the S4 pod script, pointed at this corpus:
#   CORPUS_TAG=s5 bash train_s4.sh typed
set -e
cd /c/code/covenant-agent
ARM="${1:-typed}"
case "$ARM" in
  typed)   SURFACE="--symbols typed --enums --kinds"; TAG=s5 ;;
  classic) SURFACE="";                                TAG=s5c ;;
  *) echo "usage: gen_s5.sh [typed|classic]"; exit 2 ;;
esac
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
mkdir -p data/${TAG}_shards
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260912 + i*10000000)) --drop-noops \
      $SURFACE --domains data/gen/themes --out data/${TAG}_shards/plain_$i.jsonl > results/logs/gen_${TAG}_plain_$i.log 2>&1 &
done
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 3250 --seed $((20260913 + i*10000000)) --drop-noops \
      $SURFACE --domains data/gen/themes --crowd 15:60 --out data/${TAG}_shards/crowded_$i.jsonl > results/logs/gen_${TAG}_crowded_$i.log 2>&1 &
done
wait
cat data/${TAG}_shards/plain_*.jsonl > data/${TAG}_plain.jsonl
cat data/${TAG}_shards/crowded_*.jsonl > data/${TAG}_crowded.jsonl
cat data/${TAG}_plain.jsonl data/${TAG}_crowded.jsonl > data/${TAG}_tasks.jsonl
wc -l data/${TAG}_plain.jsonl data/${TAG}_crowded.jsonl data/${TAG}_tasks.jsonl
python -m baselines.qwen.make_sft --tasks data/${TAG}_tasks.jsonl --domains data/gen/themes \
    --symbols "$ARM" --out data/sft_${TAG}_gen.jsonl

bash results/logs/finish_s5_corpus.sh "$ARM"
echo "=== gen_s5 ($ARM) done ==="
