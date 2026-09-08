# S4 corpus: the first corpus in the spec 0.4.0 surface
# (.claude/plans/spec-0.4.0.md §6 step 2/3; the A/B that authorised it is
# results/R5.md).
#
#   bash results/logs/gen_s4.sh            # typed  -> data/s4_tasks.jsonl
#   bash results/logs/gen_s4.sh classic    # control -> data/s4c_tasks.jsonl
#
# What changes from S3 (results/logs/gen_s3.sh):
#
#  1. Named composites, in BOTH arms. L4 argmax is one `MOST` instruction
#     instead of the 14-line COUNT/FOREACH/IF/LET loop, and a sibling argmin
#     recipe emits `LEAST` with the candidate list (the answer to "the
#     fewest" is usually someone absent from the filtered list). The "did
#     anything match" branch -- the NOT_FOUND guard at L11 and the L7
#     parallel-count variant -- is `IF [NOT] EMPTY r`, not COUNT against a
#     zero. R5: +12 of 130 with reasoning on the untuned 27B, 10 gains and
#     0 losses on crowded contexts.
#  2. Typed constant letters (S/N/B/D/I), string kinds, and the schema's
#     enum values as constants -- the typed arm only. R5 could not decide
#     these on the 27B (neutral with reasoning, negative without), and
#     typed-symbols.md §3 claim 2 says only a retrain can: the letters make
#     binding legible and remove a checker safety net, so the corpus has to
#     teach what the typechecker no longer catches. The classic arm is the
#     identical-task control -- same seeds, same programs, same English,
#     only the symbol table differs -- so the paired 0.8B retrains isolate
#     the letters from everything else that changed since S3.
#  3. The real-turn rows are rebuilt in whichever surface the arm uses. A
#     corpus that mixes C0 and S0 teaches neither; make_sft refuses to mix.
#
# Same level weights, same volume, same shard layout as S3; new seeds.
# Run from the repo root in Git Bash.
set -e
cd /c/code/covenant-agent
ARM="${1:-typed}"
case "$ARM" in
  typed)   SURFACE="--symbols typed --enums --kinds"; TAG=s4 ;;
  classic) SURFACE="";                                TAG=s4c ;;
  *) echo "usage: gen_s4.sh [typed|classic]"; exit 2 ;;
esac
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8"
mkdir -p data/${TAG}_shards
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260908 + i*10000000)) --drop-noops \
      $SURFACE --domains data/gen/themes --out data/${TAG}_shards/plain_$i.jsonl > results/logs/gen_${TAG}_plain_$i.log 2>&1 &
done
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 3250 --seed $((20260909 + i*10000000)) --drop-noops \
      $SURFACE --domains data/gen/themes --crowd 15:60 --out data/${TAG}_shards/crowded_$i.jsonl > results/logs/gen_${TAG}_crowded_$i.log 2>&1 &
done
wait
cat data/${TAG}_shards/plain_*.jsonl > data/${TAG}_plain.jsonl
cat data/${TAG}_shards/crowded_*.jsonl > data/${TAG}_crowded.jsonl
cat data/${TAG}_plain.jsonl data/${TAG}_crowded.jsonl > data/${TAG}_tasks.jsonl
wc -l data/${TAG}_plain.jsonl data/${TAG}_crowded.jsonl data/${TAG}_tasks.jsonl
python -m baselines.qwen.make_sft --tasks data/${TAG}_tasks.jsonl --domains data/gen/themes \
    --symbols "$ARM" --out data/sft_${TAG}_gen.jsonl

bash results/logs/finish_s4_corpus.sh "$ARM"
echo "=== gen_s4 ($ARM) done ==="
