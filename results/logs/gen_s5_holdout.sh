# S5-holdout: the unseen-world eval corpus, generated over the 42 RESERVED
# eval worlds instead of carved out of the training pool.
#
#   bash results/logs/gen_s5_holdout.sh                 # -> data/s5_holdout.jsonl
#   bash results/logs/gen_s5_holdout.sh 42              # also data/s5_holdout_inject.jsonl
#   bash results/logs/gen_s5_holdout.sh 0 classic       # the classic-symbol control
#   bash results/logs/gen_s5_holdout.sh 0 typed 1:2     # -> data/s5_holdout_decoy.jsonl
#
# Why this and not `prep.py --holdout-worlds N`. data/holdout/reserved.json
# and data/holdout/reserved_domains.json reserve 46 names, 42 of which the
# generator can build (against 104 training worlds), and `data.gen --holdout`
# has drawn from them since S2. They are generated themes like the training
# ones: full ID:entity typing, the same program recipes, the same English
# templates. Nothing has ever trained on them.
#
# R7 and R8 measured unseen-world generalisation by holding out `bookstore`
# -- one world, 275 rows, 0.96% of the training data, carved from inside the
# training pool. The 74-95% control spread across three seeds that motivates
# the injection plan is therefore an n=1-world measurement, and an unlucky
# world and a seed-unstable encoder are indistinguishable in it. Forty-two
# worlds costs zero training data, needs no split change, and may settle that
# question on its own -- before any injection arm is run.
#
# Prep it as a second cache against the unchanged training corpus. Pass the
# CONCATENATION of the plain and decoyed holdouts, so one training run scores
# both and their difference is the grounding measurement (models/tiny/README):
#
#   cd models/tiny
#   python prep.py --corpus ../../data/s5_plain.jsonl --limit 30000 \
#       --max-line 80 --holdout-corpus ../../data/s5_holdout_both.jsonl --out data_cache_ho42
#   bash selftest.sh data_cache_ho42
#
# prep refuses a --holdout-corpus that shares a world with --corpus, so the
# "unseen" claim is checked rather than assumed.
#
# data_cache_ho42_plain / pod_bundle_ho42_plain.tar.gz are the earlier
# plain-holdout-only cache, kept because it is the one whose layout matches
# R8's (max_tool 18, 145 joint ids). The combined cache is max_tool 50 and 177.
#
# --max-line 80, and this is a finding rather than a formality: prep's default
# 64 was fitted to the 104 TRAINING worlds, whose longest schema line is 56
# tokens, and the input BPE is trained on training lines only. Unseen-world
# vocabulary therefore splits into more pieces -- notary_office has a 67-token
# tool line -- so the default cap rejects the reserved corpus outright, with
# no injection involved. Both arms read the same cache, so the comparison is
# unaffected; only the absolute number moves against an R8 cache built at 64.
#
# The injected variant (second argument > 0) draws its distractors from the
# HOLDOUT side of the imported pool, so a held-out world's distractors are
# unseen too (data/gen/open_pool.py).
#
# The DECOYED variant (third argument, e.g. 1:2) writes to a `_decoy` suffix
# rather than over the plain files, and the two are not interchangeable.
# Undecoyed, the typed signature identifies the called tool in 100% of
# reference CALLs (results/GROUNDING.md), so a model that never reads a
# description still has a unique shape match for every call; that suite
# cannot carry a grounding claim. Decoyed at 1:2, with decoys covering READ
# and EXTERNAL as well as the mutating effects, it lands at 2.6% full
# uniqueness and 41.5 tools per task.
#
# 1:2 and not higher: the rate is not the knob. 1:2 / 2:4 / 3:6 gave
# 59.2 / 59.2 / 60.0 back when decoys were mutating-only, while tools per task
# went 30.7 -> 47.4 (GROUNDING.md section 5). What moved the number was
# covering READ, which is 48.9% of all reference calls and was 100% unique on
# its own -- section 8.
#
# The figure to check against a scored run is not `full` but `chance_tool_sig`
# (models/tiny/diagnose.py): the score a model gets by inferring the type
# shape and picking uniformly inside the signature collision group, reading no
# description at all. Plain 100.0%, mutating-only decoys 73.4%, READ+EXTERNAL
# decoys 43.5%. `same_tool` below that line is not evidence of grounding.
#
# Keep BOTH. The plain holdout is continuous with the bookstore measurement
# R3/R7/R8 reported, so it is what R8's 74-95% control spread compares
# against; the decoyed one is the only variant that poses the discrimination.
# Reading a decoyed number as if it were the plain one attributes the decoy
# drop to the unseen worlds.
set -e
cd /c/code/covenant-agent
N_INJECT="${1:-0}"
ARM="${2:-typed}"
DECOYS="${3:-}"
case "$ARM" in
  typed)   SURFACE="--symbols typed --enums --kinds"; TAG=s5 ;;
  classic) SURFACE="";                                TAG=s5c ;;
  *) echo "usage: gen_s5_holdout.sh [N_INJECT] [typed|classic] [MIN:MAX decoys]"; exit 2 ;;
esac
if [ -n "$DECOYS" ]; then
  DECOY_FLAG="--decoys $DECOYS"
  SUF=_decoy
else
  # 1f, 2026-09-21: predates the mandatory collision ceiling; the escape
  # reproduces this plain (undecoyed) record exactly and stamps the fact
  # into every row's provenance.
  DECOY_FLAG="--decoys 0 --allow-signature-unique"
  SUF=""
fi
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8,19:5"
mkdir -p data/${TAG}_shards

for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 1000 --seed $((20260919 + i*10000000)) --drop-noops \
      --holdout $SURFACE --domains data/gen/themes $DECOY_FLAG \
      --out data/${TAG}_shards/holdout${SUF}_$i.jsonl > results/logs/gen_${TAG}_holdout${SUF}_$i.log 2>&1 &
done
wait
# [0-7], not *: a plain re-run after an injected one would otherwise fold the
# inject shards into the plain corpus
cat data/${TAG}_shards/holdout${SUF}_[0-7].jsonl > data/${TAG}_holdout${SUF}.jsonl
wc -l data/${TAG}_holdout${SUF}.jsonl
python -c "import json,sys;w={json.loads(l)['world'] for l in open(sys.argv[1],encoding='utf-8')};print(f'  {len(w)} unseen worlds')" data/${TAG}_holdout${SUF}.jsonl
python -m harness.signature_uniqueness data/${TAG}_holdout${SUF}.jsonl

if [ "$N_INJECT" -gt 0 ]; then
  for i in 0 1 2 3 4 5 6 7; do
    python -m data.gen --levels "$LEVELS" --n 1000 --seed $((20260919 + i*10000000)) --drop-noops \
        --holdout $SURFACE --domains data/gen/themes --inject-open ${N_INJECT}:${N_INJECT} $DECOY_FLAG \
        --out data/${TAG}_shards/holdout${SUF}_inject_$i.jsonl > results/logs/gen_${TAG}_holdout${SUF}_inject_$i.log 2>&1 &
  done
  wait
  cat data/${TAG}_shards/holdout${SUF}_inject_[0-7].jsonl > data/${TAG}_holdout${SUF}_inject.jsonl
  wc -l data/${TAG}_holdout${SUF}_inject.jsonl
  python -m harness.signature_uniqueness data/${TAG}_holdout${SUF}_inject.jsonl
fi
echo "=== gen_s5_holdout ($ARM, inject $N_INJECT, decoys ${DECOYS:-none}) done ==="
