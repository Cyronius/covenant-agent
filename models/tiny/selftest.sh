#!/usr/bin/env bash
# Correctness only. Deliberately measures nothing.
#
# This machine cannot be trusted for timings -- see the README -- so the laptop's
# job is to answer "is any of this broken", which it can do reliably with tiny
# models on a handful of rows. Every performance question goes to the pod.
#
#   bash selftest.sh [cache_dir]          # default data_cache_struct (structural)
#   bash selftest.sh data_cache_flat      # the same checks on a flat cache
#
# The binding is read from the cache; every step below runs under either.
set -euo pipefail
CACHE="${1:-data_cache_struct}"
BINDING=$(python -c "import json,sys; print(json.load(open(sys.argv[1]+'/config.json')).get('binding','flat'))" "$CACHE")
echo "cache $CACHE  binding $BINDING"

echo "=== 1. do reference programs survive the round trip and reach the goal ==="
# The check everything else rests on. If this fails, no model number means
# anything, because the data path itself is wrong. Under structural binding
# this is the canvas of keywords and pointers, with r0.F6 as two slots,
# rendered back to text and executed.
python test_pipeline.py --cache "$CACHE" --split test --n 40
python test_pipeline.py --cache "$CACHE" --split holdout --n 20

echo
echo "=== 2. can both samplers reproduce a known answer ==="
# A sampler bug would not show up in training loss. It would show up as one arm
# losing the comparison, which is a confident wrong result rather than an error.
# Also checks that the receiver rule (local_mask) never blocks a reference.
python test_samplers.py --cache "$CACHE" --n 20

echo
echo "=== 3. does each arm train, sample and score end to end ==="
# Tiny and brief. The loss value is meaningless at this size; the point is that
# nothing raises, the plumbing connects, and the log carries per-slot-kind
# accuracy (acc_kw acc_tool acc_field acc_const acc_reg).
for arm in diffusion ar; do
  python train.py --arm "$arm" --cache "$CACHE" --epochs 1 --limit-train 32 \
    --limit-val 16 --batch 4 --d 64 --enc-layers 1 --dec-layers 1 \
    --eval-every 8 --out "runs/_selftest_$arm"
  grep -q '"acc_tool"' "runs/_selftest_$arm/log.jsonl" || { echo "log has no per-slot accuracy"; exit 1; }
  extra=""
  [ "$arm" = diffusion ] && extra="--steps 2 --repair-rounds 1"
  python evaluate.py --ckpt "runs/_selftest_$arm/best.pt" --cache "$CACHE" \
    --split val --limit 4 $extra --gen-out "runs/_selftest_$arm.jsonl"
done

echo
echo "=== 4. is the quantised model the model that would be deployed ==="
# Quantisation-aware training fails silently in both directions: unreached fake
# quantisation reports float accuracy for weights about to be rounded, and a
# broken estimator trains nothing while looking undertrained.
python test_quant.py

echo
echo "=== 5. does a looped, ternary run train, generate and score ==="
# The two knobs steps 2 and 3 sweep, exercised together on a handful of rows.
# Catches the failures that would otherwise surface an hour into a paid run: a
# parametrized checkpoint that will not reload, a loop count that changes a
# tensor shape, an inference-time loop override the sampler ignores.
python train.py --arm diffusion --cache "$CACHE" --epochs 1 --limit-train 32 \
  --limit-val 16 --batch 4 --d 64 --enc-layers 1 --dec-layers 1 \
  --dec-loops 4 --loop-emb --rand-loops 4 --weights tern \
  --eval-every 8 --out runs/_selftest_q
grep -q '"acc_tool"' runs/_selftest_q/log.jsonl || { echo "log has no per-slot accuracy"; exit 1; }
python -c "import json; c = json.load(open('runs/_selftest_q/config.json')); assert c['weights'] == 'tern' and c['resident_bytes'] > 0, c; print('  block', c['loop_body_params'], 'params ->', c['resident_bytes'], 'resident bytes')"
python evaluate.py --ckpt runs/_selftest_q/best.pt --cache "$CACHE" --split val \
  --limit 4 --steps 2 --dec-loops 2 --gen-out runs/_selftest_q.jsonl
python -c "import json; r = [json.loads(l) for l in open('runs/_selftest_q.jsonl', encoding='utf-8')]; assert all(x['loops'] == 2 for x in r), 'the inference-time loop override was ignored'; print('  generation ran at the overridden loop count')"

echo
echo "=== 6. does the fill-order probe read a generated file ==="
python probe.py --gen runs/_selftest_diffusion.jsonl --cache "$CACHE"

rm -rf runs/_selftest_diffusion runs/_selftest_ar runs/_selftest_q \
       runs/_selftest_diffusion.jsonl runs/_selftest_ar.jsonl runs/_selftest_q.jsonl \
       runs/_selftest_diffusion.score.json runs/_selftest_ar.score.json \
       runs/_selftest_q.score.json
echo
echo "ALL PASS -- correctness only, no timing claimed"
