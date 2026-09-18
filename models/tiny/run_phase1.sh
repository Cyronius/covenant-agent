#!/usr/bin/env bash
# The whole of phase 1, in one go, on a GPU.
#
# Nothing here should need a decision while the meter is running. Every step
# either works or fails loudly, and the script stops on the first failure so a
# broken run does not quietly consume an hour.
#
#   bash run_phase1.sh            # three seeds, both arms, the sweeps
#   SEEDS=0 bash run_phase1.sh    # one seed, for a first look
#   SKIP_BIG=1 bash run_phase1.sh # skip the capacity check
#
# This is the R3 phase-1 recipe (6 epochs, flat cache). The step-1 run of the
# NPU-native planner sequence is run_step1.sh; it trains whatever binding the
# cache declares and copies the curves out.
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-6}"
BATCH="${BATCH:-64}"
CACHE="${CACHE:-data_cache}"
STEPS_SWEEP="${STEPS_SWEEP:-1 2 4 8 16 32}"
mkdir -p out

echo "=== sanity: does the model run at all on this device ==="
python train.py --arm diffusion --cache "$CACHE" --epochs 1 --limit-train 256 \
  --limit-val 64 --batch "$BATCH" --eval-every 4 --out runs/_sanity
grep -q val_loss runs/_sanity/log.jsonl || { echo "sanity run logged nothing"; exit 1; }
echo "ok -- first step time is in runs/_sanity/log.jsonl; sanity model discarded"
rm -rf runs/_sanity

echo
echo "=== train: both arms, seeds $SEEDS ==="
for s in $SEEDS; do
  for arm in diffusion ar; do
    if [ -f "runs/${arm}_s${s}/best.pt" ]; then
      echo "  ${arm} seed ${s} already trained, skipping"
      continue
    fi
    echo "  --- ${arm} seed ${s} ---"
    python train.py --arm "$arm" --seed "$s" --cache "$CACHE" \
      --epochs "$EPOCHS" --batch "$BATCH" --pad-weight 0.5 \
      --out "runs/${arm}_s${s}"
  done
done

echo
echo "=== generate: the control, once per seed ==="
for s in $SEEDS; do
  python evaluate.py --generate --ckpt "runs/ar_s${s}/best.pt" --cache "$CACHE" \
    --split test --gen-out "out/ar_s${s}_test.jsonl"
  python evaluate.py --generate --ckpt "runs/ar_s${s}/best.pt" --cache "$CACHE" \
    --split holdout --gen-out "out/ar_s${s}_holdout.jsonl"
done

echo
echo "=== generate: diffusion across step counts ==="
# This sweep is the experiment. The control has no equivalent knob, which is
# exactly why results get reported against forward passes rather than steps.
for s in $SEEDS; do
  for k in $STEPS_SWEEP; do
    python evaluate.py --generate --ckpt "runs/diffusion_s${s}/best.pt" \
      --cache "$CACHE" --split test --steps "$k" \
      --gen-out "out/diff_s${s}_k${k}.jsonl"
  done
  python evaluate.py --generate --ckpt "runs/diffusion_s${s}/best.pt" \
    --cache "$CACHE" --split holdout --steps 8 \
    --gen-out "out/diff_s${s}_holdout.jsonl"
done

echo
echo "=== generate: diffusion with the compiler in the loop ==="
# The compiler ships in this bundle, so this runs here rather than waiting for
# the laptop. Compare against the k=8 runs above at equal step count: the
# difference is what the critic bought, and the pass count says what it cost.
for s in $SEEDS; do
  python evaluate.py --generate --ckpt "runs/diffusion_s${s}/best.pt" \
    --cache "$CACHE" --split test --steps 8 --repair-rounds 2 \
    --gen-out "out/diff_s${s}_k8_repair.jsonl"
done

if [ -z "${SKIP_BIG:-}" ]; then
  echo
  echo "=== capacity check: is 8M parameters the ceiling ==="
  # If the two arms swap places between this and the small model, the small
  # result was a capacity artifact and this one is the finding.
  for arm in diffusion ar; do
    python train.py --arm "$arm" --seed 0 --cache "$CACHE" --epochs "$EPOCHS" \
      --batch 32 --d 512 --enc-layers 6 --dec-layers 6 --pad-weight 0.5 \
      --out "runs/${arm}_big"
    python evaluate.py --generate --ckpt "runs/${arm}_big/best.pt" --cache "$CACHE" \
      --split test $([ "$arm" = diffusion ] && echo "--steps 8") \
      --gen-out "out/${arm}_big_test.jsonl"
  done
fi

echo
echo "=== done ==="
echo "bring back:  out/*.jsonl  runs/*/log.jsonl  runs/*/config.json"
echo "score on the laptop:  for f in out/*.jsonl; do python evaluate.py --score --gen-out \$f --split test; done"
ls -la out/ | tail -20
