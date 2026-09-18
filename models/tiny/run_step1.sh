#!/usr/bin/env bash
# Step 1 of the NPU-native planner sequence: structural binding on the R3 model.
# `.claude/plans/npu-native-planner.md`, "Sequence", step 1.
#
# Same data as R3 (s5_plain, every usable single-segment task, the 90/5/5 split
# and one held-out world), same scorer (harness/run.py on the laptop), both
# arms, three seeds. The only thing that differs from R3's runs is the binding,
# which prep.py fixed when it wrote the cache: this script trains whatever the
# cache declares, so a flat cache reproduces the R3 baseline through the same
# script.
#
# Gate: tool-slot accuracy (acc_tool in runs/*/log.jsonl, chance about 6%)
# leaves chance, and compile rate (scored on the laptop) leaves single digits.
#
#   bash run_step1.sh                # three seeds, both arms, the step sweep
#   SEEDS=0 bash run_step1.sh        # one seed, for a first look
#   CACHE=data_cache_flat bash run_step1.sh   # the R3 baseline, same recipe
#
# Nothing here should need a decision while the meter is running. Every step
# either works or fails loudly, and the script stops on the first failure.
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-12}"
BATCH="${BATCH:-64}"
CACHE="${CACHE:-data_cache_struct}"
STEPS_SWEEP="${STEPS_SWEEP:-1 2 4 8 16 32}"
DEC_LOOPS="${DEC_LOOPS:-1}"
TAG="${TAG:-s1}"          # prefix on run and output names
mkdir -p out

BINDING=$(python -c "import json,sys; print(json.load(open(sys.argv[1]+'/config.json')).get('binding','flat'))" "$CACHE")
echo "cache $CACHE  binding $BINDING  seeds $SEEDS  epochs $EPOCHS  batch $BATCH  dec_loops $DEC_LOOPS"

echo "=== sanity: does the model run at all on this device ==="
python train.py --arm diffusion --cache "$CACHE" --epochs 1 --limit-train 256 \
  --limit-val 64 --batch "$BATCH" --eval-every 4 --out runs/_sanity
grep -q '"acc_tool"' runs/_sanity/log.jsonl || { echo "sanity run logged no per-slot accuracy"; exit 1; }
echo "ok -- first step time is in runs/_sanity/log.jsonl; sanity model discarded"
rm -rf runs/_sanity

echo
echo "=== train: both arms, seeds $SEEDS ==="
# 12 epochs, twice R3's run 1: its control converged under 6 while still at
# chance on symbols, and the diffusion arm was still improving.
for s in $SEEDS; do
  for arm in diffusion ar; do
    run="runs/${TAG}_${arm}_s${s}"
    if [ -f "$run/best.pt" ]; then
      echo "  ${arm} seed ${s} already trained, skipping"
      continue
    fi
    echo "  --- ${arm} seed ${s} ---"
    python train.py --arm "$arm" --seed "$s" --cache "$CACHE" \
      --epochs "$EPOCHS" --batch "$BATCH" --pad-weight 0.5 --dec-loops "$DEC_LOOPS" \
      --out "$run"
  done
done

echo
echo "=== generate: the control, once per seed ==="
for s in $SEEDS; do
  python evaluate.py --generate --ckpt "runs/${TAG}_ar_s${s}/best.pt" --cache "$CACHE" \
    --split test --gen-out "out/${TAG}_ar_s${s}_test.jsonl"
  python evaluate.py --generate --ckpt "runs/${TAG}_ar_s${s}/best.pt" --cache "$CACHE" \
    --split holdout --gen-out "out/${TAG}_ar_s${s}_holdout.jsonl"
done

echo
echo "=== generate: diffusion across step counts ==="
for s in $SEEDS; do
  for k in $STEPS_SWEEP; do
    python evaluate.py --generate --ckpt "runs/${TAG}_diffusion_s${s}/best.pt" \
      --cache "$CACHE" --split test --steps "$k" \
      --gen-out "out/${TAG}_diff_s${s}_k${k}.jsonl"
  done
  python evaluate.py --generate --ckpt "runs/${TAG}_diffusion_s${s}/best.pt" \
    --cache "$CACHE" --split holdout --steps 8 \
    --gen-out "out/${TAG}_diff_s${s}_holdout.jsonl"
done

echo
echo "=== generate: diffusion with the compiler in the loop ==="
# The compiler ships in this bundle, so this runs here rather than waiting for
# the laptop. Compare against the k=8 runs above at equal step count.
for s in $SEEDS; do
  python evaluate.py --generate --ckpt "runs/${TAG}_diffusion_s${s}/best.pt" \
    --cache "$CACHE" --split test --steps 8 --repair-rounds 2 \
    --gen-out "out/${TAG}_diff_s${s}_k8_repair.jsonl"
done

echo
echo "=== curves: copy every log beside the generations so nothing is left behind ==="
mkdir -p out/curves
for d in runs/${TAG}_*/; do
  n=$(basename "$d")
  mkdir -p "out/curves/$n"
  cp "$d/log.jsonl" "$d/config.json" "out/curves/$n/"
done

echo
echo "=== done ==="
echo "bring back:  out/  (generations, and curves/ with every log.jsonl and config.json)"
echo "score on the laptop:"
echo "  for f in out/${TAG}_*_test.jsonl out/${TAG}_*_k*.jsonl; do python evaluate.py --score --cache $CACHE --gen-out \$f --split test; done"
echo "  for f in out/${TAG}_*_holdout.jsonl; do python evaluate.py --score --cache $CACHE --gen-out \$f --split holdout; done"
echo "  python probe.py --gen out/${TAG}_diff_s0_k8.jsonl --compiled-only"
echo "  python diagnose.py --gen out/${TAG}_diff_s0_k8.jsonl --cache $CACHE"
ls -la out/ | tail -20
