#!/usr/bin/env bash
# Step 2 of the NPU-native planner sequence: the loop-count sweep.
# `.claude/plans/npu-native-planner.md`, "Sequence", step 2, and bet 2.
#
# The question: at a fixed parameter count, does applying the block more times
# write better programs? On the NPU the extra applications are free -- the
# weights are already on chip and a reapplication costs no dispatch and no DMA
# -- so if accuracy rises with L, compute is available that costs nothing, and
# the design's premise holds. If it does not, the loop is decorative and the
# pipeline needs parameters instead, which the memory tiles cap at 16M ternary.
#
# Three lines get drawn, and confusing them answers the wrong question:
#
#   looped      one block of $DL distinct layers, applied L times. Identical
#               parameters at every L (test_quant.py checks that), only compute
#               grows. This is the gate.
#   unlooped    L=1 with more distinct layers ($LADDER). Compute grows the same
#               way and parameters grow with it. Looping has to come close to
#               this for depth to be substituting for parameters.
#   dial        one checkpoint trained at a loop count sampled per batch
#               (--rand-loops), run at every setting. Decision 9 calls L "the
#               effort dial", which only holds if one set of weights serves
#               several L; a model trained at a fixed L has never seen another.
#
# Why the base block is one layer rather than step 1's four: step 1's model is
# saturated. It reaches 98.6% goal on test and 96.4% on the held-out world, so a
# sweep on it would measure the ceiling, not the loop. A one-layer block has
# room to show the effect, and it is also the shape the NPU wants -- the smaller
# the block, the more loops fit in the same weight budget.
#
#   bash run_step2.sh                     # everything, cheapest first
#   SEEDS=0 bash run_step2.sh             # one seed for a first look
#   STAGES=A bash run_step2.sh            # the gate only
#   LOOPS="1 8" SEEDS=0 bash run_step2.sh # a two-point smoke test
#
# Every run skips itself if its checkpoint exists, so an interrupted sweep
# resumes. Ordering is cheapest first: the L=32 runs cost roughly eight times an
# L=1 run, so a sweep that is cut short still has the small end complete.
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
LOOPS="${LOOPS:-1 2 4 8 16 32}"
LADDER="${LADDER:-2 4 8}"          # unlooped decoder depths, the parameter line
DL="${DL:-1}"                      # distinct layers in the looped block
BEST_L="${BEST_L:-8}"              # where stage C puts its side conditions
DIAL="${DIAL:-1 2 4 8 16}"         # loop counts the --rand-loops checkpoint is run at
EPOCHS="${EPOCHS:-12}"
BATCH="${BATCH:-64}"
K="${K:-8}"                        # denoising steps at generation
CACHE="${CACHE:-data_cache_struct}"
STAGES="${STAGES:-A B C}"
TAG="${TAG:-s2}"
# Smoke-testing this script costs one line rather than a pod: EXTRA goes to
# train.py and GEN_EXTRA to evaluate.py, so the whole sweep runs tiny and local.
#   EPOCHS=1 SEEDS=0 CACHE=data_cache_smoke LOOPS="1 2" LADDER=2 DIAL="1 2" \
#     GEN_EXTRA="--limit 4" TAG=_smoke2 \
#     EXTRA="--limit-train 32 --limit-val 16 --batch 4 --d 64 --enc-layers 1" \
#     bash run_step2.sh
EXTRA="${EXTRA:-}"
GEN_EXTRA="${GEN_EXTRA:-}"
mkdir -p out

stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

BINDING=$(python -c "import json,sys; print(json.load(open(sys.argv[1]+'/config.json')).get('binding','flat'))" "$CACHE")
[ "$BINDING" = structural ] || { echo "step 2 is a structural-binding sweep; $CACHE is $BINDING"; exit 1; }
echo "cache $CACHE  seeds $SEEDS  loops $LOOPS  block ${DL}L  epochs $EPOCHS  stages $STAGES"

# train <run-dir> <extra args...>
train() {
  local run="$1"; shift
  if [ -f "$run/best.pt" ]; then echo "  $run already trained, skipping"; return 0; fi
  echo "  --- $run ---"
  python train.py --cache "$CACHE" --epochs "$EPOCHS" --batch "$BATCH" \
    --pad-weight 0.5 "$@" $EXTRA --out "$run"
}

# gen <run-dir> <out-name> <extra args...>
gen() {
  local run="$1" name="$2"; shift 2
  if [ -f "out/${name}.jsonl" ]; then echo "  out/${name}.jsonl exists, skipping"; return 0; fi
  python evaluate.py --generate --ckpt "$run/best.pt" --cache "$CACHE" \
    --split test --steps "$K" "$@" $GEN_EXTRA --gen-out "out/${name}.jsonl"
}

if stage A; then
  echo
  echo "=== A. the gate: a ${DL}-layer block applied L times, fixed parameters ==="
  for L in $LOOPS; do
    for s in $SEEDS; do
      run="runs/${TAG}_diffusion_s${s}_dl${DL}_L${L}"
      train "$run" --arm diffusion --seed "$s" --dec-layers "$DL" --dec-loops "$L"
      gen "$run" "${TAG}_diff_s${s}_dl${DL}_L${L}_k${K}"
    done
  done
  # One seed on the held-out world at each end of the sweep: if looping only
  # helps on worlds the model trained on, it is memorising, not computing.
  for L in $LOOPS; do
    run="runs/${TAG}_diffusion_s0_dl${DL}_L${L}"
    [ -f "$run/best.pt" ] || continue
    name="${TAG}_diff_s0_dl${DL}_L${L}_holdout"
    [ -f "out/${name}.jsonl" ] && continue
    python evaluate.py --generate --ckpt "$run/best.pt" --cache "$CACHE" \
      --split holdout --steps "$K" $GEN_EXTRA --gen-out "out/${name}.jsonl"
  done
fi

if stage B; then
  echo
  echo "=== B. the parameter line: unlooped depth, and the control arm ==="
  for dl in $LADDER; do
    run="runs/${TAG}_diffusion_s0_dl${dl}_L1"
    train "$run" --arm diffusion --seed 0 --dec-layers "$dl" --dec-loops 1
    gen "$run" "${TAG}_diff_s0_dl${dl}_L1_k${K}"
  done
  # The control shares the block, so the same question applies to it: step 1
  # found it ahead of the diffusion arm by 10 points, and whether looping closes
  # that gap decides which arm the NPU pipeline should carry.
  for L in 1 "$BEST_L"; do
    run="runs/${TAG}_ar_s0_dl${DL}_L${L}"
    train "$run" --arm ar --seed 0 --dec-layers "$DL" --dec-loops "$L"
    # No _k in the name: the control spends one pass per token, so a denoising
    # step count would be meaningless there, and report.py reads it as one.
    gen "$run" "${TAG}_ar_s0_dl${DL}_L${L}_test"
  done
fi

if stage C; then
  echo
  echo "=== C. side conditions at L=${BEST_L}: the loop index, and the dial ==="
  # A learned bias per iteration. Free on the NPU (one add of a resident vector)
  # and 8k parameters, so if it helps at high L it should simply be on.
  run="runs/${TAG}_diffusion_s0_dl${DL}_L${BEST_L}_le"
  train "$run" --arm diffusion --seed 0 --dec-layers "$DL" --dec-loops "$BEST_L" --loop-emb
  gen "$run" "${TAG}_diff_s0_dl${DL}_L${BEST_L}_le_k${K}"

  # One checkpoint, every effort setting. Trained at a sampled loop count, then
  # run at each: the spread across DIAL is how far the dial actually travels.
  run="runs/${TAG}_diffusion_s0_dl${DL}_R${BEST_L}"
  train "$run" --arm diffusion --seed 0 --dec-layers "$DL" --dec-loops "$BEST_L" \
    --rand-loops "$BEST_L" --loop-emb
  for L in $DIAL; do
    gen "$run" "${TAG}_diff_s0_dl${DL}_R${BEST_L}_L${L}_k${K}" --dec-loops "$L"
  done
fi

echo
echo "=== curves: every log and config beside the generations ==="
mkdir -p out/curves
for d in runs/${TAG}_*/; do
  n=$(basename "$d")
  [ -f "$d/log.jsonl" ] || continue
  mkdir -p "out/curves/$n"
  cp "$d/log.jsonl" "$d/config.json" "out/curves/$n/"
done

echo
echo "=== done ==="
echo "bring back:  out/  (generations, and curves/ with every log.jsonl and config.json)"
echo "score on the laptop, where the sandbox is:"
echo "  for f in out/${TAG}_*_k${K}.jsonl; do python evaluate.py --score --cache $CACHE --gen-out \$f --split test; done"
echo "  for f in out/${TAG}_*_holdout.jsonl; do python evaluate.py --score --cache $CACHE --gen-out \$f --split holdout; done"
echo "  python report.py --dir out          # prints the sweep table and the gate verdict"
ls -la out/ | tail -20
