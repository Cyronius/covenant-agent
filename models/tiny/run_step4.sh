#!/usr/bin/env bash
# Description-reading plan step 4: join the description and name stages to
# the planner. One pod session, four arms, R10's recipe (ar, 12 epochs,
# batch 64, seed 0, pad-weight 0.5), step 1's data (the S6 cache: authored
# decoys on every training row, uniform twin roles, names on the tool line,
# 15% opaque-name rows), packing on in every arm.
#
#   A0   one line encoder, names in the line        the control
#   S1   split encoder, from scratch                 + contrastive + relational
#   SPf  split, stages from step 3, FROZEN
#   SPt  split, stages from step 3, fine-tuned       + both losses, stage lr x0.1
#
#   CACHE=data_cache_s6 STAGES=runs/stages_d256/stages.pt ARMS="A0 S1 SPf SPt" \
#     bash run_step4.sh > step4.log 2>&1
#
# PAR=2 trains two arms at once. Check `nvidia-smi` after the first starts:
# packing makes the 21 GB of R10 a ceiling, not an estimate.
#
# Bring back out/ AND runs/*/best.pt before terminating the pod: R10's
# checkpoint was lost exactly that way (results/R10.md section 7).
set -euo pipefail
CACHE="${CACHE:-data_cache_s6}"
STAGES="${STAGES:-runs/stages/stages.pt}"
ARMS="${ARMS:-A0 S1 SPf SPt}"
EPOCHS="${EPOCHS:-12}"
BATCH="${BATCH:-64}"
PAR="${PAR:-1}"
HOLDOUT_LIMIT="${HOLDOUT_LIMIT:-2000}"
# step 3's recipe (results/R11.md §3): mean pooling -- a summary token left
# flip-slot decisions at chance -- and a relational weight that matters (the
# MSE between cosine matrices is ~0.02, so 0.5 contributed nothing)
WIDTH="--desc-w ${DESC_W:-128} --desc-layers ${DESC_LAYERS:-4} --stage-pool ${POOL:-mean}"
LOSSES="${LOSSES:---lam-nce 0.5 --lam-rel 20}"
mkdir -p out

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"

arm_flags() {
  case "$1" in
    A0)  echo "--pack" ;;
    S1)  echo "--pack --split $WIDTH $LOSSES" ;;
    SPf) echo "--pack --split $WIDTH --stages-from $STAGES --freeze-stages" ;;
    SPt) echo "--pack --split $WIDTH --stages-from $STAGES --stage-lr-mult 0.1 $LOSSES" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

run_arm() {
  local arm="$1" run="runs/s6_${1}"
  if [ ! -f "$run/best.pt" ]; then
    # shellcheck disable=SC2046
    python train.py --arm ar --seed 0 --cache "$CACHE" --epochs "$EPOCHS" \
      --batch "$BATCH" --pad-weight 0.5 --out "$run" $(arm_flags "$arm")
  fi
  cp "$run/config.json" "out/s6_${arm}_config.json"
  cp "$run/log.jsonl" "out/s6_${arm}_curve.jsonl"
  for half in plain decoy flip; do
    case "$half" in
      plain) sel="--id-not-contains +"; ;;
      decoy) sel="--id-contains +decoy" ;;
      flip)  sel="--id-contains +flip" ;;
    esac
    # shellcheck disable=SC2086
    python evaluate.py --generate --ckpt "$run/best.pt" --cache "$CACHE" \
      --split holdout --limit "$HOLDOUT_LIMIT" $sel \
      --gen-out "out/s6_${arm}_holdout_${half}.jsonl"
  done
  python evaluate.py --generate --ckpt "$run/best.pt" --cache "$CACHE" \
    --split test --limit 1000 --gen-out "out/s6_${arm}_test.jsonl"
}

echo "=== smoke: every arm builds and steps ==="
for arm in $ARMS; do
  # shellcheck disable=SC2046
  python train.py --arm ar --cache "$CACHE" --epochs 1 --limit-train 128 \
    --limit-val 32 --batch 16 --eval-every 4 --out "runs/_smoke_$arm" $(arm_flags "$arm")
  rm -rf "runs/_smoke_$arm"
done

running=0
for arm in $ARMS; do
  run_arm "$arm" > "out/s6_${arm}.log" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge "$PAR" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "=== step 4 done: bring back out/ and runs/s6_*/best.pt ==="
