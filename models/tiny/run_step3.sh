#!/usr/bin/env bash
# Step 3 of the NPU-native planner sequence: the weight format, trained from
# step zero. `.claude/plans/npu-native-planner.md`, "Sequence", step 3, and
# decision 3.
#
# What the format decides is capacity, not speed. The NPU holds a stage's
# weights in 4 MB of memory tiles, so the format sets how many parameters are
# resident: 4M at int8, 8M at 4-bit, 16M at ternary. `kernels/tern_mk/README.md`
# measured what each costs in cycles and found the speed argument for ternary is
# not available on this silicon -- 4-bit gets its unpack folded into the MAC by
# the hardware, ternary pays 0.031 cycles per weight in software, a 25% tax at a
# 64-slot canvas. So the trade is 2x the parameters against 25% more time per
# weight, and only training says whether that is worth taking.
#
# Which is why this sweep is at MATCHED RESIDENT BYTES, not matched parameters.
# Comparing ternary and int8 at the same parameter count asks whether rounding
# hurts, and of course it does. The question is which format writes better
# programs in the same 4 MB, because that is the choice the kernel has to be
# built for.
#
# The ladder that matches the bytes, at one width so nothing else moves:
#
#   format  bits/weight  layers in the block  resident
#   int8         8                1              1x
#   u4           4                2              1x
#   tern         2                4              1x
#   fp          32                1              the int8 shape, unrounded
#   fpx         32                4              the ternary shape, unrounded
#
# `fp` says what rounding to int8 costs. `fpx` says how much of ternary's extra
# capacity is real capacity rather than an artefact of the rounding: if fpx
# beats tern by a lot, ternary's parameters are not being used; if tern is close
# to fpx, the format is nearly free and the trade is worth taking.
#
# Width is a knob because the loop cost scales with it. D=256 puts the ternary
# arm at about 1 MB resident, which is a quarter of the budget at a quarter of
# the training cost and preserves every ratio above. D=512 with 4 layers is
# decision 14's shape at the full 4 MB and is the run to make once the ordering
# is known.
#
#   bash run_step3.sh                    # five arms at D=256, one seed
#   SEEDS="0 1 2" bash run_step3.sh      # the seed spread the gate needs
#   D=512 MODES="int8 tern" bash run_step3.sh   # the full-budget comparison
#
# The gate: ternary lands within the seed-to-seed spread of int8 at the same
# resident bytes. If it loses badly, the fallback is 4-bit at 8M rather than
# int8 at 4M, which is where the kernel measurement moved it.
set -euo pipefail

SEEDS="${SEEDS:-0}"
MODES="${MODES:-fp int8 u4 tern fpx}"
D="${D:-256}"                     # model width; 512 is decision 14's shape
LOOPS="${LOOPS:-8}"               # the loop count step 2 settles on
EPOCHS="${EPOCHS:-12}"
BATCH="${BATCH:-64}"
K="${K:-8}"
CACHE="${CACHE:-data_cache_struct}"
TAG="${TAG:-s3}"
# Smoke-testing this script costs one line rather than a pod: EXTRA goes to
# train.py and GEN_EXTRA to evaluate.py, so the whole sweep runs tiny and local.
#   EPOCHS=1 SEEDS=0 CACHE=data_cache_smoke LOOPS=2 GEN_EXTRA="--limit 4" \
#     TAG=_smoke3 EXTRA="--limit-train 32 --limit-val 16 --batch 4 --enc-layers 1" \
#     D=64 bash run_step3.sh
EXTRA="${EXTRA:-}"
GEN_EXTRA="${GEN_EXTRA:-}"
mkdir -p out

BINDING=$(python -c "import json,sys; print(json.load(open(sys.argv[1]+'/config.json')).get('binding','flat'))" "$CACHE")
[ "$BINDING" = structural ] || { echo "step 3 is a structural-binding sweep; $CACHE is $BINDING"; exit 1; }
echo "cache $CACHE  modes $MODES  width $D  loops $LOOPS  seeds $SEEDS  epochs $EPOCHS"

for mode in $MODES; do
  # Layers per format is what matches the resident bytes; the weight flag the
  # trainer takes is the format itself, and fpx is fp at the ternary shape.
  case "$mode" in
    fp)   layers=1; w=fp ;;
    int8) layers=1; w=int8 ;;
    u4)   layers=2; w=u4 ;;
    tern) layers=4; w=tern ;;
    fpx)  layers=4; w=fp ;;
    *) echo "unknown arm $mode"; exit 1 ;;
  esac
  for s in $SEEDS; do
    run="runs/${TAG}_diffusion_s${s}_${mode}_dl${layers}_L${LOOPS}"
    if [ -f "$run/best.pt" ]; then
      echo "  $run already trained, skipping"
    else
      echo "  --- $mode: $w weights, $layers layer(s) at d=$D, seed $s ---"
      # train.py prints the resident byte count and says OVER BUDGET when the
      # block does not fit. Nothing here should fit by accident.
      python train.py --arm diffusion --seed "$s" --cache "$CACHE" \
        --epochs "$EPOCHS" --batch "$BATCH" --pad-weight 0.5 \
        --d "$D" --dec-layers "$layers" --dec-loops "$LOOPS" --loop-emb \
        --weights "$w" $EXTRA --out "$run"
    fi
    name="${TAG}_diff_s${s}_${mode}_dl${layers}_L${LOOPS}_k${K}"
    [ -f "out/${name}.jsonl" ] || python evaluate.py --generate --ckpt "$run/best.pt" \
      --cache "$CACHE" --split test --steps "$K" $GEN_EXTRA --gen-out "out/${name}.jsonl"
    hname="${TAG}_diff_s${s}_${mode}_dl${layers}_L${LOOPS}_holdout"
    [ -f "out/${hname}.jsonl" ] || python evaluate.py --generate --ckpt "$run/best.pt" \
      --cache "$CACHE" --split holdout --steps "$K" $GEN_EXTRA --gen-out "out/${hname}.jsonl"
  done
done

echo
echo "=== resident budget per arm, read back from the run configs ==="
python - <<'PY'
import json
from pathlib import Path
rows = []
for c in sorted(Path("runs").glob("s3_*/config.json")):
    d = json.loads(c.read_text(encoding="utf-8"))
    rows.append((d["weights"], d["d"], d["dec_layers"], d["loop_body_params"],
                 d["resident_bytes"], d["params"], c.parent.name))
print(f"{'format':7} {'width':>6} {'layers':>7} {'block params':>13} "
      f"{'resident':>10} {'total':>11}  run")
for w, d, dl, body, rb, tot, name in rows:
    print(f"{w:7} {d:>6} {dl:>7} {body:>13,} {rb/(1<<20):9.2f}M {tot:>11,}  {name}")
quantised = [r for r in rows if r[0] != "fp"]
if quantised:
    span = max(r[4] for r in quantised) / max(min(r[4] for r in quantised), 1)
    print()
    print(f"resident bytes span {span:.2f}x across the quantised arms "
          f"({'matched' if span < 1.2 else 'NOT matched, the comparison is confounded'})")
    over = [r for r in quantised if r[4] > (4 << 20)]
    for r in over:
        print(f"  OVER BUDGET: {r[6]} needs {r[4]/(1<<20):.2f} MB of 4.00 MB")
PY

echo
echo "=== curves ==="
mkdir -p out/curves
for d in runs/${TAG}_*/; do
  n=$(basename "$d")
  [ -f "$d/log.jsonl" ] || continue
  mkdir -p "out/curves/$n"
  cp "$d/log.jsonl" "$d/config.json" "out/curves/$n/"
done

echo
echo "score on the laptop, where the sandbox is:"
echo "  for f in out/${TAG}_*_k${K}.jsonl; do python evaluate.py --score --cache $CACHE --gen-out \$f --split test; done"
echo "  for f in out/${TAG}_*_holdout.jsonl; do python evaluate.py --score --cache $CACHE --gen-out \$f --split holdout; done"
echo "  python report.py --dir out        # one row per format"
