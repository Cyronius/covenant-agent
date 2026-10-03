# Steps 3 and 4 of .claude/plans/staged-decoder-experts.md on one GPU.
# Needs step 2's checkpoints of the winning shape in runs/ (run_staged.sh):
#
#   BASE=d3_DR STAGES=draft:4x4,refine:2x1 bash run_experts.sh
#
# Experts are copies of the shared model's decoder, fine-tuned on one task
# type's rows with the encoder and output head fixed (train.py
# --freeze-shared), so a bundle of them shares one encoder:
#
#   x_s{S}_pair_{obsact,pages,data}    every stage per type          (pairs)
#   x_s{S}_draft_{obsact,pages,data}   the draft stages per type; the
#                                      shared model's refiner stays   (drafts)
#   x_s{S}_shared1p   the same fine-tuning on every type at once: the control
#                     for "more training", one decoder, same cost per request
#   x_s{S}_shared3    a decoder three times as deep, from scratch, all rows:
#                     the same total parameters as three experts
#
# Step 4, bolt-on (seed 0): np_base_s0 is the shared shape trained without
# the pages rows; np_s0_pair_{obsact,data} are its experts; then a pages
# expert is bolted on (np_s0_pair_pages; np_s0_draft_pages keeps np_base's
# refiner) and the router retrained with pages.
#
# Routers (router.py, on each shared model's encoder), temperatures
# (calibrate.py temp), bundles (staged.save_bundle), then greedy generation:
# every bundle with the true type (--oracle-route) and with its router.
# Thresholds, scoring and the play.py exams run on the laptop
# (results/logs/score_experts.sh). Everything goes into experts_all.tar.gz.
set -uo pipefail
cd "$(dirname "$0")"
BASE="${BASE:-d3_DR}"
STAGES="${STAGES:-draft:4x4,refine:2x1}"
SHARED3="${SHARED3:-draft:12x4,refine:6x1}"
SEEDS="${SEEDS:-0 1}"
JOBS="${JOBS:-3}"
CACHE=data_cache_d3
TYPES="obsact pages data"
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')"

TRAIN="python train.py --arm ar --cache $CACHE --batch 64 --pad-weight 0.5 --pack"
expert() {  # name base_ckpt parts types [extra...]
  local name=$1 ck=$2 parts=$3 types=$4; shift 4
  [ -f runs/$name/last.pt ] && return 0
  $TRAIN --stages $STAGES --init-from $ck --freeze-shared --train-parts $parts --types $types \
      --lr 1e-4 --warmup 100 --epochs 12 --min-steps 1500 --out runs/$name "$@" \
      > out/$name.train.log 2>&1 || echo "train $name failed"
  cp runs/$name/log.jsonl out/${name}_curve.jsonl 2>/dev/null
}
full() {  # name seed stages [extra...]
  local name=$1 seed=$2 stages=$3; shift 3
  [ -f runs/$name/last.pt ] && return 0
  $TRAIN --stages $stages --seed $seed --epochs 12 --out runs/$name "$@" \
      > out/$name.train.log 2>&1 || echo "train $name failed"
  cp runs/$name/log.jsonl out/${name}_curve.jsonl 2>/dev/null
}
pool() {  # run each line of stdin as a command, JOBS at a time
  while read -r cmd; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 20; done
    eval "$cmd" &
  done
  wait
}

echo "=== 1. training $(date)"
{
  echo "full np_base_s0 0 $STAGES --types obsact,data"
  for s in $SEEDS; do
    b=runs/${BASE}_s$s/best.pt
    for t in $TYPES; do
      echo "expert x_s${s}_pair_$t $b decoder $t"
      echo "expert x_s${s}_draft_$t $b draft $t"
    done
    echo "expert x_s${s}_shared1p $b decoder obsact,pages,data --min-steps 8800"
    echo "full x_s${s}_shared3 $s $SHARED3"
  done
} | pool
# the bolt-on experts need np_base_s0
{
  for t in obsact data pages; do echo "expert np_s0_pair_$t runs/np_base_s0/best.pt decoder $t"; done
  echo "expert np_s0_draft_pages runs/np_base_s0/best.pt draft pages"
} | pool

echo "=== 2. temperatures and routers $(date)"
temp() {  # name types
  [ -f out/$1_temp.json ] || python calibrate.py temp --ckpt runs/$1/best.pt --cache $CACHE \
      --types $2 --out out/$1_temp.json > out/$1_temp.log 2>&1
}
for s in $SEEDS; do
  for t in $TYPES; do temp x_s${s}_pair_$t $t; temp x_s${s}_draft_$t $t; done
done
for t in obsact data pages; do temp np_s0_pair_$t $t; done
temp np_s0_draft_pages pages
for s in $SEEDS; do
  python router.py --ckpt runs/${BASE}_s$s/best.pt --cache $CACHE --out runs/router_s$s.pt \
      --report out/router_s$s.json --exam-root .. > out/router_s$s.log 2>&1
done
python router.py --ckpt runs/np_base_s0/best.pt --cache $CACHE --out runs/router_np2.pt \
    --exclude pages --report out/router_np2.json --exam-root .. > out/router_np2.log 2>&1
python router.py --ckpt runs/np_base_s0/best.pt --cache $CACHE --out runs/router_np3.pt \
    --report out/router_np3.json --exam-root .. > out/router_np3.log 2>&1

echo "=== 3. bundles $(date)"
bundle() {  # out router margin name:ckpt...
  local dest=$1 router=$2 margin=$3; shift 3
  python - "$dest" "$router" "$margin" "$@" <<'EOF'
import json, sys, torch
from staged import Router, save_bundle
dest, router, margin, *pairs = sys.argv[1:]
r = torch.load(router)
R = Router(r["d"], r["names"], r["hidden"]); R.load_state_dict(r["state"])
experts = {}
for p in pairs:
    name, run = p.split(":")
    temp = json.load(open(f"out/{run}_temp.json"))["temp"]
    experts[name] = (f"runs/{run}/best.pt", {"temp": temp})
save_bundle(dest, experts, R, float(margin))
print("bundle", dest, list(experts))
EOF
}
for s in $SEEDS; do
  bundle runs/b_s${s}_pairs.pt runs/router_s$s.pt 0.8 \
      obsact:x_s${s}_pair_obsact pages:x_s${s}_pair_pages data:x_s${s}_pair_data
  bundle runs/b_s${s}_drafts.pt runs/router_s$s.pt 0.8 \
      obsact:x_s${s}_draft_obsact pages:x_s${s}_draft_pages data:x_s${s}_draft_data
done
bundle runs/b_np_old.pt runs/router_np2.pt 0.8 obsact:np_s0_pair_obsact data:np_s0_pair_data
bundle runs/b_np_bolt.pt runs/router_np3.pt 0.8 \
    obsact:np_s0_pair_obsact pages:np_s0_pair_pages data:np_s0_pair_data
bundle runs/b_np_bolt_draft.pt runs/router_np3.pt 0.8 \
    obsact:np_s0_pair_obsact pages:np_s0_draft_pages data:np_s0_pair_data

echo "=== 4. generation $(date)"
gen() {  # ckpt tag split limit [flags...]
  local ck=$1 tag=$2 split=$3 lim=$4; shift 4
  [ -f out/$tag.jsonl ] && return 0
  python evaluate.py --generate --ckpt $ck --cache $CACHE --split $split --limit $lim "$@" \
      --gen-out out/$tag.jsonl > out/$tag.gen.log 2>&1 || echo "gen $tag failed"
}
halves() {  # ckpt tag [flags...]
  local ck=$1 tag=$2; shift 2
  echo "gen $ck ${tag}_plain holdout 1000 --id-not-contains + $*"
  echo "gen $ck ${tag}_decoy holdout 1000 --id-contains +decoy $*"
  echo "gen $ck ${tag}_flip holdout 1000 --id-contains +flip $*"
  echo "gen $ck ${tag}_clut holdout 1000 --id-contains +clut $*"
  echo "gen $ck ${tag}_test test 2000 $*"
}
JOBS=6
{
  for s in $SEEDS; do
    for k in pairs drafts; do
      halves runs/b_s${s}_$k.pt b_s${s}_${k}_or --oracle-route
      halves runs/b_s${s}_$k.pt b_s${s}_${k}_rt
      echo "gen runs/b_s${s}_$k.pt b_s${s}_${k}_or_val val 2000 --oracle-route"
    done
    for m in shared1p shared3; do
      halves runs/x_s${s}_$m/best.pt x_s${s}_$m
      echo "gen runs/x_s${s}_$m/best.pt x_s${s}_${m}_val val 2000"
    done
  done
  # np_old has no pages expert, so no true-type route for pages rows
  echo "gen runs/b_np_old.pt b_np_old_rt_test test 2000"
  for b in np_bolt np_bolt_draft; do
    echo "gen runs/b_$b.pt b_${b}_rt_test test 2000"
    echo "gen runs/b_$b.pt b_${b}_or_test test 2000 --oracle-route"
    echo "gen runs/b_$b.pt b_${b}_or_val val 2000 --oracle-route"
  done
} | pool

tar czf experts_all.tar.gz runs/x_s* runs/np_* runs/b_*.pt runs/router_*.pt out
ls -l experts_all.tar.gz
echo "=== run_experts done $(date)"
