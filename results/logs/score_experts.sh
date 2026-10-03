# Score run_experts.sh on the laptop (.claude/plans/staged-decoder-experts.md
# steps 3 and 4). After unpacking experts_all.tar.gz into
# models/tiny/runs/pod_experts/:
#
#   bash results/logs/score_experts.sh
#
# 1. Sandbox scores for every generation and its drafts.
# 2. Each bundle's early-exit thresholds, one per expert, on its val
#    generation (calibrate.py thr --by-expert); the bundle rewritten with them
#    (b_*_cal.pt); every exam at those thresholds, offline.
# 3. The held-out exams per turn (play.py, backoff 3), router-routed, for the
#    bundles and the two controls: the brief worlds, the coursebuilder app,
#    telecom and the demo requests. Bundles play at their thresholds.
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
POD=$TINY/runs/pod_experts
OUT=$POD/out
LOGS=$ROOT/results/logs/experts
P="${P:-8}"
CACHE=data_cache_d3
SEEDS="${SEEDS:-0 1}"
export HF_HUB_OFFLINE=1
mkdir -p "$LOGS"

# --- 1. sandbox scores --------------------------------------------------------
for f in "$OUT"/*.jsonl; do
  case $f in *_curve.jsonl) continue ;; esac
  [ -f "${f%.jsonl}.score.json" ] && continue
  split=holdout
  case $f in *_test.jsonl|*_test.draft.jsonl) split=test ;; *_val.jsonl|*_val.draft.jsonl) split=val ;; esac
  echo "$split $f"
done | xargs -P "$P" -L 1 bash -c 'cd '"$TINY"' && python -W ignore evaluate.py --score --cache '"$CACHE"' --split $0 --gen-out $1 > ${1%.jsonl}.score.log 2>&1'

# --- 2. thresholds per expert --------------------------------------------------
BUNDLES=""
for s in $SEEDS; do BUNDLES="$BUNDLES b_s${s}_pairs b_s${s}_drafts"; done
for b in $BUNDLES b_np_bolt b_np_bolt_draft; do
  (cd "$TINY" && python -W ignore calibrate.py thr --by-expert --val-gen "$OUT/${b}_or_val.jsonl" \
      --bundle "$POD/runs/$b.pt" --out "$LOGS/${b}_thr.json" --write "$POD/runs/${b}_cal.pt" \
      > "$LOGS/${b}_thr.log" 2>&1)
  ls "$OUT"/${b}_*.jsonl 2>/dev/null | grep -v -e '\.draft\.' -e '_val\.' | \
    (cd "$TINY" && xargs python -W ignore calibrate.py apply --by-expert --thr "$LOGS/${b}_thr.json" --gen \
      > "$LOGS/${b}_at_thr.log" 2>&1)
done

# --- 3. held-out exams per turn, router-routed -----------------------------------
{
  for s in $SEEDS; do
    for k in pairs drafts; do echo "b_s${s}_${k}_cal.pt b_s${s}_$k"; done
    for m in shared1p shared3; do echo "x_s${s}_$m/best.pt x_s${s}_$m"; done
  done
  echo "b_np_old.pt b_np_old"
  echo "b_np_bolt_cal.pt b_np_bolt"
  echo "b_np_bolt_draft_cal.pt b_np_bolt_draft"
} | while read -r ck tag; do
  for ex in e_brief_boatyard e_brief_rooms_after e_brief_house e_brief_coursebuilder e_service_telecom e_demo_requests; do
    echo "$ck $tag $ex"
  done
done | xargs -P "$P" -L 1 bash -c '
  ck=$0 tag=$1 ex=$2
  extra="--max-const 40 --max-field 56"
  [ "$ex" = e_demo_requests ] && extra="--retype --max-const 24"
  dest='"$LOGS"'/${ex}_${tag}
  [ -f $dest.jsonl ] && exit 0
  cd '"$TINY"' && python -W ignore play.py --ckpt '"$POD"'/runs/$ck --cache '"$CACHE"' \
      --tasks ../../data/holdout/$ex.jsonl --backoff 3 $extra --show 0 --out $dest.jsonl > $dest.log 2>&1'

cp "$OUT"/*.log "$OUT"/*.json "$LOGS"/ 2>/dev/null
echo "=== score_experts done $(date)"
