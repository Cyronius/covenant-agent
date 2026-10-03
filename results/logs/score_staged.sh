# Score run_staged.sh's twelve runs on the laptop (.claude/plans/
# staged-decoder-experts.md steps 0b and 2). After unpacking staged_all.tar.gz
# into models/tiny/runs/pod_staged/:
#
#   bash results/logs/score_staged.sh
#
# 1. Sandbox scores for every generation: the holdout's four halves (plain,
#    decoy, flip, cluttered), test, and for the staged runs their drafts and
#    the val split.
# 2. Calibration of the staged runs on val (calibrate.py fit): temperature and
#    the early-exit threshold; then every exam at that threshold, offline.
# 3. The new kinds and the demo requests, per turn (play.py, backoff 3): the
#    held-out brief worlds (boatyard, rooms_after, house), the held-out page
#    app (coursebuilder), telecom, and the 70 demo requests. Staged runs play
#    twice: always refining, and at their threshold.
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
POD=$TINY/runs/pod_staged
OUT=$POD/out
LOGS=$ROOT/results/logs/staged
P="${P:-8}"
export HF_HUB_OFFLINE=1
mkdir -p "$LOGS"
RUNS="${RUNS:-clt_base_s0 clt_base_s1 d3_base_s0 d3_base_s1 d3_ctrl6_s0 d3_ctrl6_s1 \
      d3_draft_s0 d3_draft_s1 d3_DR_s0 d3_DR_s1 d3_D2R_s0 d3_D2R_s1}"
STAGED="${STAGED-d3_DR_s0 d3_DR_s1 d3_D2R_s0 d3_D2R_s1}"
cache() { case $1 in clt_*) echo data_cache_clt ;; *) echo data_cache_d3 ;; esac; }

# --- 1. sandbox scores --------------------------------------------------------
jobs1() {
  for run in $RUNS; do
    for f in "$OUT"/${run}_*.jsonl; do
      case $f in *_curve.jsonl) continue ;; esac
      local tag=${f##*/${run}_}; tag=${tag%.jsonl}; tag=${tag%.draft}
      local split=holdout; [ "$tag" = test ] && split=test; [ "$tag" = val ] && split=val
      [ -f "${f%.jsonl}.score.json" ] && continue
      echo "$(cache $run) $split $f"
    done
  done
}
jobs1 | xargs -P "$P" -L 1 bash -c 'cd '"$TINY"' && python -W ignore evaluate.py --score --cache $0 --split $1 --gen-out $2 > ${2%.jsonl}.score.log 2>&1'

# --- 2. calibration and every exam at the threshold ---------------------------
for run in $STAGED; do
  (cd "$TINY" && python -W ignore calibrate.py fit --ckpt "$POD/runs/$run/best.pt" --cache $(cache $run) \
      --val-gen "$OUT/${run}_val.jsonl" --out "$LOGS/${run}_cal.json" --device cpu \
      > "$LOGS/${run}_cal.log" 2>&1)
  (cd "$TINY" && python -W ignore calibrate.py apply --thr "$LOGS/${run}_cal.json" \
      --gen "$OUT/${run}_plain.jsonl" "$OUT/${run}_decoy.jsonl" "$OUT/${run}_flip.jsonl" \
            "$OUT/${run}_clut.jsonl" "$OUT/${run}_test.jsonl" > "$LOGS/${run}_at_thr.log" 2>&1)
done

# --- 3. new kinds and the demo, per turn ---------------------------------------
jobs3() {
  for run in $RUNS; do
    local modes=never
    case " $STAGED " in *" $run "*) modes="never thr" ;; esac
    case $run in *_draft_*) modes=always ;; esac
    for mode in $modes; do
      for ex in e_brief_boatyard e_brief_rooms_after e_brief_house e_brief_coursebuilder e_service_telecom e_demo_requests; do
        echo "$run $(cache $run) $ex $mode"
      done
    done
  done
}
jobs3 | xargs -P "$P" -L 1 bash -c '
  run=$0 cache=$1 ex=$2 mode=$3
  exit_=$mode
  [ "$mode" = thr ] && exit_=$(python -c "import json,sys; t=json.load(open(sys.argv[1]))[\"thr\"]; print(\"never\" if t is None else t)" '"$LOGS"'/${run}_cal.json)
  extra="--max-const 40 --max-field 56"
  [ "$ex" = e_demo_requests ] && extra="--retype --max-const 24"
  dest='"$LOGS"'/${ex}_${run}_${mode}
  [ -f $dest.jsonl ] && exit 0
  cd '"$TINY"' && STAGED_EXIT=$exit_ python -W ignore play.py --ckpt '"$POD"'/runs/$run/best.pt --cache $cache \
      --tasks ../../data/holdout/$ex.jsonl --backoff 3 $extra --show 0 --out $dest.jsonl > $dest.log 2>&1'

cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$OUT"/*.log "$LOGS"/ 2>/dev/null
echo "=== score_staged done $(date)"
