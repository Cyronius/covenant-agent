# Score run_brw.sh's three mixes on the laptop (.claude/plans/borrowed-worlds.md
# step 7). After unpacking brw_all.tar.gz into models/tiny/runs/pod_brw/:
#
#   bash results/logs/score_brw.sh
#
# 1. Regressions, as R23 measured them: sandbox scores for every holdout half
#    (plain, decoy, flip, clut) and test; play.py on the 500-task plain and
#    cluttered samples and the 70 demo requests.
# 2. The new kinds, per turn (play.py, live reader, backoff 3): the held-out
#    brief exams (boatyard, rooms_after, house) and the telecom service exam,
#    for the three mixes and clt_RD (the bar).
# 3. The new kinds, live (harness.rpg_suite --planner tiny --exits, 6 x 20):
#    the dungeon, house, boatyard, rooms_after, and the trainable workshop and
#    rooms, for the three mixes (clt_RD's rows: results/logs/tinyA_*).
set -u
ROOT=/c/code/covenant-agent
TINY=$ROOT/models/tiny
OUT=$TINY/runs/pod_brw/out
LOGS=$ROOT/results/logs/brw
RUNS="brw10_RD brw20_RD brw30_RD"
READER=$ROOT/models/tiny/reader
export HF_HUB_OFFLINE=1
mkdir -p "$LOGS"
cache() { echo "data_cache_${1%%_*}"; }
ckpt() { case $1 in clt_RD) echo runs/pod_clt/runs/clt_RD/best.pt;; *) echo runs/pod_brw/runs/$1/best.pt;; esac; }
ccache() { case $1 in clt_RD) echo data_cache_clt;; *) cache $1;; esac; }

# --- 1. regressions ---------------------------------------------------------
score() {  # run name split
  (cd "$TINY" && python -W ignore evaluate.py --score --cache $(cache $1) --split $3 \
      --gen-out "$OUT/$2.jsonl" > "$OUT/$2.score.log" 2>&1)
}
for run in $RUNS; do
  for half in plain decoy flip clut; do score $run ${run}_$half holdout & done
  score $run ${run}_test test &
done
wait
for run in $RUNS; do
  (cd "$TINY" && python -W ignore play.py --ckpt $(ckpt $run) --cache $(cache $run) \
      --tasks ../../data/holdout/e_demo_requests.jsonl --retype --reader-live $READER --backoff 3 \
      --show 0 --out "$LOGS/demo_$run.jsonl" > "$LOGS/demo_$run.log" 2>&1) &
  for v in s6 clt; do
    (cd "$TINY" && python -W ignore play.py --ckpt $(ckpt $run) --cache $(cache $run) \
        --tasks ../../data/${v}_holdout_s500.jsonl --reader-live $READER --backoff 3 \
        --show 0 --out "$LOGS/s500_${v}_$run.jsonl" > "$LOGS/s500_${v}_$run.log" 2>&1) &
  done
  wait
done

# --- 2. new kinds, per turn -------------------------------------------------
for run in ${BAR:+clt_RD} $RUNS; do   # BAR=1 re-scores clt_RD too
  for ex in e_brief_boatyard e_brief_rooms_after e_brief_house e_service_telecom; do
    (cd "$TINY" && python -W ignore play.py --ckpt $(ckpt $run) --cache $(ccache $run) \
        --tasks ../../data/holdout/$ex.jsonl --reader-live $READER --backoff 3 \
        --max-const 40 --max-field 56 --show 0 --out "$LOGS/${ex}_$run.jsonl" \
        > "$LOGS/${ex}_$run.log" 2>&1) &
  done
  wait
done

# --- 3. new kinds, live episodes --------------------------------------------
for run in $RUNS; do
  for w in rpg house boatyard rooms_after workshop rooms; do
    python -m harness.rpg_suite --world $w --planner tiny --tiny tiny:$run --exits \
        --episodes 6 --max-turns 20 --quiet --out "$LOGS/live_${w}_$run.jsonl" \
        > "$LOGS/live_${w}_$run.log" 2>&1
  done
done

cp "$OUT"/*.score.log "$OUT"/*_config.json "$OUT"/*_curve.jsonl "$OUT"/*.log "$LOGS"/ 2>/dev/null
grep -H -E "^  goal " "$OUT"/*.score.log
grep -H "tasks:" "$LOGS"/*.log
grep -H -E "won |compile" "$LOGS"/live_*.log
echo "=== score_brw done $(date)"
