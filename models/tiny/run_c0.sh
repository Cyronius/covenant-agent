# C0 on one GPU (.claude/plans/tiny-general-agent-menu.md): R23's recipe
# (A0 + frozen Ternlight-mini reader, results/R23.md) on the same cluttered
# corpus, with the reader also on every constant, every field and the request
# in 4-word chunks (data_cache_c0, prep.py --reader-lines). Three arms, two
# seeds each, since single-seed plain differences under ~2.5 points are noise
# (.claude/plans/role-aware-reader.md step 6):
#
#   c0_RL / c0_RLs1     today's Ternlight (reader.pt), the planner's own
#                       request words kept
#   c0_RLr / c0_RLrs1   the role-aware reader (reader_role.pt, tern_train.py),
#                       request words kept
#   c0_RVr / c0_RVrs1   the role-aware reader, the request's own words dropped:
#                       the request reaches the planner only as reader vectors
#
# The baseline is clt_RD / clt_RDs1 (R23): same rows, same recipe, same exams.
# Greedy generation on the holdout's four halves (plain, decoy, flip, clut)
# and the test split; backoff, scoring and the demo run on the laptop.
#
#   bash run_c0.sh      # from tiny/
#
# Everything is archived into c0_all.tar.gz at the end: bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
RUNS="c0_RL c0_RLs1 c0_RLr c0_RLrs1 c0_RVr c0_RVrs1"
CACHE=data_cache_c0
mkdir -p out
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(2048,2048,device='cuda'); print('cuda ok', float((x@x).sum()))"
[ -f $CACHE/reader_role.pt ] || { echo "no $CACHE/reader_role.pt (reader_table.py --reader-model)"; exit 2; }

flags() {  # run
  case "$1" in
    c0_RL) echo "--pack --seed 0 --reader --reader-lines" ;;
    c0_RLs1) echo "--pack --seed 1 --reader --reader-lines" ;;
    c0_RLr) echo "--pack --seed 0 --reader --reader-lines --reader-file reader_role.pt" ;;
    c0_RLrs1) echo "--pack --seed 1 --reader --reader-lines --reader-file reader_role.pt" ;;
    c0_RVr) echo "--pack --seed 0 --reader --reader-lines --reader-file reader_role.pt --no-req-words" ;;
    c0_RVrs1) echo "--pack --seed 1 --reader --reader-lines --reader-file reader_role.pt --no-req-words" ;;
    *) echo "unknown run $1" >&2; exit 2 ;;
  esac
}
gen() {  # run split tag selector...
  local run=$1 split=$2 tag=$3; shift 3
  local lim=2000; [ "$split" = test ] && lim=1000
  python evaluate.py --generate --ckpt runs/$run/best.pt --cache $CACHE --split $split \
      --limit $lim "$@" --gen-out out/${run}_${tag}.jsonl
}
arm() {  # run
  local run=$1
  if [ ! -f "runs/$run/best.pt" ]; then
    # shellcheck disable=SC2046
    python train.py --arm ar --cache $CACHE --epochs 12 --batch 64 --pad-weight 0.5 \
        --out runs/$run $(flags "$run") || { echo "train $run failed"; return 1; }
  fi
  cp runs/$run/config.json out/${run}_config.json
  cp runs/$run/log.jsonl out/${run}_curve.jsonl
  gen $run holdout plain --id-not-contains +
  gen $run holdout decoy --id-contains +decoy
  gen $run holdout flip --id-contains +flip
  gen $run holdout clut --id-contains +clut
  gen $run test test
}
smoke() {  # run
  # shellcheck disable=SC2046
  python train.py --arm ar --cache $CACHE --epochs 1 --limit-train 128 \
    --limit-val 32 --batch 16 --eval-every 4 --out runs/_smoke_$1 $(flags $1) || return 1
  rm -rf runs/_smoke_$1
}
echo "=== smoke: every run builds and steps ==="
for r in $RUNS; do smoke $r || exit 1; done

for r in $RUNS; do
  arm $r > out/$r.log 2>&1 &
done
sleep 300
nvidia-smi > out/nvidia_smi_5min.txt 2>&1
wait
tar czf c0_all.tar.gz $(for r in $RUNS; do echo runs/$r; done) out
ls -l c0_all.tar.gz
echo "=== run_c0 done $(date)"
