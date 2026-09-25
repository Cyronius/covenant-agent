# R11 §7 item 2, done without regenerating: X4's recipe on the SAME S6 cache,
# with the unreadable flip-slot calls left out of training (--drop-unreadable).
#
#   bash results/logs/stages_u.sh    # -> runs/stages_u1, runs/stages_u2
#
# Why not regenerate (gen_s6b.sh, 2026-09-24): the generator's authored-role
# fit check from 7c3e765 rejects every L9 "ambiguous" row and most L12 rows,
# whose requests have fixed wording by design; every shard died at
# "FATAL: level 9 failed 25 consecutive attempts" with 1,701 of 30,000 rows.
# In S6's training split those calls are 2,985 of 14,742 called flip-slot
# tools, all with the authored tool as the answer (L9 1,037 rows, L12 1,116,
# L15/17/19/4/7 the rest). Masking them in the stage's losses removes the
# "authored tool wins" signal without dropping the levels from the corpus.
#
#   U1  3 epochs  -- U1 vs X4 isolates the mask
#   U2  6 epochs  -- U2 vs U1 isolates duration (X4's flat end was the cosine
#                    schedule reaching zero, not evidence either way)
# Graded on the full exam like X4's full_exam.json.
# Gate: acc_flip_comb >= acc_flip_t_desc (the teacher's 89.9%).
set -e
cd /c/code/covenant-agent/models/tiny
RECIPE="--desc-w 128 --pool mean --init-emb teacher --lam-rel 20 --flip-weight 3 --drop-unreadable --limit-eval 2000 --eval-every 400"
for arm in "u1 3" "u2 6"; do
  set -- $arm
  python -W ignore stage_pretrain.py --cache data_cache_s6 --out runs/stages_$1 $RECIPE --epochs $2 \
      > ../../results/logs/stages_$1.log 2>&1
  python -W ignore stage_quant.py --stages runs/stages_$1/stages.pt --cache data_cache_s6 \
      --limit-eval 100000 --modes fp --json runs/stages_$1/full_exam.json \
      > ../../results/logs/stages_$1_full.log 2>&1
  echo "=== $1 done $(date) ==="
done
