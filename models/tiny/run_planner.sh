# Step 4 with R12's stages: the first planners trained since the schema-graph
# fix (R11 §2a), on data_cache_s6g (general-English tokenizer, name words).
#
#   A0   one line encoder, names in the line (control)
#   SPt  split encoder, V2's stages loaded and fine-tuned
#
#   PAR=2 bash run_planner.sh     # from tiny/
#
# Everything the runs write is archived into planner_all.tar.gz at the end
# (every checkpoint, config, curve, generation and log): bring ALL of it back.
set -uo pipefail
cd "$(dirname "$0")"
export CACHE=data_cache_s6g
export STAGES="${STAGES:-runs/pod_vocab/runs/vocab/v2_6ep/stages_last.pt}"
export ARMS="${ARMS:-A0 SPt}"
export PAR="${PAR:-2}"
bash run_step4.sh > step4.log 2>&1
echo "run_step4 exit $?" >> step4.log
tar czf planner_all.tar.gz runs/s6_* out step4.log
ls -l planner_all.tar.gz
echo "=== run_planner done $(date)"
