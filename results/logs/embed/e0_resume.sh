#!/bin/bash
# Resume run_e0.sh after head a's OOM: wait for head b (already training), then a, then the checks.
set -euo pipefail
cd /workspace/repo/models/tiny
export HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1
OUT=reader/embed
LOGS=../../results/logs/embed
while kill -0 1010 2>/dev/null; do sleep 20; done
test -f $OUT/b/head.pt
mv $LOGS/train_a.log $LOGS/train_a_oom.log
python electra_reader.py train --mode alone --out $OUT/a > $LOGS/train_a.log 2>&1
python electra_reader.py check --head a=$OUT/a/head.pt --head b=$OUT/b/head.pt \
    --out $LOGS/check.json 2>&1 | tee $LOGS/check.log
cp /workspace/repo/models/tiny/run_e0.sh $LOGS/ 2>/dev/null || true
tar czf /workspace/e0_pull.tar.gz $OUT/a $OUT/b $OUT/_smoke/check.json $LOGS /workspace/e0_console.log /workspace/e0_resume.log
echo "E0 DONE"
touch /workspace/E0_DONE
