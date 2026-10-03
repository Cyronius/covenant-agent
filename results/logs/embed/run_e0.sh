#!/bin/bash
# Step 0 of .claude/plans/electra-only-reader.md on a pod: the teacher's
# targets, the two embedding heads (alone, mixed) side by side, then the
# checks and gates. A smoke pass over tiny data runs first, so a bug costs a
# minute rather than the run.
#
#   bash run_e0.sh > /workspace/e0_console.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1
OUT=reader/embed
LOGS=../../results/logs/embed
mkdir -p $OUT/_smoke $LOGS

python -m pip install -q --break-system-packages transformers tokenizers
python -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print('cuda ok', float((x @ x).abs().mean()))"

python electra_reader.py targets --out $OUT/_smoke/targets.pt --n-texts 2000 --n-names 2000 --n-held 200
python electra_reader.py train --mode mixed --out $OUT/_smoke/b --targets $OUT/_smoke/targets.pt \
    --steps 40 --log-every 10 --eval-every 40 --n-eval 50
python electra_reader.py check --head b=$OUT/_smoke/b/head.pt --targets $OUT/_smoke/targets.pt \
    --limit-tasks 10 --n-pairs 500 --out $OUT/_smoke/check.json
echo "smoke done"

python electra_reader.py targets 2>&1 | tee $LOGS/targets.log
# One after the other: side by side they need ~23 GB and one dies of OOM on a
# 24 GB card (2026-10-02, RTX 3090).
python electra_reader.py train --mode mixed --out $OUT/b > $LOGS/train_b.log 2>&1
python electra_reader.py train --mode alone --out $OUT/a > $LOGS/train_a.log 2>&1
python electra_reader.py check --head a=$OUT/a/head.pt --head b=$OUT/b/head.pt \
    --out $LOGS/check.json 2>&1 | tee $LOGS/check.log
tar czf /workspace/e0_pull.tar.gz $OUT/a $OUT/b $OUT/_smoke/check.json $LOGS
echo "E0 DONE"
touch /workspace/E0_DONE
