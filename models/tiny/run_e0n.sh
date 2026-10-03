#!/bin/bash
# R27's next step (.claude/plans/electra-only-reader.md, step 0): retrain the
# one-pass head (b) with generated names in its pieces, then check it beside
# R27's b. Needs reader/embed/b/head.pt. A smoke pass over tiny data first.
#
#   bash run_e0n.sh > /workspace/e0n_console.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1
OUT=reader/embed
LOGS=../../results/logs/embed
mkdir -p $OUT/_smoke $LOGS

python -m pip install -q --break-system-packages transformers tokenizers faker
python -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print('cuda ok', float((x @ x).abs().mean()))"

python electra_reader.py targets --out $OUT/_smoke/targets_n.pt --n-texts 2000 --n-names 2000 --n-held 200 \
    --names-extra 3000 --sub-rows 300
python electra_reader.py train --mode mixed --name-frac 0.25 --out $OUT/_smoke/n --targets $OUT/_smoke/targets_n.pt \
    --steps 40 --log-every 10 --eval-every 40 --n-eval 50
python electra_reader.py check --head n=$OUT/_smoke/n/head.pt --targets $OUT/_smoke/targets_n.pt \
    --limit-tasks 10 --n-pairs 500 --out $OUT/_smoke/check_n.json
echo "smoke done"

python electra_reader.py targets --out $OUT/targets_n.pt --names-extra 170000 --sub-rows 40000 \
    2>&1 | tee $LOGS/targets_n.log
python electra_reader.py train --mode mixed --name-frac 0.25 --out $OUT/n --targets $OUT/targets_n.pt \
    > $LOGS/train_n.log 2>&1
python electra_reader.py check --head b=$OUT/b/head.pt --head n=$OUT/n/head.pt --targets $OUT/targets_n.pt \
    --out $LOGS/check_n.json 2>&1 | tee $LOGS/check_n.log
echo "E0N DONE"
touch /workspace/E0N_DONE
