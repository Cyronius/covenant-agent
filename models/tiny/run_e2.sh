#!/bin/bash
# Step 2 of .claude/plans/electra-only-reader.md on a pod: ELECTRA reads every
# text of data_cache_c0 (descriptions, requests, constants, fields alone; the
# request's role slots and chunks in one pass) -> reader/embed/c0_patch.pt,
# which electra_cache.py apply turns into data_cache_c0e / data_cache_c0esc on
# the laptop. Ships only the cache's config, teacher.pt and reader_texts.jsonl.
#
#   bash run_e2.sh > /workspace/e2_console.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1
python -m pip install -q --break-system-packages transformers tokenizers
python -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print('cuda ok', float((x @ x).abs().mean()))"
python electra_cache.py build --limit 3000 --out reader/embed/_smoke/c0_patch.pt
echo "smoke done"
python electra_cache.py build
echo "E2 DONE"
touch /workspace/E2_DONE
