#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
for f in out_s1/s1_*_test.jsonl out_s1/s1_*_k*.jsonl; do
  [ -f "${f%.jsonl}.score.json" ] && continue
  echo "== $f"
  python evaluate.py --score --cache data_cache_struct --gen-out "$f" --split test 2>&1 | tail -3
done
for f in out_s1/s1_*_holdout.jsonl; do
  [ -f "${f%.jsonl}.score.json" ] && continue
  echo "== $f"
  python evaluate.py --score --cache data_cache_struct --gen-out "$f" --split holdout 2>&1 | tail -3
done
echo DONE
