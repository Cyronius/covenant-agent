#!/usr/bin/env bash
# Score the 42-unseen-world run, both holdout halves, on the laptop.
#
#   bash score_ho42.sh [out_dir] [cache]
#
# Scoring executes programs against world state, so it needs Node and the
# themed worlds -- which is why it happens here and not on the pod
# (models/tiny/pack.py ships neither).
#
# What the halves are. data_cache_ho42's holdout is data/s5_holdout_both.jsonl:
# 7,624 plain rows then 7,624 decoyed ones carrying `+decoy` on their ids, the
# same tasks either way. run_step1.sh generates HOLDOUT_LIMIT of each into
# `*_holdout.jsonl` and `*_holdout_decoy.jsonl`, so one trained model produces
# both numbers and their difference is attributable to the decoys alone.
#
# READ THE DECOY HALF AGAINST chance_tool_sig, NOT chance_tool. chance_tool is
# a uniform pick over every declared tool (~2.4% at 41 tools/task); a model
# that infers the type shape and reads no description has already narrowed to
# the signature collision group and scores 43.5%. On the plain half that line
# is 100.0%, because the signature is unique in every task
# (results/GROUNDING.md). So:
#
#   plain half   same_tool tells you about binding, not grounding -- there is
#                no score a pure shape matcher cannot reach there
#   decoy half   same_tool above 43.5% is description reading; at or below it
#                is not, whatever the absolute number looks like
#
# `decoy_called` is the companion: a decoy is never in a reference program, so
# calling one is a miss whatever the task returns -- and on an abstain task the
# outcome is identical either way, which is the whole reason the column exists.
set -euo pipefail
OUT="${1:-out}"
CACHE="${2:-data_cache_ho42}"

shopt -s nullglob
for f in "$OUT"/ho42_*_test.jsonl "$OUT"/ho42_*_k*.jsonl; do
  echo "=== $f ==="
  python evaluate.py --score --cache "$CACHE" --gen-out "$f" --split test
done
for f in "$OUT"/ho42_*_holdout.jsonl "$OUT"/ho42_*_holdout_decoy.jsonl; do
  echo "=== $f ==="
  python evaluate.py --score --cache "$CACHE" --gen-out "$f" --split holdout
done

echo
echo "=== the comparison this run exists to make ==="
python ho42_report.py "$OUT"
