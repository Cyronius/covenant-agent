# The D3 cache (staged-decoder-experts.md step 0b, step 2's training data):
# every clt_train row plus every new row (results/logs/mix_d3.py), prepped
# with gen_brw.sh's flags so its rows render exactly as data_cache_clt's and
# data_cache_brw30's do. --limit is raised to keep all 39,034 rows. No
# reader table: the owner took Ternlight out (2026-10-02), so every run in
# this plan trains without --reader.
#
#   bash results/logs/gen_d3.sh
set -e
cd /c/code/covenant-agent
export HF_HUB_OFFLINE=1
python results/logs/mix_d3.py
python data/borrowed/canary_check.py data/d3_train.jsonl
PREP="--limit 40000 --holdout-corpus ../../data/clt_holdout_both.jsonl --names --desc-chars 120 --split \
      --max-line 112 --max-sig 64 --name-words --in-tok data_cache_s6g/in_tok.json --teacher unsloth/bge-small-en-v1.5"
cd models/tiny
python -W ignore prep.py --corpus ../../data/d3_train.jsonl $PREP --out data_cache_d3 \
    > ../../results/logs/prep_d3.log 2>&1
grep -E "layout|longest|kept|dropped|train " ../../results/logs/prep_d3.log | head -8
echo "=== gen_d3 done $(date)"
