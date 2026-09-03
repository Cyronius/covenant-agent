# S2 corpus (plan .claude/plans/lane-c-retrain.md §3): the S1 recipe
# (35k plain + 15k crowded 15:60 over the non-reserved themes) with the
# level mix re-weighted — L0-L10 as before, L11 (abort NOT_FOUND) added,
# L12-L18 (surface v2) at 40% — generated in 12 parallel shards (one
# sequential run measured ~2 tasks/s, i.e. 7 h), then the B3 open-data
# draw. Run from the repo root in Git Bash:  bash results/logs/gen_s2.sh
set -e
cd /c/code/covenant-agent
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8"
mkdir -p data/s2_shards
for i in 0 1 2 3 4 5 6 7; do
  python -m data.gen --levels "$LEVELS" --n 4375 --seed $((20260903 + i*10000000)) --drop-noops \
      --domains data/gen/themes --out data/s2_shards/plain_$i.jsonl > results/logs/gen_s2_plain_$i.log 2>&1 &
done
for i in 0 1 2 3; do
  python -m data.gen --levels "$LEVELS" --n 3750 --seed $((20260904 + i*10000000)) --drop-noops \
      --domains data/gen/themes --crowd 15:60 --out data/s2_shards/crowded_$i.jsonl > results/logs/gen_s2_crowded_$i.log 2>&1 &
done
wait
cat data/s2_shards/plain_*.jsonl > data/s2_plain.jsonl
cat data/s2_shards/crowded_*.jsonl > data/s2_crowded.jsonl
cat data/s2_plain.jsonl data/s2_crowded.jsonl > data/s2_tasks.jsonl
python -m baselines.qwen.make_sft --tasks data/s2_tasks.jsonl --domains data/gen/themes --out data/sft_s2_gen.jsonl

# B3: convert (cached after the first run) and draw to the real-session shape
python -m data.gen.convert_open --source hermes --out data/open_pairs/hermes_full.jsonl
python -m data.gen.convert_open --source toolace --limit 4000 --out data/open_pairs/toolace_4k.jsonl
python -m data.gen.convert_open --source glaive --limit 12000 --out data/open_pairs/glaive_12k.jsonl
python -m data.gen.draw_open --out data/open_pairs/b3_draw.jsonl

bash results/logs/finish_s2_corpus.sh
echo "=== gen_s2 done ==="
