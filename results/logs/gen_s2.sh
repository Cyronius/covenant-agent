# S2 corpus (plan .claude/plans/lane-c-retrain.md §3): the S1 recipe
# (35k plain + 15k crowded 15:60 over the non-reserved themes) with the
# level mix re-weighted — L0-L10 as before, L11 (abort NOT_FOUND) added,
# L12-L18 (surface v2) at 40% — plus the B3 open-data draw.
# Run from the repo root in Git Bash:  bash results/logs/gen_s2.sh
set -e
cd /c/code/covenant-agent
LEVELS="0:5,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5,9:5,10:5,11:5,12:5.7,13:5.7,14:5.7,15:5.7,16:5.7,17:5.7,18:5.8"
python -m data.gen --levels "$LEVELS" --n 35000 --seed 20260903 --drop-noops \
    --domains data/gen/themes --out data/s2_plain.jsonl
python -m data.gen --levels "$LEVELS" --n 15000 --seed 20260904 --drop-noops \
    --domains data/gen/themes --crowd 15:60 --out data/s2_crowded.jsonl
cat data/s2_plain.jsonl data/s2_crowded.jsonl > data/s2_tasks.jsonl
python -m baselines.qwen.make_sft --tasks data/s2_tasks.jsonl --domains data/gen/themes --out data/sft_s2_gen.jsonl
python -m data.gen.draw_open --out data/open_pairs/b3_draw.jsonl
python - <<'EOF'
import json, random
rows = [json.loads(l) for l in open('data/sft_s2_gen.jsonl', encoding='utf-8')]
rows += [{"messages": json.loads(l)["messages"], "source": "b3"} for l in open('data/open_pairs/b3_draw.jsonl', encoding='utf-8')]
random.Random(20260903).shuffle(rows)
with open('data/sft_s2.jsonl', 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps({"messages": r["messages"]}) + "\n")
print(len(rows), 'SFT rows -> data/sft_s2.jsonl')
EOF
echo "=== gen_s2 done ==="
