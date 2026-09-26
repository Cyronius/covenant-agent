# S6's decoyed and flip exams for the two S5-trained planners, on an
# evaluation-only cache with data_cache_s5g's exact layout (19 of 4,400 rows
# with more than 54 tools left out). Separate archive: run_s6split.sh's tar
# may run before these finish.
cd "$(dirname "$0")"
for r in s5g_A0 s5g_SPt; do
  python evaluate.py --generate --ckpt runs/$r/best.pt --cache data_cache_s5g_s6dec --split holdout \
      --limit 2000 --id-contains +decoy --gen-out out/${r}_s6decoy.jsonl > out/${r}_s6decoy.log 2>&1 &
  python evaluate.py --generate --ckpt runs/$r/best.pt --cache data_cache_s5g_s6dec --split holdout \
      --limit 2000 --id-contains +flip --gen-out out/${r}_s6flip.jsonl > out/${r}_s6flip.log 2>&1 &
done
wait
tar czf s6dec_all.tar.gz out/*_s6decoy.* out/*_s6flip.* run_s6dec.sh
ls -l s6dec_all.tar.gz
echo "=== run_s6dec done $(date)"
