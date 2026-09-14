# The per-turn decision baseline for one checkpoint, on the three episode
# worlds (results/FAMILIES.md §3). Same Vulkan server path as rpg_gpu.sh:
# ~3 s a turn against the iGPU instead of ~23 s in process.
#
#   bash results/logs/perturn.sh <gguf> [typed|classic] [episodes]
#
# The number to read is the whole-turn score. First-tool agreement is
# gameable - S3 scores 98% of it on the dungeon while winning nothing,
# because it opens with the oracle's move and then keeps going.
set -u
cd "$(dirname "$0")/../.."
M=${1:-baselines/qwen/models/qwen3.5-0.8b-s4-q8-fixed.gguf}
ARM=${2:-typed}
EP=${3:-3}
PORT=${PORT:-8079}
RT="$HOME/.lmstudio/extensions/backends/llama.cpp-win-x86_64-vulkan-avx2-2.33.0"
TAG=$(basename "$M" .gguf)
case "$ARM" in
  typed)   SURFACE="--symbols typed --enums --kinds" ;;
  classic) SURFACE="" ;;
  *) echo "usage: perturn.sh <gguf> [typed|classic] [episodes]"; exit 2 ;;
esac

MSYS_NO_PATHCONV=1 "$RT/llama-server.exe" -m "$(pwd -W)/$M" -c 4096 \
  -ngl 99 --host 127.0.0.1 --port "$PORT" > results/logs/_perturn_server.log 2>&1 &
PID=$!
for i in $(seq 1 60); do
  curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q ok && break
  sleep 5
done
for W in rpg house app_coursebuilder; do
  echo "=== $W ($TAG, $ARM) ==="
  python -m harness.rpg_suite --server "http://127.0.0.1:$PORT" --model "$M" \
    --world "$W" $SURFACE --episodes "$EP" --quiet \
    --out "results/logs/${TAG}_perturn_${W}.jsonl" 2>&1 \
    | tee "results/logs/${TAG}_perturn_${W}.log" | tail -10
done
kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
echo "=== perturn done ==="
