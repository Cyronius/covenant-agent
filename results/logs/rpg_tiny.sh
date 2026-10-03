# E-rpg on the tiny planner, and whether saying which way is open helps
# (.claude/plans/rpg-exits-perception.md part 4). $0: CPU for the tiny arms,
# LM Studio's Vulkan llama-server on the iGPU for the one GGUF reference row.
#
#   bash results/logs/rpg_tiny.sh [episodes] [max_turns]
set -u
cd "$(dirname "$0")/../.."
EP=${1:-6}; TURNS=${2:-20}; PORT=${PORT:-8078}
TINY=${TINY:-tiny:clt_RD}
RT="$HOME/.lmstudio/extensions/backends/llama.cpp-win-x86_64-vulkan-avx2-2.33.0"
GGUF=baselines/qwen/models/qwen3.5-0.8b-s5-q8.gguf

tiny () {  # tiny <tag> [flags...]
  local tag="$1"; shift
  echo "=== $tag $* ==="
  python -m harness.rpg_suite --planner tiny --tiny "$TINY" "$@" \
    --episodes "$EP" --max-turns "$TURNS" \
    --out "results/logs/${tag}_e_rpg.jsonl" 2>&1 \
    | tee "results/logs/${tag}_e_rpg.log" | tail -13
}

python -m harness.rpg_suite --planner oracle --exits --episodes "$EP" \
  --max-turns "$TURNS" --out results/logs/tiny_oracle_e_rpg.jsonl --quiet | tail -11

tiny tiny_T0
tiny tiny_T1 --exits
tiny tiny_T2 --exits --paths
tiny tiny_T1a1 --exits --max-actions 1

# G1: the previous demo default, full observation with exits, on the typed
# surface the demo served it
echo "=== tiny_G1 s5 GGUF --exits ==="
MSYS_NO_PATHCONV=1 "$RT/llama-server.exe" -m "$(pwd -W)/$GGUF" -c 4096 \
  -ngl 99 --host 127.0.0.1 --port "$PORT" > results/logs/_rpg_tiny_server.log 2>&1 &
pid=$!
for i in $(seq 1 60); do
  curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q ok && break
  sleep 5
done
python -m harness.rpg_suite --server "http://127.0.0.1:$PORT" --model "$GGUF" \
  --symbols typed --enums --kinds --exits --episodes "$EP" --max-turns "$TURNS" \
  --out results/logs/tiny_G1_e_rpg.jsonl 2>&1 \
  | tee results/logs/tiny_G1_e_rpg.log | tail -13
kill "$pid" 2>/dev/null || true
