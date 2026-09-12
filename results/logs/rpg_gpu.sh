# E-rpg on the tuned checkpoints, generated through LM Studio's bundled
# Vulkan llama-server against the AMD iGPU. This box has no CUDA card and the
# installed llama-cpp-python is a CPU build, so in-process generation runs at
# ~23 s a turn; this path runs at ~3 s and returns the same programs token
# for token (checked on turns 0-2 of the S4 arm).
#
#   bash results/logs/rpg_gpu.sh [episodes] [max_turns]
#
# Each arm runs on the surface its checkpoint was trained on - a typed model
# scored on classic prompts looks broadly broken (results/R6.md §0). The
# unpruned GGUFs are the ones scored here, per RPG.md: the pruned vocabulary
# never saw the map glyphs or words like "goblin".
set -u
cd "$(dirname "$0")/../.."
EP=${1:-6}; TURNS=${2:-20}; PORT=${PORT:-8078}
RT="$HOME/.lmstudio/extensions/backends/llama.cpp-win-x86_64-vulkan-avx2-2.33.0"
M=baselines/qwen/models
LOG=results/logs/_rpg_gpu_server.log

arm () {  # arm <gguf> <tag> [surface flags...]
  local gguf="$1" tag="$2"; shift 2
  echo "=== $tag ${*:-classic} ==="
  MSYS_NO_PATHCONV=1 "$RT/llama-server.exe" -m "$(pwd -W)/$gguf" -c 4096 \
    -ngl 99 --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 60); do
    curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q ok && break
    sleep 5
  done
  python -m harness.rpg_suite --server "http://127.0.0.1:$PORT" \
    --model "$gguf" "$@" --episodes "$EP" --max-turns "$TURNS" \
    --out "results/logs/${tag}_e_rpg.jsonl" 2>&1 \
    | tee "results/logs/${tag}_e_rpg.log" | tail -14
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 3
}

python -m harness.rpg_suite --planner oracle --episodes "$EP" \
  --max-turns "$TURNS" --out results/logs/oracle_e_rpg.jsonl --quiet 2>&1 | tail -3
arm $M/qwen3.5-0.8b-s4-q8-fixed.gguf qwen3.5-0.8b-s4-q8-fixed --symbols typed --enums --kinds
arm $M/qwen3.5-0.8b-s3-q8.gguf qwen3.5-0.8b-s3-q8
arm $M/qwen3.5-0.8b-s2r-q8.gguf qwen3.5-0.8b-s2r-q8
echo "=== rpg_gpu done ==="
