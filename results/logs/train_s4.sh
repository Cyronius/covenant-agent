# S4 — pod side. A fresh LoRA from a base on the S4 corpus (gen_s4.sh):
# the S3 recipe, unchanged, on the first corpus that carries the spec 0.4.0
# surface. Not a continuation — same reason as S3.
#
#   bash train_s4.sh                        # 0.8B, typed   (sft_s4.jsonl)
#   bash train_s4.sh classic                # 0.8B, classic (sft_s4c.jsonl)
#   bash train_s4.sh typed minicpm2b        # MiniCPM5-2B, typed
#   bash train_s4.sh typed minicpm1b        # MiniCPM5-1B, typed
#
# Two questions, two axes, and they do NOT need a full cross:
#
#   surface (typed vs classic)  ask it once, on the 0.8B. Same tasks from
#                               the same seeds; only the symbol table moves,
#                               so the pair isolates the typed letters —
#                               the claim-2 measurement typed-symbols.md §3
#                               asks for and the one R5 could not settle on
#                               an untuned model.
#   base (0.8B / 2B / 1B)       the S2.md bake-off. Run these on whichever
#                               surface the 0.8B pair picks; training both
#                               MiniCPM arms on both surfaces would be four
#                               runs to answer a question the first pair
#                               already answered.
#
# So: the 0.8B pair first, then the two MiniCPM arms on the winner. The
# untuned smoke (7b64050) does not rank these bases — the 2B managed 3/25
# and the 1B 0/25 in context, and LFM-350M went from 0/25 untuned to 85% of
# S3 tuned, so in-context ability is not the signal. The retrain is.
#
# Upload to the pod's working dir:
#   data/sft_s4.jsonl  data/s4_used_token_ids.json  data/s4_train_cap.txt
#   (and/or the s4c triple for the control arm)
#   baselines/qwen/train_b.py  baselines/qwen/prune_vocab.py  baselines/qwen/convert_pruned.py
# Download back BEFORE deleting the pod: lora_*_top.tar.gz, the pruned
# tokenizer.json, both GGUFs.
set -e
ARM="${1:-typed}"
BASE="${2:-qwen08b}"
case "$ARM" in
  typed)   TAG=s4 ;;
  classic) TAG=s4c ;;
  *) echo "usage: train_s4.sh [typed|classic] [qwen08b|minicpm2b|minicpm1b]"; exit 2 ;;
esac
case "$BASE" in
  qwen08b)   MODEL=Qwen/Qwen3.5-0.8B;      SUF="";     GGUF=qwen3.5-0.8b ;;
  minicpm2b) MODEL=openbmb/MiniCPM5-2B;    SUF="_m2b"; GGUF=minicpm5-2b ;;
  minicpm1b) MODEL=openbmb/MiniCPM5-1B;    SUF="_m1b"; GGUF=minicpm5-1b ;;
  *) echo "unknown base $BASE"; exit 2 ;;
esac
RUN="${TAG}${SUF}"
export PIP_BREAK_SYSTEM_PACKAGES=1
python -c "import torch,torchvision;print('torch=='+torch.__version__);print('torchvision=='+torchvision.__version__)" > pipc.txt
pip install -q -c pipc.txt "transformers>=5.0" peft "trl==1.12.0" datasets accelerate "fsspec<=2026.6.0"
pip install -q -c pipc.txt flash-linear-attention liger-kernel
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" || { echo "fla kernel missing -- training would be ~8x slower"; exit 1; }
python -c "import torch,sys;print('torch',torch.__version__);sys.exit(0 if torch.cuda.is_available() else 1)"

# The LoRA targets default to the Qwen attention+MLP set. MiniCPM5 is
# Llama-shaped and should match, but check before burning an hour on an
# adapter attached to nothing:
if [ "$BASE" != "qwen08b" ]; then
  python - "$MODEL" <<'EOF'
import sys, collections
from transformers import AutoConfig, AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype="bfloat16")
names = collections.Counter(n.split('.')[-1] for n, _ in m.named_modules())
want = "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj".split()
missing = [w for w in want if not names.get(w)]
print("module leaf names:", sorted(k for k, v in names.items() if v > 4))
if missing:
    sys.exit(f"FAIL: {sys.argv[1]} has no {missing} -- pass --target-modules")
print("target modules OK")
EOF
fi

CAP=$(cat ${TAG}_train_cap.txt)
echo "arm: $ARM | base: $BASE ($MODEL) | corpus: sft_${TAG}.jsonl | cap: $CAP"

# 1. LoRA. bs 2 x ga 16 = effective 32 (the S1/S2/S3 batch); on an OOM exit
#    fall back to bs 1 x ga 32 -- same effective batch, same recipe.
train() {
  python train_b.py --model "$MODEL" --data sft_${TAG}.jsonl --out lora_${RUN} \
      --max-len "$CAP" --batch "$1" --grad-accum "$2" --epochs 2 --lr 2e-4 --rank 32 \
      --grad-checkpointing --liger
}
if ! train 2 16 2>&1 | tee train_${RUN}_bs2.log; then
  if grep -q "OutOfMemoryError\|CUDA out of memory" train_${RUN}_bs2.log; then
    echo "=== OOM at bs 2; retrying bs 1 x ga 32 ==="
    rm -rf lora_${RUN}
    train 1 32 2>&1 | tee train_${RUN}_bs1.log
  else
    echo "train failed for a reason other than OOM"; exit 1
  fi
fi
python train_b.py --merge lora_${RUN} --model "$MODEL" --out merged_${RUN}_hf
tar -czf lora_${RUN}_top.tar.gz -C lora_${RUN} adapter_config.json adapter_model.safetensors

# 2. Vocab pruning is Qwen-only here on purpose: the keep-set in
#    ${TAG}_used_token_ids.json was built with the Qwen tokenizer, so it says
#    nothing about which MiniCPM ids the corpus uses. Pruning is a deployment
#    size win, not part of the bake-off — if a MiniCPM base wins, build its
#    keep-set with its own tokenizer and prune then.
if [ "$BASE" = "qwen08b" ]; then
  python prune_vocab.py --src merged_${RUN}_hf --out merged_${RUN}_pruned_hf --floor 40000 \
      --used-ids ${TAG}_used_token_ids.json --check-corpus sft_${TAG}.jsonl
fi

# 3. GGUF. The Qwen files need the NextN metadata fix; other bases do not.
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
export LLAMA_CPP="$PWD/llama.cpp"
OUTS=""
if [ "$BASE" = "qwen08b" ]; then
  python convert_pruned.py merged_${RUN}_pruned_hf --outfile ${GGUF}-${RUN}-pruned-q8.gguf --outtype q8_0
  OUTS="${GGUF}-${RUN}-pruned-q8.gguf"
fi
python llama.cpp/convert_hf_to_gguf.py merged_${RUN}_hf --outfile ${GGUF}-${RUN}-q8.gguf --outtype q8_0
OUTS="$OUTS ${GGUF}-${RUN}-q8.gguf"
if [ "$BASE" = "qwen08b" ]; then
  for f in $OUTS; do
    python -m gguf.scripts.gguf_set_metadata "$f" qwen35.block_count 24 --force
    python -m gguf.scripts.gguf_set_metadata "$f" qwen35.nextn_predict_layers 0 --force
  done
fi
echo "=== train_s4 ($ARM, $BASE) done: lora_${RUN}/ $OUTS ==="
