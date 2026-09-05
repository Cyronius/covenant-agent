# LFM2.5-350M on the S3 corpus — pod side. The size comparison: same
# corpus, same recipe shape, same eval, a 350M hybrid-conv model against
# Qwen3.5-0.8B. If "size is the constraint" were true this run is where it
# would show.
#
# Differences from train_s3.sh, all forced by the architecture:
#   - LoRA targets are LFM2's module names (attention q/k/v/out_proj, the
#     short-conv in_proj, FFN w1/w2/w3); lm_head stays frozen.
#   - No flash-linear-attention (Qwen3.5's gated-delta kernel) and no liger
#     (a 65k vocab does not need the fused CE). Its chat template carries
#     {% generation %} markers, so TRL's assistant-only loss works as-is
#     (probed 2026-09-05: 2-step dry run, loss 3.5 -> 2.0).
#   - The cap is measured with LFM's own tokenizer: 65k vocab vs Qwen's
#     248k means the same rows are longer here.
#   - No vocab prune (prune_vocab.py is Qwen-specific and a 65k vocab has
#     little to prune) and no NextN metadata fix (Qwen-only converter bug).
#     The comparison row is therefore against qwen3.5-0.8b-s3-q8.gguf, the
#     UNPRUNED S3 -- pruning was shown behaviour-neutral on S2 anyway.
#   - GGUF conversion runs with gguf-py on PYTHONPATH rather than pip
#     installing llama.cpp's requirements, which would replace torch with a
#     CPU wheel (the S2 lesson).
#
# Upload: data/sft_s3.jsonl  baselines/qwen/train_b.py   Then: bash train_lfm.sh
# Download back: lora_lfm_top.tar.gz, lfm2.5-350m-s3-q8.gguf
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
python -c "import torch,torchvision;print('torch=='+torch.__version__);print('torchvision=='+torchvision.__version__)" > pipc.txt
pip install -q -c pipc.txt "transformers>=5.0" peft "trl==1.12.0" datasets accelerate "fsspec<=2026.6.0"
python -c "import torch,sys;print('torch',torch.__version__);sys.exit(0 if torch.cuda.is_available() else 1)"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp

MID=LiquidAI/LFM2.5-350M
# Cap from the corpus max under THIS tokenizer, rounded up to 512.
CAP=$(python - <<'PY'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")
mx = 0; n = 0
for line in open("sft_s3.jsonl", encoding="utf-8"):
    r = json.loads(line); n += 1
    mx = max(mx, len(tok(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False)["input_ids"]))
print(((mx + 511) // 512) * 512)
PY
)
echo "train cap (LFM tokenizer): $CAP"

# 1. LoRA. 350M leaves room: try bs 4 x ga 8 (effective 32); OOM falls back
#    to bs 2 x ga 16, then bs 1 x ga 32 -- same effective batch throughout.
TM="q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3"
train() {
  python train_b.py --model "$MID" --data sft_s3.jsonl --out lora_lfm \
      --max-len "$CAP" --batch "$1" --grad-accum "$2" --epochs 2 --lr 2e-4 --rank 32 \
      --grad-checkpointing --target-modules "$TM"
}
ok=0
for cfg in "4 8" "2 16" "1 32"; do
  set -- $cfg
  if train "$1" "$2" 2>&1 | tee "train_lfm_bs$1.log"; then ok=1; break; fi
  if grep -q "OutOfMemoryError\|CUDA out of memory" "train_lfm_bs$1.log"; then
    echo "=== OOM at bs $1; falling back ==="; rm -rf lora_lfm
  else
    echo "train failed for a reason other than OOM"; exit 1
  fi
done
[ "$ok" = 1 ] || { echo "all batch sizes OOM"; exit 1; }
python train_b.py --merge lora_lfm --model "$MID" --out merged_lfm_hf

# 2. GGUF (no prune). gguf-py on the path, nothing pip-installed.
PYTHONPATH=llama.cpp/gguf-py python llama.cpp/convert_hf_to_gguf.py merged_lfm_hf \
    --outfile lfm2.5-350m-s3-q8.gguf --outtype q8_0
tar -czf lora_lfm_top.tar.gz -C lora_lfm adapter_config.json adapter_model.safetensors
echo "=== train_lfm done: lora_lfm/ lfm2.5-350m-s3-q8.gguf ==="
