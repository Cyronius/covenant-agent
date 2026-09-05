# S3 — pod side. A fresh LoRA from the base on the S3 corpus (gen_s3.sh):
# the S2 recipe with the crowded tail restored, the cap set from the
# corpus's measured max, and the 900 real-turn rows folded in from the
# start. Not a continuation: S2R's two faults (E-known forgetting,
# over-abstention on human-written text) were both artifacts of warm-
# starting on a concentrated mix.
#
# Upload to the pod's working dir:
#   data/sft_s3.jsonl  data/s3_used_token_ids.json  data/s3_train_cap.txt
#   baselines/qwen/train_b.py  baselines/qwen/prune_vocab.py  baselines/qwen/convert_pruned.py
# Then:  bash train_s3.sh
# Download back: lora_s3/ (the adapter -- S2R's was lost by deleting the
# pod first), merged_s3_pruned_hf/tokenizer.json, both GGUFs.
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
python -c "import torch,torchvision;print('torch=='+torch.__version__);print('torchvision=='+torchvision.__version__)" > pipc.txt
pip install -q -c pipc.txt "transformers>=5.0" peft "trl==1.12.0" datasets accelerate "fsspec<=2026.6.0"
pip install -q -c pipc.txt flash-linear-attention liger-kernel
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" || { echo "fla kernel missing -- training would be ~8x slower"; exit 1; }
python -c "import torch,sys;print('torch',torch.__version__);sys.exit(0 if torch.cuda.is_available() else 1)"

CAP=$(cat s3_train_cap.txt)
echo "train cap: $CAP (corpus max rounded up to 512; nothing is truncated)"

# 1. LoRA. bs 2 x ga 16 = effective 32 (the S1/S2 batch). A longer cap can
#    OOM a 24 GB card at bs 2 on the longest crowded rows, so on an OOM
#    exit fall back to bs 1 x ga 32 -- same effective batch, same recipe.
train() {
  python train_b.py --model Qwen/Qwen3.5-0.8B --data sft_s3.jsonl --out lora_s3 \
      --max-len "$CAP" --batch "$1" --grad-accum "$2" --epochs 2 --lr 2e-4 --rank 32 \
      --grad-checkpointing --liger
}
if ! train 2 16 2>&1 | tee train_s3_bs2.log; then
  if grep -q "OutOfMemoryError\|CUDA out of memory" train_s3_bs2.log; then
    echo "=== OOM at bs 2; retrying bs 1 x ga 32 ==="
    rm -rf lora_s3
    train 1 32 2>&1 | tee train_s3_bs1.log
  else
    echo "train failed for a reason other than OOM"; exit 1
  fi
fi
python train_b.py --merge lora_s3 --model Qwen/Qwen3.5-0.8B --out merged_s3_hf

# 2. Prune with the S3 keep-set (S2 set UNION this corpus's tokens).
python prune_vocab.py --src merged_s3_hf --out merged_s3_pruned_hf --floor 40000 \
    --used-ids s3_used_token_ids.json --check-corpus data/sft_s3.jsonl

# 3. GGUF. Both files need the NextN metadata fix.
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
export LLAMA_CPP="$PWD/llama.cpp"
python convert_pruned.py merged_s3_pruned_hf --outfile qwen3.5-0.8b-s3-pruned-q8.gguf --outtype q8_0
python llama.cpp/convert_hf_to_gguf.py merged_s3_hf --outfile qwen3.5-0.8b-s3-q8.gguf --outtype q8_0
for f in qwen3.5-0.8b-s3-pruned-q8.gguf qwen3.5-0.8b-s3-q8.gguf; do
    python -m gguf.scripts.gguf_set_metadata "$f" qwen35.block_count 24 --force
    python -m gguf.scripts.gguf_set_metadata "$f" qwen35.nextn_predict_layers 0 --force
done
# The adapter, small enough to keep: tar it so one scp brings it back.
tar -czf lora_s3_top.tar.gz -C lora_s3 adapter_config.json adapter_model.safetensors
echo "=== train_s3 done: lora_s3/ merged_s3_pruned_hf/ *.gguf ==="
