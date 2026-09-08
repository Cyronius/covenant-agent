# S4 — pod side. A fresh LoRA from the base on the S4 corpus (gen_s4.sh):
# the S3 recipe, unchanged, on the first corpus that carries the spec 0.4.0
# surface. Not a continuation — same reason as S3.
#
#   bash train_s4.sh            # typed arm   (sft_s4.jsonl  -> lora_s4)
#   bash train_s4.sh classic    # control arm (sft_s4c.jsonl -> lora_s4c)
#
# The two arms are the same tasks from the same seeds; only the symbol
# table differs (C0… vs S0/N0/B0/D0/I0 with kinds and schema enums). Run
# both and the difference is the typed letters and nothing else — that is
# the claim-2 measurement typed-symbols.md §3 asks for and the one R5 could
# not settle on an untuned model.
#
# Upload to the pod's working dir:
#   data/sft_s4.jsonl  data/s4_used_token_ids.json  data/s4_train_cap.txt
#   (and/or the s4c triple for the control arm)
#   baselines/qwen/train_b.py  baselines/qwen/prune_vocab.py  baselines/qwen/convert_pruned.py
# Download back BEFORE deleting the pod: lora_s4*/ (tarred below),
# merged_s4*_pruned_hf/tokenizer.json, both GGUFs.
set -e
ARM="${1:-typed}"
case "$ARM" in
  typed)   TAG=s4 ;;
  classic) TAG=s4c ;;
  *) echo "usage: train_s4.sh [typed|classic]"; exit 2 ;;
esac
export PIP_BREAK_SYSTEM_PACKAGES=1
python -c "import torch,torchvision;print('torch=='+torch.__version__);print('torchvision=='+torchvision.__version__)" > pipc.txt
pip install -q -c pipc.txt "transformers>=5.0" peft "trl==1.12.0" datasets accelerate "fsspec<=2026.6.0"
pip install -q -c pipc.txt flash-linear-attention liger-kernel
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" || { echo "fla kernel missing -- training would be ~8x slower"; exit 1; }
python -c "import torch,sys;print('torch',torch.__version__);sys.exit(0 if torch.cuda.is_available() else 1)"

CAP=$(cat ${TAG}_train_cap.txt)
echo "arm: $ARM | corpus: sft_${TAG}.jsonl | train cap: $CAP (corpus max rounded up to 512)"

# 1. LoRA. bs 2 x ga 16 = effective 32 (the S1/S2/S3 batch); on an OOM exit
#    fall back to bs 1 x ga 32 -- same effective batch, same recipe.
train() {
  python train_b.py --model Qwen/Qwen3.5-0.8B --data sft_${TAG}.jsonl --out lora_${TAG} \
      --max-len "$CAP" --batch "$1" --grad-accum "$2" --epochs 2 --lr 2e-4 --rank 32 \
      --grad-checkpointing --liger
}
if ! train 2 16 2>&1 | tee train_${TAG}_bs2.log; then
  if grep -q "OutOfMemoryError\|CUDA out of memory" train_${TAG}_bs2.log; then
    echo "=== OOM at bs 2; retrying bs 1 x ga 32 ==="
    rm -rf lora_${TAG}
    train 1 32 2>&1 | tee train_${TAG}_bs1.log
  else
    echo "train failed for a reason other than OOM"; exit 1
  fi
fi
python train_b.py --merge lora_${TAG} --model Qwen/Qwen3.5-0.8B --out merged_${TAG}_hf

# 2. Prune with this corpus's keep-set (the S3 set UNION every token the
#    corpus uses -- the typed letters are new sequences).
python prune_vocab.py --src merged_${TAG}_hf --out merged_${TAG}_pruned_hf --floor 40000 \
    --used-ids ${TAG}_used_token_ids.json --check-corpus sft_${TAG}.jsonl

# 3. GGUF. Both files need the NextN metadata fix.
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
export LLAMA_CPP="$PWD/llama.cpp"
python convert_pruned.py merged_${TAG}_pruned_hf --outfile qwen3.5-0.8b-${TAG}-pruned-q8.gguf --outtype q8_0
python llama.cpp/convert_hf_to_gguf.py merged_${TAG}_hf --outfile qwen3.5-0.8b-${TAG}-q8.gguf --outtype q8_0
for f in qwen3.5-0.8b-${TAG}-pruned-q8.gguf qwen3.5-0.8b-${TAG}-q8.gguf; do
    python -m gguf.scripts.gguf_set_metadata "$f" qwen35.block_count 24 --force
    python -m gguf.scripts.gguf_set_metadata "$f" qwen35.nextn_predict_layers 0 --force
done
tar -czf lora_${TAG}_top.tar.gz -C lora_${TAG} adapter_config.json adapter_model.safetensors
echo "=== train_s4 ($ARM) done: lora_${TAG}/ merged_${TAG}_pruned_hf/ *.gguf ==="
