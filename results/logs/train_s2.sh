# S2 retrain — pod side (plan .claude/plans/lane-c-retrain.md §4).
# Runs on a rented single-GPU box (RunPod 4090 class). Same recipe as S1
# (baselines/qwen/CONDITION_B.md) with a longer sequence cap: the v2
# surface (18 native tools) plus crowding makes rows long.
#
# Upload to the pod's working dir:
#   data/sft_s2.jsonl                       (generated + B3 draw; no user text)
#   data/s2_used_token_ids.json             (keep-set, computed locally by
#                                            prune_vocab --dump-used; no text)
#   baselines/qwen/train_b.py  baselines/qwen/prune_vocab.py  baselines/qwen/convert_pruned.py
# Then:  bash train_s2.sh
# Download back: lora_s2/ , merged_s2_pruned_hf/tokenizer.json (for local
# identity checks), qwen3.5-0.8b-s2-pruned-q8.gguf and qwen3.5-0.8b-s2-q8.gguf.
set -e
pip install -q "transformers>=4.51" peft trl datasets accelerate
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# 1. LoRA on the full base. Measured on sft_s2.jsonl (3000-row sample):
#    tokens/row p50 1632, p95 5346, max 6459; 19% of rows exceed 4096 (the
#    crowded ones: 18 native tools + up to 60 distractors). TRL truncates
#    on the right, so a row over the cap loses its assistant turn and
#    contributes nothing under assistant_only_loss; max-len 6144 keeps
#    p95 and more. bs 2 x ga 16 = the S1 effective batch (32) within 24 GB.
python train_b.py --model Qwen/Qwen3.5-0.8B --data sft_s2.jsonl --out lora_s2 \
    --max-len 6144 --batch 2 --grad-accum 16 --epochs 2 --lr 2e-4 --rank 32
python train_b.py --merge lora_s2 --model Qwen/Qwen3.5-0.8B --out merged_s2_hf

# 2. Prune the merged vocab with the precomputed keep-set (B4 procedure).
python prune_vocab.py --src merged_s2_hf --out merged_s2_pruned_hf --floor 40000 \
    --used-ids s2_used_token_ids.json

# 3. GGUF: pruned Q8 and an unpruned Q8 for the A/B (IQ4 is quantized
#    locally from the pruned Q8 with the existing imatrix, if wanted).
python convert_pruned.py merged_s2_pruned_hf --outfile qwen3.5-0.8b-s2-pruned-q8.gguf --outtype q8_0
python llama.cpp/convert_hf_to_gguf.py merged_s2_hf --outfile qwen3.5-0.8b-s2-q8.gguf --outtype q8_0
python -m gguf.scripts.gguf_set_metadata qwen3.5-0.8b-s2-q8.gguf qwen35.block_count 24 --force
python -m gguf.scripts.gguf_set_metadata qwen3.5-0.8b-s2-q8.gguf qwen35.nextn_predict_layers 0 --force
echo "=== train_s2 done: lora_s2/ merged_s2_pruned_hf/ *.gguf ==="
