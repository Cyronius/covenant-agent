# S2R continuation — pod side. Teaches the real request distribution on top
# of the S2 adapter instead of retraining from scratch.
#
# Why: S2 scores 106/106 on its own held-out synthetic levels (24/24 correct
# abstains, 82/82 correct actions, 0 false abstains, all four abort reasons
# used) and 0/73 correct abstains on real turns. The capability is present;
# the trigger is keyed to generator phrasing. So this is a small, cheap pass
# over real requests, not another 12.5-hour run.
#
# Corpus: data/sft_s2r.jsonl = 900 verified real-turn rows (x4, 25%) mixed
# with a 10,800-row replay slice of sft_s2.jsonl (75%) so the curriculum is
# not forgotten. Real rows come from harness/real_train_build.py: every
# target was executed through the eval harness and kept only if it scored
# the metric it teaches. 14,400 rows, ~900 steps, ~3.5 h on a 4090 (~$1.20).
#
# *** Contains verbatim (strict-clean, PII-scanned) customer request text.
# *** Requires the owner's sign-off before it goes on a rented box.
#
# Upload to the pod's working dir:
#   data/sft_s2r.jsonl                      data/s2_used_token_ids.json
#   baselines/qwen/models/lora_s2/          (the S2 adapter to continue)
#   baselines/qwen/train_b.py  baselines/qwen/prune_vocab.py  baselines/qwen/convert_pruned.py
# Then:  bash train_s2r.sh
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
python -c "import torch,torchvision;print('torch=='+torch.__version__);print('torchvision=='+torchvision.__version__)" > pipc.txt
pip install -q -c pipc.txt "transformers>=5.0" peft "trl==1.12.0" datasets accelerate "fsspec<=2026.6.0"
pip install -q -c pipc.txt flash-linear-attention liger-kernel
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" || { echo "fla kernel missing -- training would be ~8x slower"; exit 1; }
python -c "import torch,sys;print('torch',torch.__version__);sys.exit(0 if torch.cuda.is_available() else 1)"

# 1. Continue lora_s2 on the mixed corpus. Lower LR than the S2 run (1e-4 vs
#    2e-4): the adapter already fits the curriculum and only needs to move
#    far enough to fire ABORT on real phrasing. Same effective batch (32).
python train_b.py --model Qwen/Qwen3.5-0.8B --data sft_s2r.jsonl --out lora_s2r \
    --init-adapter lora_s2 --max-len 6144 --batch 2 --grad-accum 16 \
    --epochs 2 --lr 1e-4 --rank 32 --grad-checkpointing --liger
python train_b.py --merge lora_s2r --model Qwen/Qwen3.5-0.8B --out merged_s2r_hf

# 2. Prune with the S2 keep-set (unchanged: the real rows add no new tokens
#    that the keep-set's real-session sources did not already cover).
python prune_vocab.py --src merged_s2r_hf --out merged_s2r_pruned_hf --floor 40000 \
    --used-ids s2_used_token_ids.json --check-corpus data/sft_s2r.jsonl

# 3. GGUF. Both files need the NextN metadata fix, not just one.
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
export LLAMA_CPP="$PWD/llama.cpp"
python convert_pruned.py merged_s2r_pruned_hf --outfile qwen3.5-0.8b-s2r-pruned-q8.gguf --outtype q8_0
python llama.cpp/convert_hf_to_gguf.py merged_s2r_hf --outfile qwen3.5-0.8b-s2r-q8.gguf --outtype q8_0
for f in qwen3.5-0.8b-s2r-pruned-q8.gguf qwen3.5-0.8b-s2r-q8.gguf; do
    python -m gguf.scripts.gguf_set_metadata "$f" qwen35.block_count 24 --force
    python -m gguf.scripts.gguf_set_metadata "$f" qwen35.nextn_predict_layers 0 --force
done
echo "=== train_s2r done: lora_s2r/ merged_s2r_pruned_hf/ *.gguf ==="
