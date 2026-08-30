# R2 Condition B runbook — LoRA on a rented GPU (a few hours total)

Everything is prepared locally; the GPU box only trains and converts.

## 1. Local prep (done / in progress on the dev machine)

- `data/train_50k.jsonl` — 50k F4 tasks (seed 31337, all levels, no-ops
  dropped, holdout worlds/tools excluded by the generator).
- `python -m baselines.qwen.make_sft --tasks data/train_50k.jsonl --out data/sft_50k.jsonl`
  → chat-format SFT pairs (~54k rows incl. PAUSE continuations; the
  continuation examples embed real register values via local sandbox runs).

## 2. GPU box (any single A10/L4/4090-class card, ~2-4 h)

Upload: `data/sft_50k.jsonl`, `baselines/qwen/train_b.py`.

```bash
pip install "transformers>=4.51" peft trl datasets accelerate
git clone --depth 1 https://github.com/ggml-org/llama.cpp   # for the converter
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# 0.8B (~1-2 h on an A10 at bs 8 x ga 4, 2 epochs)
python train_b.py --model Qwen/Qwen3.5-0.8B --data sft_50k.jsonl --out lora_08b
python train_b.py --merge lora_08b --model Qwen/Qwen3.5-0.8B --out merged_08b
python llama.cpp/convert_hf_to_gguf.py merged_08b --outfile qwen3.5-0.8b-condB-q8.gguf --outtype q8_0

# 2B (optional second arm, ~2x cost)
python train_b.py --model Qwen/Qwen3.5-2B --data sft_50k.jsonl --out lora_2b --batch 4 --grad-accum 8
python train_b.py --merge lora_2b --model Qwen/Qwen3.5-2B --out merged_2b
python llama.cpp/convert_hf_to_gguf.py merged_2b --outfile qwen3.5-2b-condB-q8.gguf --outtype q8_0
```

Download the `*-condB-q8.gguf` file(s) back to
`baselines/qwen/models/`.

**Known converter bug (hit 2026-08-30):** converting a Qwen3.5 checkpoint
that was loaded via `AutoModelForCausalLM` (which drops the NextN/MTP head)
still writes `qwen35.block_count = 25` and `qwen35.nextn_predict_layers = 1`
— but no `blk.24.nextn.*` tensors — so every runtime rejects the file with
`check_tensor_dims: tensor ... not found`. Fix in place (no reconvert):

```bash
python -m gguf.scripts.gguf_set_metadata FILE.gguf qwen35.block_count 24 --force
python -m gguf.scripts.gguf_set_metadata FILE.gguf qwen35.nextn_predict_layers 0 --force
```

**Test-load the GGUF (locally or on the pod) BEFORE deleting the pod**, and
pull the LoRA adapter dir too — the merged HF model is too big to keep, the
adapter is not.

## 3. Local eval (same harness as Condition A)

```bash
python -m baselines.qwen.run_a --model baselines/qwen/models/qwen3.5-0.8b-condB-q8.gguf \
    --tasks data/r1_tasks.jsonl --out results/r2_b_0.8b_grammar.jsonl
python -m baselines.qwen.run_a --model baselines/qwen/models/qwen3.5-0.8b-condB-q8.gguf \
    --tasks data/holdout/r2_holdout.jsonl --out results/r2_b_0.8b_holdout.jsonl
```

Record in `results/R2.md`; the by-level goal_success table is the R3 bar.
