"""R2 Condition B: LoRA SFT of Qwen3.5-0.8B / 2B on F4 pairs. GPU box only.

  pip install "transformers>=4.51" peft trl datasets accelerate bitsandbytes
  python train_b.py --model Qwen/Qwen3.5-0.8B --data sft_50k.jsonl --out lora_0.8b

Then merge + convert for the CPU eval harness (llama.cpp checkout needed):
  python train_b.py --merge lora_0.8b --model Qwen/Qwen3.5-0.8B --out merged_0.8b
  python llama.cpp/convert_hf_to_gguf.py merged_0.8b --outfile qwen3.5-0.8b-b.gguf --outtype q8_0

Loss is masked to assistant tokens (completion-only). Chat template applied
with enable_thinking=False to match eval-time prompting.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", help="LoRA dir to merge into --model")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="required at long --max-len: Qwen3.5's linear-attention "
                         "layers keep large per-layer state and OOM a 24 GB card "
                         "on a full-length row without it")
    ap.add_argument("--liger", action="store_true",
                    help="fused linear cross-entropy; the 248k vocab makes the "
                         "materialised logits the largest tensor in the step")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.merge:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="bfloat16")
        merged = PeftModel.from_pretrained(base, args.merge).merge_and_unload()
        merged.save_pretrained(args.out)
        AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
        print("merged ->", args.out)
        return

    import datasets
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Keep the dataset in messages form; TRL applies the chat template and
    # masks loss to assistant tokens. enable_thinking=False matches the
    # eval-time empty-think prompt.
    ds = datasets.load_dataset("json", data_files=args.data, split="train")

    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=100, bf16=True, max_length=args.max_len,
        logging_steps=50, save_strategy="steps", save_steps=500,
        save_total_limit=2, seed=args.seed,
        # NOTE: install flash-linear-attention on the training box. Qwen3.5 is
        # a hybrid model; without `fla` transformers falls back to a pure-torch
        # chunked gated-delta-rule that is launch-bound and ~8x slower end to
        # end (measured 1.05k vs 9.4k tok/s on a 4090, 2026-09-03).
        # NOTE: packing=True was tried for the mixed plain/crowded corpus
        # and was ~5x SLOWER on a 4090 (dense 3072-token batches every
        # step); plain padded batches win despite the waste. group_by_length
        # does not exist in TRL 1.12's SFTConfig.
        assistant_only_loss=True, report_to=[],
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=args.liger)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="bfloat16", attn_implementation="sdpa")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    print("LoRA saved ->", args.out)


if __name__ == "__main__":
    main()
