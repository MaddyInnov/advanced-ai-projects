"""
Sentinel-LLM — PRODUCTION GPU fine-tune (NOT run on local CPU).
Author: Ashutosh Sahoo, M.Tech AI, NIT Silchar

Full QLoRA fine-tune of a real instruction LLM on dataset.jsonl using
Hugging Face transformers + PEFT + TRL SFTTrainer + bitsandbytes 4-bit.

Target model : Qwen/Qwen2.5-0.5B-Instruct (small, Kaggle/Colab friendly)
               — swap --model to TinyLlama/TinyLlama-1.1B-Chat-v1.0 or
                 Qwen/Qwen2.5-1.5B-Instruct for a stronger run.
Method       : QLoRA — 4-bit NF4 base + LoRA r=16, alpha=32 on
               q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
               Only ~0.5-1% of parameters are trained; adapter is saved
               separately and can be merged or served via vLLM / Ollama.

Run on Kaggle / Colab (T4 / P100 GPU):
  pip install -r requirements_gpu.txt
  python data_gen.py                      # if dataset.jsonl not present
  python train_lora_peft.py --epochs 3 --output_dir ./sentinel-qlora-out

After training:
  - Adapter weights land in <output_dir>/adapter (a few tens of MB)
  - Merge:  peft PeftModel.from_pretrained(base, adapter).merge_and_unload()
  - Eval on the JSONL test split with plain generation + exact label match
"""
import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description="Sentinel-LLM QLoRA fine-tune (GPU)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default="dataset.jsonl")
    p.add_argument("--output_dir", default="./sentinel-qlora-out")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def format_example(ex, tokenizer):
    """Chat-format one instruction example for SFT."""
    messages = [
        {"role": "system", "content": "You are Sentinel, a cybersecurity analyst assistant. Answer precisely and cite the threat type, MITRE technique or IOCs where relevant."},
        {"role": "user", "content": f"{ex['instruction']}\n\n{ex['input']}"},
        {"role": "assistant", "content": ex["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}


def main():
    args = parse_args()
    # --- GPU-only imports (kept inside main so this file parses on CPU too)
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. This script is for Kaggle/Colab — see README. For CPU use train.py.")

    rows = [json.loads(l) for l in open(args.data)]
    print(f"Loaded {len(rows)} examples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list([format_example(r, tokenizer) for r in rows])
    dataset = dataset.train_test_split(test_size=0.15, seed=args.seed)
    print(dataset)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        seed=args.seed,
        max_length=args.max_len,
    )
    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=dataset["train"], eval_dataset=dataset["test"],
        peft_config=peft_config, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Done. LoRA adapter saved to {args.output_dir}/adapter")
    print("Next: merge with PeftModel.merge_and_unload() or serve the adapter with vLLM.")


if __name__ == "__main__":
    main()
