#!/usr/bin/env python
"""
Track B: QLoRA supervised fine-tuning (SFT) for Dzongkha.

Stages (beginner map):
  1) Load YAML config + CLI overrides
  2) Load tokenizer + 4-bit quantized Qwen2.5-7B-Instruct
  3) Attach LoRA adapters on attention/MLP projection layers
  4) Format each JSONL row with Qwen chat template
  5) Train with TRL SFTTrainer (short smoke run by default)
  6) Save LoRA adapter + tokenizer under output_dir

Does NOT merge adapters into the base model (keeps VRAM/disk small).
First run downloads the base model from Hugging Face (several GB on disk).
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA SFT for Dzongkha (Qwen2.5-7B)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--max_steps", type=int, default=None, help="Override max_steps (-1 = full epochs)")
    p.add_argument("--data_path", type=str, default=None, help="Override training JSONL path")
    p.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def format_example(example: dict, tokenizer) -> dict:
    """Build a single chat-formatted training string from instruction/input/output."""
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

    user_content = f"{instruction}\n\n{inp}" if inp else instruction
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_sft_trainer(model, training_args, dataset, tokenizer, max_seq_length: int):
    """Construct SFTTrainer across slightly different TRL versions."""
    from trl import SFTTrainer

    sig = inspect.signature(SFTTrainer.__init__)
    params = set(sig.parameters.keys())

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
    }

    # Tokenizer argument name changed in newer TRL
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_length
    if "packing" in params:
        kwargs["packing"] = False

    return SFTTrainer(**kwargs)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    cfg = load_config(config_path)

    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    data_path = Path(cfg["data_path"])
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 1

    print(f"Config: {config_path}")
    print(f"Model:  {cfg['model_name']}")
    print(f"Data:   {data_path}")
    print(f"Out:    {output_dir}")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

    # --- Stage 2a: tokenizer ---
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Stage 2b: 4-bit base model (QLoRA) ---
    bnb_config = None
    if cfg.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    print("Loading base model (downloads on first run)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # --- Stage 3: LoRA on Qwen proj modules ---
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Stage 4: dataset + chat template ---
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = dataset.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    max_steps = int(cfg.get("max_steps", -1))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        logging_steps=int(cfg.get("logging_steps", 1)),
        save_steps=int(cfg.get("save_steps", 50)),
        save_total_limit=2,
        fp16=False,  # forced off: 2070 + bnb GradScaler/bf16 crash
        bf16=False,
        max_steps=max_steps,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=185,
        report_to="none",
        remove_unused_columns=False,
    )

    # --- Stage 5: train ---
    trainer = build_sft_trainer(
        model=model,
        training_args=training_args,
        dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=int(cfg.get("max_seq_length", 1024)),
    )
    print("Starting training...")
    trainer.train()

    # --- Stage 6: save adapter + tokenizer ---
    print(f"Saving adapter + tokenizer to {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
