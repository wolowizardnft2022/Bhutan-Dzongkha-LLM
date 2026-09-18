#!/usr/bin/env python
"""
Load Qwen2.5-7B-Instruct (4-bit) + a saved LoRA adapter and generate a reply.

Fits ~8GB VRAM. Example:
  python infer.py --adapter outputs/qwen25-7b-dzongkha-qlora
  python infer.py --adapter outputs/qwen25-7b-dzongkha-qlora --prompt "བཀྲ་ཤིས་བདེ་ལེགས། ཁྱོད་ག་དེ་སྡོད་དོ་ག?"
"""
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PROMPT = "བཀྲ་ཤིས་བདེ་ལེགས། ཁྱོད་ག་དེ་སྡོད་དོ་ག?"


def parse_args():
    p = argparse.ArgumentParser(description="Inference with Dzongkha QLoRA adapter")
    p.add_argument(
        "--adapter",
        type=str,
        default="outputs/qwen25-7b-dzongkha-qlora",
        help="Path to saved LoRA adapter directory",
    )
    p.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base HF model id (must match training)",
    )
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User prompt")
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    adapter_path = Path(args.adapter)
    if not adapter_path.is_absolute():
        adapter_path = (project_root / adapter_path).resolve()
    if not adapter_path.exists():
        print(f"ERROR: adapter not found: {adapter_path}")
        print("Train first: python train_sft.py --config configs/qlora_qwen25_7b.yaml")
        return 1

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Base:    {args.base_model}")
    print(f"Adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    gen = out[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    print("\n--- prompt ---")
    print(args.prompt)
    print("\n--- reply ---")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
