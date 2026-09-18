#!/usr/bin/env python
"""
Interactive multi-turn chat REPL for Dzongkha QLoRA adapter (4-bit + LoRA).

Loads the model once. Empty line or /quit exits; /clear resets history.

Example:
  python serve_chat.py --adapter outputs/qwen25-7b-dzongkha-qlora
  python serve_chat.py --adapter outputs/qwen25-7b-dzongkha-qlora --system "You are a helpful Dzongkha assistant."
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Interactive chat with Dzongkha QLoRA adapter")
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
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system prompt prepended to chat history",
    )
    return p.parse_args()


def resolve_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def load_model(adapter_path: Path, base_model: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Base:    {base_model}")
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
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_reply(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def initial_history(system: str | None) -> list[dict]:
    if system and system.strip():
        return [{"role": "system", "content": system.strip()}]
    return []


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    adapter_path = resolve_path(project_root, args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found: {adapter_path}")
        print("Train first: python train_sft.py --config configs/qlora_qwen25_7b.yaml")
        return 1

    model, tokenizer = load_model(adapter_path, args.base_model)
    history = initial_history(args.system)

    print("\nChat ready. Empty line or /quit to exit; /clear to reset history.\n")
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user or user in ("/quit", "/exit", "/q"):
            print("Bye.")
            break
        if user == "/clear":
            history = initial_history(args.system)
            print("(history cleared)")
            continue

        history.append({"role": "user", "content": user})
        try:
            reply = generate_reply(
                model,
                tokenizer,
                history,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"ERROR during generation: {e}")
            history.pop()  # drop failed user turn
            continue

        history.append({"role": "assistant", "content": reply})
        print(f"Bot> {reply}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
