#!/usr/bin/env python
"""
Tiny local FastAPI server for Dzongkha QLoRA (4-bit + LoRA).

Local use only — not for production / public exposure.

  pip install fastapi uvicorn
  python serve_api.py --adapter outputs/qwen25-7b-dzongkha-qlora

POST /generate  {"prompt": "...", "max_new_tokens": 128, "temperature": 0.7}
  -> {"text": "..."}

Optional OpenAI-ish POST /v1/chat/completions with messages[].
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Heavy imports deferred until after argparse / adapter check


def parse_args():
    p = argparse.ArgumentParser(description="Local FastAPI serve for Dzongkha QLoRA")
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
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional default system prompt for /v1/chat/completions",
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


def generate_from_messages(
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


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    adapter_path = resolve_path(project_root, args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found: {adapter_path}")
        print("Train first: python train_sft.py --config configs/qlora_qwen25_7b.yaml")
        return 1

    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
        import uvicorn
    except ImportError:
        print("ERROR: fastapi/uvicorn not installed.")
        print("Install with: pip install fastapi uvicorn")
        print("Or use the REPL instead: python serve_chat.py --adapter ...")
        return 1

    model, tokenizer = load_model(adapter_path, args.base_model)
    default_max = args.max_new_tokens
    default_temp = args.temperature
    default_system = args.system

    app = FastAPI(title="Dzongkha QLoRA local API", docs_url="/docs")

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int | None = None
        temperature: float | None = None

    class GenerateResponse(BaseModel):
        text: str

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: list[ChatMessage]
        max_tokens: int | None = Field(default=None, alias="max_new_tokens")
        temperature: float | None = None

        class Config:
            populate_by_name = True

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        messages = [{"role": "user", "content": req.prompt}]
        if default_system:
            messages = [{"role": "system", "content": default_system}] + messages
        text = generate_from_messages(
            model,
            tokenizer,
            messages,
            max_new_tokens=req.max_new_tokens or default_max,
            temperature=default_temp if req.temperature is None else req.temperature,
        )
        return GenerateResponse(text=text)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        if default_system and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": default_system}] + messages
        text = generate_from_messages(
            model,
            tokenizer,
            messages,
            max_new_tokens=req.max_tokens or default_max,
            temperature=default_temp if req.temperature is None else req.temperature,
        )
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
        }

    print(f"Local API on http://{args.host}:{args.port}  (local use only)")
    print("  POST /generate  {\"prompt\": \"...\"}")
    print("  POST /v1/chat/completions  {\"messages\": [...]}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
