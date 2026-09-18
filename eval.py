#!/usr/bin/env python
"""
Batch eval for Dzongkha QLoRA adapter (4-bit base + LoRA).

Metrics (per example + averages):
  - exact_match: normalized strip; Latin lowered; Tibetan kept as-is
  - contains_ref: reference appears in prediction after strip
  - char_f1: character-level F1 (useful for Dzongkha)

Example:
  python eval.py --adapter outputs/qwen25-7b-dzongkha-qlora
  python eval.py --adapter outputs/qwen25-7b-dzongkha-qlora --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Batch eval for Dzongkha QLoRA adapter")
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
    p.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Eval JSONL (default: data/eval/holdout.jsonl if present)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval",
        help="Directory for results JSONL + summary",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=None, help="Optional smoke limit (first N rows)")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 = greedy, default for eval)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def build_user_prompt(example: dict) -> str:
    """Same user-side formatting as train_sft.format_example (instruction + optional input)."""
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    return f"{instruction}\n\n{inp}" if inp else instruction


_LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_for_exact(text: str) -> str:
    """Strip; lower only when Latin letters are present; keep Tibetan as-is."""
    t = (text or "").strip()
    if _LATIN_RE.search(t):
        return t.lower()
    return t


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_for_exact(pred) == normalize_for_exact(ref) else 0.0


def contains_ref(pred: str, ref: str) -> float:
    p = (pred or "").strip()
    r = (ref or "").strip()
    if not r:
        return 0.0
    # Prefer casefold for Latin-heavy refs; Tibetan unaffected
    if _LATIN_RE.search(r):
        return 1.0 if r.lower() in p.lower() else 0.0
    return 1.0 if r in p else 0.0


def char_f1(pred: str, ref: str) -> float:
    """Character-level F1 over Unicode code points (after strip)."""
    p = list((pred or "").strip())
    r = list((ref or "").strip())
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    pc, rc = Counter(p), Counter(r)
    overlap = sum((pc & rc).values())
    precision = overlap / len(p)
    recall = overlap / len(r)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN skip {path.name}:{lineno}: {e}", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                continue
            rows.append(obj)
    return rows


def resolve_eval_data(project_root: Path, cli_path: str | None) -> Path | None:
    if cli_path:
        return resolve_path(project_root, cli_path)
    default = project_root / "data" / "eval" / "holdout.jsonl"
    if default.exists():
        return default.resolve()
    return None


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


def generate_one(
    model,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    messages = [{"role": "user", "content": user_prompt}]
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

    random.seed(args.seed)

    adapter_path = resolve_path(project_root, args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found: {adapter_path}")
        print("Train first: python train_sft.py --config configs/qlora_qwen25_7b.yaml")
        return 1

    eval_path = resolve_eval_data(project_root, args.eval_data)
    if eval_path is None:
        print("ERROR: no eval data found.")
        print("Expected data/eval/holdout.jsonl, or pass --eval_data PATH")
        print("Tip: create a holdout with: python scripts/split_eval.py --input data/sample/dzongkha_sft_sample.jsonl")
        return 1
    if not eval_path.exists():
        print(f"ERROR: eval data not found: {eval_path}")
        return 1

    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(eval_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        print(f"ERROR: no examples in {eval_path}")
        return 1

    print(f"Eval data: {eval_path} ({len(rows)} examples)")
    print(f"Out dir:   {output_dir}")

    # Seed torch if available (after we know we will load)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    model, tokenizer = load_model(adapter_path, args.base_model)

    from tqdm import tqdm

    results = []
    sum_em = sum_cr = sum_f1 = 0.0

    for ex in tqdm(rows, desc="eval", unit="ex"):
        prompt = build_user_prompt(ex)
        reference = (ex.get("output") or "").strip()
        prediction = generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        scores = {
            "exact_match": exact_match(prediction, reference),
            "contains_ref": contains_ref(prediction, reference),
            "char_f1": char_f1(prediction, reference),
        }
        sum_em += scores["exact_match"]
        sum_cr += scores["contains_ref"]
        sum_f1 += scores["char_f1"]
        results.append(
            {
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "scores": scores,
            }
        )

    n = len(results)
    summary = {
        "n": n,
        "eval_data": str(eval_path),
        "adapter": str(adapter_path),
        "base_model": args.base_model,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "averages": {
            "exact_match": sum_em / n,
            "contains_ref": sum_cr / n,
            "char_f1": sum_f1 / n,
        },
    }

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_json = output_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "average"])
        for k, v in summary["averages"].items():
            w.writerow([k, f"{v:.6f}"])
        w.writerow(["n", n])

    print("\n=== Eval summary ===")
    print(f"n={n}")
    for k, v in summary["averages"].items():
        print(f"  {k}: {v:.4f}")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
