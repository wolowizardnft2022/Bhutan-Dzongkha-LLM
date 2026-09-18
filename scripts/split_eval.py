#!/usr/bin/env python
"""
Split a JSONL into train/eval by fraction (default 0.9 / 0.1).

Example:
  python scripts/split_eval.py --input data/processed/train.jsonl --seed 0
  python scripts/split_eval.py --input data/sample/dzongkha_sft_sample.jsonl \\
      --train_out data/processed/train.jsonl --eval_out data/eval/holdout.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split JSONL into train/eval by fraction")
    p.add_argument("--input", type=str, required=True, help="Source JSONL")
    p.add_argument(
        "--train_out",
        type=str,
        default="data/processed/train_split.jsonl",
        help="Train output path",
    )
    p.add_argument(
        "--eval_out",
        type=str,
        default="data/eval/holdout.jsonl",
        help="Eval output path",
    )
    p.add_argument(
        "--train_frac",
        type=float,
        default=0.9,
        help="Fraction for train (rest goes to eval; default 0.9)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_path(root: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


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
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    if not (0.0 < args.train_frac < 1.0):
        print("ERROR: --train_frac must be between 0 and 1 (exclusive)", file=sys.stderr)
        return 1

    src = resolve_path(root, args.input)
    if not src.exists():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        return 1

    rows = load_jsonl(src)
    if len(rows) < 2:
        print(f"ERROR: need at least 2 rows to split (got {len(rows)})", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n_train = max(1, min(len(rows) - 1, int(round(len(rows) * args.train_frac))))
    train_idxs = set(idxs[:n_train])
    train_rows = [rows[i] for i in range(len(rows)) if i in train_idxs]
    eval_rows = [rows[i] for i in range(len(rows)) if i not in train_idxs]

    train_out = resolve_path(root, args.train_out)
    eval_out = resolve_path(root, args.eval_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    print(f"Input:  {src} ({len(rows)} rows)")
    print(f"Train:  {train_out} ({len(train_rows)} rows, frac~{args.train_frac})")
    print(f"Eval:   {eval_out} ({len(eval_rows)} rows)")
    print("Note: for real experiments, ensure eval does not overlap training data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
