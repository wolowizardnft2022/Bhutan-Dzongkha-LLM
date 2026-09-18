"""
CPU from-scratch pretraining for Dzongkha GPT (nanoGPT-style).

Usage (from project root, with .venv active):
  python train.py --config configs/v0_cpu_tiny.json --smoke
  python train.py --config configs/v0_cpu_tiny.json --max_steps 2000

Default --max_steps is small (100) so a laptop CPU run finishes in minutes.
Raise toward train_notes.max_steps_hint (2000) for a fuller pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from model import GPT, GPTConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v0_cpu_tiny.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "pretrain_cpu_tiny"


def resolve_path(p: str | Path) -> Path:
    """Resolve config paths on Linux box or Windows laptop.

    Accepts absolute paths that exist, relative paths from project root,
    and Windows-style abs paths (e.g. D:\\Dzongkha-LLM\\data\\train.bin)
    by mapping known project suffixes onto PROJECT_ROOT.
    """
    raw = str(p)
    path = Path(raw)
    if path.exists():
        return path.resolve()
    # Relative from project root
    cand = (PROJECT_ROOT / path).resolve()
    if cand.exists():
        return cand
    # Normalize Windows separators / drive paths
    s = raw.replace("\\", "/")
    for marker in ("/data/", "/tokenizer/", "/configs/", "/outputs/"):
        if marker in s:
            rel = s.split(marker, 1)[1]
            cand = (PROJECT_ROOT / marker.strip("/") / rel).resolve()
            if cand.exists():
                return cand
    # Basename under data/
    cand = (PROJECT_ROOT / "data" / Path(s).name).resolve()
    if cand.exists():
        return cand
    # Last resort: return absolute-as-is or project-relative
    if path.is_absolute() or (len(raw) >= 2 and raw[1] == ":"):
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous blocks of length block_size from a uint16 memmap."""
    n = len(data)
    # Need block_size + 1 tokens for x / y shift
    max_start = n - block_size - 1
    if max_start < 1:
        raise ValueError(f"Dataset too small: {n} tokens for block_size={block_size}")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix.tolist()]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix.tolist()
        ]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.memmap,
    val_data: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(
    step: int,
    learning_rate: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay to min_lr_ratio * learning_rate."""
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return learning_rate * min_lr_ratio
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * (min_lr_ratio + (1.0 - min_lr_ratio) * coeff)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Dict[str, Any],
    best_val: float,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "best_val": best_val,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": model.config.__dict__,
        },
        path,
    )


def append_metrics(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU pretrain Dzongkha GPT from scratch")
    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to Architect JSON config",
    )
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Optimization steps (default 100 for smoke; raise toward 2000 for fuller run)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--eval_iters", type=int, default=10)
    p.add_argument("--ckpt_interval", type=int, default=50)
    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stop after this many evals without val improvement",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Short smoke: max_steps=50, eval every 10, ckpt every 25",
    )
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_steps = 50
        args.eval_interval = 10
        args.ckpt_interval = 25
        args.eval_iters = 5
        args.warmup_steps = min(args.warmup_steps, 10)
        args.patience = 5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = resolve_path(args.config)
    cfg = load_config(cfg_path)
    notes = cfg.get("train_notes", {})
    batch_size = args.batch_size if args.batch_size is not None else int(notes.get("batch_size", 4))
    grad_accum = args.grad_accum if args.grad_accum is not None else int(notes.get("grad_accum", 8))

    data_cfg = cfg.get("data", {})
    train_bin = resolve_path(data_cfg.get("train_bin", "data/train.bin"))
    val_bin = resolve_path(data_cfg.get("val_bin", "data/val.bin"))

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    device = torch.device("cpu")
    # float32, no AMP (Architect laptop/CPU path)
    dtype = torch.float32

    gpt_cfg = GPTConfig.from_dict(cfg)
    model = GPT(gpt_cfg).to(device=device, dtype=dtype)
    n_params = model.get_num_params()
    print(f"Model params: {n_params:,} (~{n_params / 1e6:.2f}M)")
    print(f"Config: {gpt_cfg}")
    print(f"Device: {device}, dtype: float32, AMP: False")
    print(f"batch_size={batch_size}, grad_accum={grad_accum}, effective={batch_size * grad_accum}")
    print(f"max_steps={args.max_steps}, lr={args.lr}, seed={args.seed}")
    print(f"train_bin={train_bin}")
    print(f"val_bin={val_bin}")
    print(f"out_dir={out_dir}")

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}, val tokens: {len(val_data):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    evals_no_improve = 0
    t0 = time.time()
    model.train()
    running_loss = 0.0

    for step in range(args.max_steps):
        lr = get_lr(step, args.lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _micro in range(grad_accum):
            X, Y = get_batch(train_data, batch_size, gpt_cfg.block_size, device)
            _, loss = model(X, Y)
            loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        running_loss = loss_accum

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{args.max_steps} | train_loss {running_loss:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )

        do_eval = (step > 0 and step % args.eval_interval == 0) or step == args.max_steps - 1
        train_loss_est: Optional[float] = None
        val_loss_est: Optional[float] = None
        if do_eval:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                gpt_cfg.block_size,
                device,
                args.eval_iters,
            )
            train_loss_est = losses["train"]
            val_loss_est = losses["val"]
            print(
                f"  eval @ {step}: train {train_loss_est:.4f} | val {val_loss_est:.4f}"
            )
            append_metrics(
                metrics_path,
                {
                    "step": step,
                    "train_loss": train_loss_est,
                    "val_loss": val_loss_est,
                    "lr": lr,
                    "time_s": time.time() - t0,
                },
            )
            improved = val_loss_est < best_val - 1e-4
            if improved:
                best_val = val_loss_est
                evals_no_improve = 0
                best_path = out_dir / "ckpt_best.pt"
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    step,
                    cfg,
                    best_val,
                    train_loss_est,
                    val_loss_est,
                )
                print(f"  saved best -> {best_path} (val={best_val:.4f})")
            else:
                evals_no_improve += 1
                print(f"  no val improve ({evals_no_improve}/{args.patience})")
                if evals_no_improve >= args.patience:
                    print(f"Early stop at step {step} (patience={args.patience})")
                    save_checkpoint(
                        out_dir / f"ckpt_{step}.pt",
                        model,
                        optimizer,
                        step,
                        cfg,
                        best_val,
                        train_loss_est,
                        val_loss_est,
                    )
                    break

        if step > 0 and step % args.ckpt_interval == 0:
            ckpt_path = out_dir / f"ckpt_{step}.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                step,
                cfg,
                best_val,
                train_loss_est,
                val_loss_est,
            )
            print(f"  checkpoint -> {ckpt_path}")

    # Final checkpoint
    final_path = out_dir / f"ckpt_{step}.pt"
    if not final_path.exists():
        save_checkpoint(
            final_path,
            model,
            optimizer,
            step,
            cfg,
            best_val,
            train_loss_est,
            val_loss_est,
        )
        print(f"Final checkpoint -> {final_path}")

    # Always write a last.pt pointer-style copy of final state
    save_checkpoint(
        out_dir / "ckpt_last.pt",
        model,
        optimizer,
        step,
        cfg,
        best_val,
        train_loss_est,
        val_loss_est,
    )

    total = time.time() - t0
    print(
        f"Done. steps={step + 1}, best_val={best_val:.4f}, "
        f"last train_micro={running_loss:.4f}, wall={total:.1f}s"
    )
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoints under: {out_dir}")


if __name__ == "__main__":
    main()
