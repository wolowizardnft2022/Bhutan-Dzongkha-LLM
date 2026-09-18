#!/usr/bin/env python
"""Environment checker for Dzongkha QLoRA training (Windows / CUDA)."""
from __future__ import annotations

import sys


def main() -> int:
    print("=== Dzongkha LLM env check ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")

    try:
        import torch
    except Exception as e:
        print(f"torch: IMPORT FAILED ({e})")
        print("Install CUDA torch first (see README), then re-run this script.")
        return 1

    print(f"torch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")

    if not cuda_ok:
        print("ERROR: CUDA is not available. Training on 8GB GPU requires CUDA.")
        print("Check: NVIDIA driver, CUDA torch wheel (cu121/cu124), reboot if needed.")
        return 1

    try:
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"GPU: {name}")
        print(f"VRAM total: {total_gb:.2f} GB")
    except Exception as e:
        print(f"GPU query failed: {e}")
        return 1

    try:
        import bitsandbytes as bnb  # noqa: F401

        ver = getattr(bnb, "__version__", "unknown")
        print(f"bitsandbytes: OK (version {ver})")
    except Exception as e:
        print(f"bitsandbytes: IMPORT FAILED ({e})")
        print("QLoRA 4-bit training needs bitsandbytes. Re-install requirements.")
        return 1

    print("=== check_env: OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
