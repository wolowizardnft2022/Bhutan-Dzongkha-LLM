# Bhutan-Dzongkha-LLM

**Status: experimental WIP** — research / lab recipes, not a finished Dzongkha foundation model and not a product release.

## What this is

Adapters-on-**Qwen2.5-7B-Instruct** (QLoRA) plus a tiny from-scratch CPU lab config (`configs/v0_cpu_tiny.json`). Goal: Dzongkha chat / translate experiments.

## What this is not

- Not a from-scratch Bhutan foundation model
- Not "Dzongkha LLM v1"
- Not ready for production deploy
- Weights and full training corpora are **not** in this repo (licenses + eval gate)

## Layout

| Path | Notes |
|------|--------|
| `train.py` / `train_sft.py` / `model.py` | Pretrain lab + QLoRA SFT |
| `configs/qlora_qwen25_7b.yaml` | Locked LoRA recipe (r/α/targets) |
| `configs/v0_cpu_tiny.json` | Tiny CPU lab card |
| `configs/ollama_chat.yaml` | Retired local demo reference |
| `serve_ollama_chat.py` | Retired Ollama chat helper |
| `infer.py` | Inference helpers |
| `scripts/` | Corpus packing / env checks |
| `DATA_LICENSE.md` | Source licenses (some MT/ASR still unknown) |
| `data/sample/` | Tiny sample seed only |

## Quick start (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# See PRETRAIN.md and configs/qlora_qwen25_7b.yaml for train recipes
```

## Publish policy

Code + recipes only until: full-epoch adapter + held-out eval pass + cleared licenses. See `GITHUB_ALLOW_DENY.md`.

## License

Code in this repo: see `DATA_LICENSE.md` for data source status. Do not redistribute ignored data/weight paths.
