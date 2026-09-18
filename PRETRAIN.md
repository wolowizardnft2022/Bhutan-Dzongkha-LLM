# From-scratch CPU pretraining (Track A)

Teachable **nanoGPT-style** GPT decoder trained from scratch on packed Dzongkha tokens.
Does **not** replace Track B QLoRA (`train_sft.py`); both live side by side.

## Model (Architect `configs/v0_cpu_tiny.json`)

| Field | Value |
|-------|-------|
| n_layer / n_embd / n_head / n_inner | 4 / 128 / 4 / 512 |
| block_size / vocab_size | 128 / 32000 |
| params | ~4.9M (tied embeddings) |
| dtype / AMP | float32 / off (laptop CPU) |

Data (already packed by Data Engineer):

- `data/train.bin` — ~1.35M uint16 tokens  
- `data/val.bin` — ~79K  
- `data/meta.json` — vocab + SentencePiece path  

## Install CPU PyTorch

Use the project venv (do not pull a giant CUDA wheel on this path):

```bash
cd /workspace/dzongkha-llm
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
# QLoRA deps remain in requirements.txt; torch is documented there as a separate install.
```

## Smoke run (minutes on CPU)

```bash
cd /workspace/dzongkha-llm
source .venv/bin/activate
python train.py --config configs/v0_cpu_tiny.json --smoke
```

`--smoke` sets `max_steps=50`, eval every 10 steps, checkpoints every 25.

Default without `--smoke` is still modest (`--max_steps 100`). For a fuller pass toward the Architect hint:

```bash
python train.py --config configs/v0_cpu_tiny.json --max_steps 2000 --eval_interval 50 --ckpt_interval 200
```

CLI overrides: `--batch_size`, `--grad_accum`, `--lr`, `--patience` (early-stop on val, default 5 evals).

## Outputs

Under `outputs/pretrain_cpu_tiny/`:

- `metrics.jsonl` — train/val loss per eval  
- `ckpt_best.pt`, `ckpt_last.pt`, `ckpt_<step>.pt` — model, optimizer, step, config, best_val  

## Expected behavior on tiny data

With only ~1.35M train tokens and a ~4.9M model, a short run should **drive train loss down quickly** and can **overfit** (val may lag or rise). That is expected for this smoke config — use it to verify the loop, checkpoints, and logging before scaling data or steps.
