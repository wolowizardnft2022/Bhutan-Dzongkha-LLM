# GitHub WIP allow/deny (Data Engineer) — 2026-09-19

## SAFE to commit (code / docs / recipes)
- `DATA_LICENSE.md`
- `.gitignore` (this policy)
- `scripts/pack_corpus.py`, `scripts/*.py`
- `configs/*.yaml`, `configs/*.json` (recipes only)
- `train.py`, `train_sft.py`, `model.py`, `eval.py`, `serve_chat.py`, `serve_ollama_chat.py` (if kept as retired reference)
- `requirements.txt`, `README.md` (must say experimental WIP; no “Dzongkha LLM v1”)
- `data/sample/dzongkha_sft_sample.jsonl` (tiny hand/sample seed only, if present)
- empty stubs: `data/raw/.gitkeep`, `data/processed/.gitkeep`

## MUST `.gitignore` / do not upload
- `data/train.bin`, `data/val.bin`, `data/train.txt`, `data/val.txt`, `data/meta.json`
- `data/sft/train.jsonl`, `data/sft/val.jsonl` (Eng–Dzo MT license **unknown**)
- `data/rag/corpus.*` (mixed FineWeb + uncleared ASR text)
- `raw/**` (FineWeb-2 dzo extract, wiki dump, MT zips, HF caches)
- `tokenizer/*.model`, `tokenizer/*.vocab` until dzoseg license confirmed
- `outputs/**` (QLoRA mid-run / smoke ckpts — not a release)
- `.venv/`, HF caches, `*.pt`, `*.safetensors`

## After Eval GO + license clear
Then optionally publish: cleared SFT subset, adapter weights, model card — not before.

HANDOFF -> publisher: use this allow/deny; need Tharchen GitHub user/org + repo name.
