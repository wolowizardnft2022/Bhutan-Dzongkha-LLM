# Data provenance & license notes (Dzongkha-LLM)

**Status:** research WIP — not an open-data product dump.  
Do **not** redistribute derived `train.bin` / `val.bin` / full `data/sft/*.jsonl` / FineWeb extracts until each source below is verified for your intended use (especially commercial).

This file lists what we ingested for the lab under `D:\Dzongkha-LLM` / `/workspace/dzongkha-llm`. Licenses marked **unknown** mean the upstream card did not declare one clearly — treat as **not redistributable** until you get written terms.

## Sources in the current pack

| Source | Path / use | Declared license (as of 2026-09-19) | Notes |
|--------|------------|--------------------------------------|-------|
| HuggingFaceFW/fineweb-2 (`dzo_Tibt`) | pretrain bins + RAG snippets | **ODC-By** | Keep attribution; follow ODC-By share conditions. Filtered `language_score ≥ 0.65`. |
| dz.wikipedia dump | pretrain + RAG holdouts | **CC BY-SA 3.0** (Wikimedia text) | Attribute Wikimedia; share-alike applies to adapted text dumps. |
| TenzinKhorloVIT/Eng_Dzo_MT_Datasets | SFT (`data/sft`) + MT-val holdouts | **unknown** (no license on card) | Parallel Eng↔Dzo. **Do not publish the text dump** until license is confirmed with the author. |
| Jyoti-77 Dzongkha TTS/ASR (text column only) | RAG (`hf_asr_tts`) | **unknown** | Audio not redistributed; text-only extracts still need license clearance. |
| UgyenP/dzongkha-language | tiny seed texts | **unknown** | 13 rows; negligible size. |
| dzongkhastudent/dzongkha-companion | tiny seed | **Apache-2.0** | OK under Apache-2.0 terms. |
| KarmaCST/dzoseg tokenizer | `tokenizer/dzo_unigram_32000.*` | **check upstream LICENSE** (repo LICENSE file was empty in our clone) | Confirm with https://github.com/KarmaCST/dzoseg before shipping the `.model`. |
| Handwritten Bhutan QA / phrasebook | RAG + SFT seeds | **project-owned** (created for this lab) | Short factual prompts; not a corpus claim. |

## Explicitly **not** in this pack

- **DDC ~27M-char corpus** — referenced in research papers; **not obtained**. Do not claim DDC data in any release.
- **Tibetan-only** corpora (TiBERT, OpenPecha, etc.) — deliberately excluded as Dzongkha substitutes.
- **OSCAR / C4 `dzo` slices** — none found for Dzongkha in our search.
- **facebook/flores** / **flores_plus** `dzo_Tibt` — gated; not ingested (needs HF auth + agreement).

## Derived artifacts

| Artifact | Redistribute? |
|----------|----------------|
| `data/train.bin`, `data/val.bin`, `data/meta.json` | Only if **all** contributing text sources allow it (FineWeb ODC-By + wiki BY-SA + cleared MT). Prefer publishing **scripts** (`scripts/pack_corpus.py`) over binaries. |
| `data/sft/train.jsonl`, `val.jsonl` | Blocked while Eng–Dzo MT license is unknown. |
| `data/rag/corpus.jsonl` | Mixed; strip uncleared sources before any public data upload. |
| Code (`train*.py`, `eval.py`, `serve_*.py`, configs) | Separate from data — choose a code license (e.g. MIT/Apache-2.0) in the repo root. |

## Recommended GitHub WIP contents (data lane)

Safe tonight:

1. This `DATA_LICENSE.md`
2. `scripts/pack_corpus.py` (+ README section: how to rebuild bins from upstream downloads)
3. **Pointers** to upstream dataset IDs (not the full FineWeb / MT dumps)

Defer:

- Uploading full FineWeb `dzo` extract, MT parallel files, or packed bins until MT/ASR licenses are cleared
- Any "open Dzongkha corpus" branding

## Contact / next clearance steps

1. Ask Eng–Dzo MT author (TenzinKhorloVIT) for an explicit license.
2. Confirm Jyoti-77 text redistribution.
3. Confirm dzoseg tokenizer license.
4. Optional: obtain DDC terms separately if/when Tharchen provides a dump.

Last updated: 2026-09-19 (UTC+9) by LLM Data Engineer for the Dzongkha-LLM lab.
