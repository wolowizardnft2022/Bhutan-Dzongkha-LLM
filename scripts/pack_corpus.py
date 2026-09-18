#!/usr/bin/env python3
"""Build Dzongkha pretrain bins: clean → dedupe → SentencePiece Unigram → uint16 bins."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import mwparserfromhell
import mwxml
import numpy as np
import sentencepiece as spm

ROOT = Path("/workspace/dzongkha-llm")
RAW = ROOT / "raw"
OUT = ROOT / "data"
EVAL = ROOT / "eval"
TOK = ROOT / "tokenizer" / "dzo_unigram_32000.model"

# Tibetan/Dzongkha script block + common punctuation / digits / latin for code-mix
TIBETAN_RE = re.compile(r"[\u0F00-\u0FFF]")
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = WS_RE.sub(" ", text).strip()
    return text


def is_mostly_dzongkha(text: str, min_chars: int = 20) -> bool:
    if len(text) < min_chars:
        return False
    tib = len(TIBETAN_RE.findall(text))
    return tib / max(len(text), 1) >= 0.3


def dedupe_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def extract_wiki(path: Path) -> list[str]:
    import bz2
    docs: list[str] = []
    dump = mwxml.Dump.from_file(bz2.open(path, "rb"))
    for page in dump:
        if page.namespace != 0:
            continue
        title = page.title or ""
        if title.startswith(("Wikipedia:", "Template:", "Category:", "Help:", "File:", "MediaWiki:")):
            continue
        rev = None
        for r in page:
            rev = r
        if rev is None or not rev.text:
            continue
        try:
            wikicode = mwparserfromhell.parse(rev.text)
            for tag in wikicode.filter_tags(recursive=True):
                if tag.tag.lower() in {"ref", "gallery", "timeline", "math", "syntaxhighlight", "source"}:
                    try:
                        wikicode.remove(tag)
                    except Exception:
                        pass
            text = wikicode.strip_code()
        except Exception:
            text = rev.text
        text = normalize(text)
        # split into paragraphs
        for para in re.split(r"\n{2,}", text):
            para = normalize(para)
            if is_mostly_dzongkha(para, min_chars=40):
                docs.append(para)
    return docs


def load_lines(path: Path) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = normalize(line)
            if is_mostly_dzongkha(line, min_chars=8):
                out.append(line)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    sources: dict[str, list[str]] = {}

    wiki_bz2 = RAW / "wiki" / "dzwiki-latest-pages-articles.xml.bz2"
    print("extracting wiki...", flush=True)
    sources["wiki"] = extract_wiki(wiki_bz2)
    print(f"  wiki paras: {len(sources['wiki'])}", flush=True)

    mt_train = RAW / "mt" / "extracted" / "train" / "train_dzo.txt"
    mt_val = RAW / "mt" / "extracted" / "val" / "validation_dzo.txt"
    mt_test = RAW / "mt" / "extracted" / "test" / "test_dzo.txt"
    sources["mt_train"] = load_lines(mt_train)
    sources["mt_val"] = load_lines(mt_val)
    sources["mt_test"] = load_lines(mt_test)
    print(
        f"  mt train/val/test: {len(sources['mt_train'])}/{len(sources['mt_val'])}/{len(sources['mt_test'])}",
        flush=True,
    )

    # HF tiny set
    hf_docs = []
    try:
        from datasets import load_dataset

        ds = load_dataset("UgyenP/dzongkha-language", split="train")
        for row in ds:
            for k in ("Title", "Description", "text"):
                if k in row and row[k]:
                    t = normalize(str(row[k]))
                    if is_mostly_dzongkha(t, min_chars=8):
                        hf_docs.append(t)
    except Exception as e:
        print("hf skip", e)
    sources["hf"] = hf_docs
    print(f"  hf: {len(hf_docs)}", flush=True)

    # companion json
    companion_docs = []
    for name in ("stories.json", "exams.json", "vocabulary.json"):
        p = RAW / "companion" / name
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        blob = json.dumps(data, ensure_ascii=False)
        # pull any Tibetan strings recursively
        def walk(o):
            if isinstance(o, str):
                t = normalize(o)
                if is_mostly_dzongkha(t, min_chars=8):
                    companion_docs.append(t)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)

        walk(data)
    sources["companion"] = companion_docs
    print(f"  companion: {len(companion_docs)}", flush=True)

    # Hold out MT val+test + a slice of wiki for eval; pretrain = wiki + mt_train + hf + companion
    # Deduplicate: remove any pretrain line that appears in eval holdouts
    eval_docs = []
    seen_eval = set()
    for key in ("mt_val", "mt_test"):
        for t in sources[key]:
            k = dedupe_key(t)
            if k not in seen_eval:
                seen_eval.add(k)
                eval_docs.append(t)

    # freeze ~200 wiki paras as generate/smoke eval (document-level holdout)
    wiki_hold = []
    for t in sources["wiki"][:: max(1, len(sources["wiki"]) // 200)][:200]:
        k = dedupe_key(t)
        if k not in seen_eval:
            seen_eval.add(k)
            wiki_hold.append(t)
    eval_docs.extend(wiki_hold)

    pretrain_docs = []
    seen = set()
    for key in ("wiki", "mt_train", "hf", "companion"):
        for t in sources[key]:
            k = dedupe_key(t)
            if k in seen_eval or k in seen:
                continue
            seen.add(k)
            pretrain_docs.append(t)

    print(f"pretrain docs: {len(pretrain_docs)}, eval docs: {len(eval_docs)}", flush=True)

    # document-level 95/5 split
    rng = np.random.default_rng(42)
    idx = np.arange(len(pretrain_docs))
    rng.shuffle(idx)
    n_val = max(1, int(0.05 * len(pretrain_docs)))
    val_idx = set(idx[:n_val].tolist())
    train_texts = [pretrain_docs[i] for i in idx if i not in val_idx]
    val_texts = [pretrain_docs[i] for i in idx if i in val_idx]

    # write plain text corpora for inspection
    (OUT / "train.txt").write_text("\n".join(train_texts) + "\n", encoding="utf-8")
    (OUT / "val.txt").write_text("\n".join(val_texts) + "\n", encoding="utf-8")
    (EVAL / "holdout.txt").write_text("\n".join(eval_docs) + "\n", encoding="utf-8")
    (EVAL / "smoke_prompts.txt").write_text(
        "\n".join(wiki_hold[:30]) + "\n", encoding="utf-8"
    )

    # also write a small fertility word list from MT val (space-ish / tsek segments)
    fertility_words = []
    for t in sources["mt_val"][:5000]:
        # split on tsek and spaces
        for w in re.split(r"[་\s]+", t):
            w = w.strip("།༎༏༐༑༔.,;:!?\"'()[]{}")
            if len(w) >= 2 and TIBETAN_RE.search(w):
                fertility_words.append(w)
    fertility_words = list(dict.fromkeys(fertility_words))[:5000]
    (EVAL / "fertility_words.txt").write_text("\n".join(fertility_words) + "\n", encoding="utf-8")

    print("tokenizing...", flush=True)
    sp = spm.SentencePieceProcessor(model_file=str(TOK))
    assert sp.get_piece_size() == 32000 or sp.vocab_size() == 32000

    def encode_docs(docs: list[str]) -> np.ndarray:
        ids: list[int] = []
        eos = sp.eos_id() if sp.eos_id() >= 0 else sp.piece_to_id("</s>")
        if eos < 0:
            eos = 1  # common fallback
        for doc in docs:
            piece_ids = sp.encode(doc, out_type=int)
            ids.extend(piece_ids)
            ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)
        return arr

    train_ids = encode_docs(train_texts)
    val_ids = encode_docs(val_texts)
    train_ids.tofile(OUT / "train.bin")
    val_ids.tofile(OUT / "val.bin")

    char_train = sum(len(t) for t in train_texts)
    char_val = sum(len(t) for t in val_texts)

    meta = {
        "vocab_size": 32000,
        "tokenizer": "sentencepiece_unigram",
        "sp_model": str(TOK),
        "dtype": "uint16",
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "train_docs": len(train_texts),
        "val_docs": len(val_texts),
        "train_chars": char_train,
        "val_chars": char_val,
        "eval_holdout_docs": len(eval_docs),
        "sources": {k: len(v) for k, v in sources.items()},
        "notes": "MT val/test + wiki holdout kept out of pretrain. DDC 27M-char dump still missing.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("DONE", OUT / "train.bin", OUT / "val.bin", OUT / "meta.json")


if __name__ == "__main__":
    main()
