#!/usr/bin/env python
"""
Dzongkha chat UI: local Ollama + simple RAG (Gradio ChatInterface).

Not the QLoRA path (see serve_chat.py). Uses configs/ollama_chat.yaml.

Examples:
  .\\.venv\\Scripts\\python.exe serve_ollama_chat.py --config configs/ollama_chat.yaml
  python serve_ollama_chat.py --no-rag --port 7860
  python serve_ollama_chat.py --rag-ab
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required: pip install pyyaml") from e

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dzongkha Ollama + RAG Gradio chat")
    p.add_argument("--config", type=str, default="configs/ollama_chat.yaml")
    p.add_argument("--model", type=str, default=None, help="Override Ollama model")
    p.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval")
    p.add_argument("--port", type=int, default=None)
    p.add_argument(
        "--rag-ab",
        action="store_true",
        help="CLI A/B: print answers with and without RAG for one prompt",
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument(
        "--repl",
        action="store_true",
        help="Force REPL even if Gradio is available",
    )
    return p.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cfg


def tokenize(text: str) -> set[str]:
    """Unicode-safe keyword tokens (letters/numbers; keeps Dzongkha syllables)."""
    text = text.casefold()
    parts = re.findall(r"[\w\u0F00-\u0FFF]+", text, flags=re.UNICODE)
    return {t for t in parts if len(t) > 1}


class SimpleRAG:
    def __init__(self, corpus_path: Path, top_k: int = 4) -> None:
        self.top_k = top_k
        self.docs: list[dict[str, Any]] = []
        if not corpus_path.is_file():
            print(f"[RAG] corpus missing: {corpus_path}", file=sys.stderr)
            return
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("content") or ""
                if not text:
                    continue
                self.docs.append(
                    {
                        "id": obj.get("id", ""),
                        "kind": obj.get("kind", ""),
                        "text": text,
                        "tokens": tokenize(text),
                    }
                )
        print(f"[RAG] loaded {len(self.docs)} docs from {corpus_path}")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not self.docs:
            return []
        k = top_k or self.top_k
        q = tokenize(query)
        if not q:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.docs:
            overlap = q & doc["tokens"]
            if not overlap:
                continue
            # Jaccard-ish + hit count; prefer denser overlaps
            score = len(overlap) / (len(q) ** 0.5)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]

    def format_context(self, docs: list[dict[str, Any]], max_chars: int = 3500) -> str:
        if not docs:
            return ""
        chunks: list[str] = []
        used = 0
        for i, d in enumerate(docs, 1):
            header = f"[{i}] id={d.get('id','')} kind={d.get('kind','')}"
            body = d["text"].strip()
            piece = f"{header}\n{body}"
            if used + len(piece) + 2 > max_chars:
                remain = max_chars - used - len(header) - 8
                if remain > 80:
                    chunks.append(f"{header}\n{body[:remain]}…")
                break
            chunks.append(piece)
            used += len(piece) + 2
        return "\n\n".join(chunks)


def ollama_reachable(host: str, timeout: float = 3.0) -> bool:
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def ollama_chat(
    host: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    num_ctx: int = 4096,
    stream: bool = False,
    timeout: float = 300.0,
) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {host}: {e}") from e

    if stream:
        # concatenate streamed JSON lines
        parts: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message") or {}
            if msg.get("content"):
                parts.append(msg["content"])
        return "".join(parts)

    obj = json.loads(raw)
    msg = obj.get("message") or {}
    return (msg.get("content") or "").strip()


def build_messages(
    system: str,
    history: list[tuple[str, str]],
    user_message: str,
    context: str,
) -> list[dict[str, str]]:
    sys = system.strip()
    msgs: list[dict[str, str]] = []
    if context:
        sys = (
            f"{sys}\n\nCONTEXT (retrieved; may be incomplete):\n{context}\n"
            "Use CONTEXT when relevant; if weak or irrelevant, say so briefly."
        )
    if sys:
        msgs.append({"role": "system", "content": sys})
    for u, a in history:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_message})
    return msgs


def ensure_gradio() -> Any:
    try:
        import gradio as gr

        return gr
    except ImportError:
        print("[ui] gradio missing — installing into project .venv / current env…")
        import subprocess

        py = sys.executable
        subprocess.check_call([py, "-m", "pip", "install", "gradio"], cwd=str(PROJECT_ROOT))
        import gradio as gr

        return gr


def run_rag_ab(cfg: dict[str, Any], model: str, rag: SimpleRAG | None) -> None:
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    chat = cfg.get("chat") or {}
    system = chat.get("system") or ""
    temperature = float(chat.get("temperature", 0.7))
    num_ctx = int(chat.get("num_ctx", 4096))
    prompt = input("A/B prompt> ").strip()
    if not prompt:
        return
    ctx = ""
    if rag is not None:
        docs = rag.retrieve(prompt)
        ctx = rag.format_context(docs)
        print("\n=== Retrieved CONTEXT ===\n")
        print(ctx or "(none)")
    print("\n=== WITH RAG ===\n")
    msgs_a = build_messages(system, [], prompt, ctx)
    print(ollama_chat(host, model, msgs_a, temperature=temperature, num_ctx=num_ctx))
    print("\n=== WITHOUT RAG ===\n")
    msgs_b = build_messages(system, [], prompt, "")
    print(ollama_chat(host, model, msgs_b, temperature=temperature, num_ctx=num_ctx))


def run_repl(cfg: dict[str, Any], model: str, rag: SimpleRAG | None) -> None:
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    chat = cfg.get("chat") or {}
    system = chat.get("system") or ""
    temperature = float(chat.get("temperature", 0.7))
    num_ctx = int(chat.get("num_ctx", 4096))
    history: list[tuple[str, str]] = []
    print("REPL mode. Empty line or /quit exits; /clear resets.")
    print("Prefer Gradio: pip install gradio then re-run without --repl")
    print("Open WebUI alternative: point it at Ollama http://127.0.0.1:11434")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user in {"/quit", "/exit"}:
            break
        if user == "/clear":
            history.clear()
            print("(cleared)")
            continue
        ctx = ""
        if rag is not None:
            docs = rag.retrieve(user)
            ctx = rag.format_context(docs)
        msgs = build_messages(system, history, user, ctx)
        try:
            reply = ollama_chat(host, model, msgs, temperature=temperature, num_ctx=num_ctx)
        except RuntimeError as e:
            print(f"error: {e}")
            continue
        print(f"bot> {reply}")
        history.append((user, reply))


def launch_gradio(
    cfg: dict[str, Any],
    model: str,
    rag: SimpleRAG | None,
    host: str,
    port: int,
) -> None:
    gr = ensure_gradio()
    ollama_host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    chat = cfg.get("chat") or {}
    system = chat.get("system") or ""
    temperature = float(chat.get("temperature", 0.7))
    num_ctx = int(chat.get("num_ctx", 4096))
    ui = cfg.get("ui") or {}
    title = ui.get("title") or "Dzongkha Chat (Ollama + RAG)"

    def respond(message: str, history: list) -> str:
        if not message or not str(message).strip():
            return ""
        # history: list of [user, assistant] or messages format depending on gradio version
        pairs: list[tuple[str, str]] = []
        if history:
            for item in history:
                if isinstance(item, dict):
                    # messages format — skip, ChatInterface may pass differently
                    continue
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append((str(item[0] or ""), str(item[1] or "")))
        ctx = ""
        if rag is not None:
            docs = rag.retrieve(message)
            ctx = rag.format_context(docs)
        msgs = build_messages(system, pairs, message, ctx)
        return ollama_chat(
            ollama_host, model, msgs, temperature=temperature, num_ctx=num_ctx
        )

    demo = gr.ChatInterface(
        fn=respond,
        title=title,
        description=(
            f"Model: **{model}** · Ollama: `{ollama_host}` · "
            f"RAG: {'on' if rag is not None else 'off'}"
        ),
    )
    url = f"http://{host}:{port}"
    print("=" * 60)
    print(f"Dzongkha Chat URL: {url}")
    print("Ollama must be running (ollama serve) with model pulled.")
    print("=" * 60)
    demo.launch(server_name=host, server_port=port, share=bool(ui.get("share", False)))


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path)

    model = args.model or cfg.get("model") or "qwen3.5:latest"
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    ui = cfg.get("ui") or {}
    port = args.port if args.port is not None else int(ui.get("port", 7860))
    rag_cfg = cfg.get("rag") or {}
    rag_enabled = bool(rag_cfg.get("enabled", True)) and not args.no_rag

    print(f"[cfg] {cfg_path}")
    print(f"[model] {model}")
    print(f"[ollama] {host}")
    if not ollama_reachable(host):
        print(
            f"WARNING: Ollama not reachable at {host}. Start it, then refresh.",
            file=sys.stderr,
        )
    else:
        print("[ollama] reachable")

    rag: SimpleRAG | None = None
    if rag_enabled:
        corpus = resolve_path(rag_cfg.get("corpus_path") or "data/rag/corpus.jsonl")
        rag = SimpleRAG(corpus, top_k=int(rag_cfg.get("top_k", 4)))
    else:
        print("[RAG] disabled")

    if args.rag_ab:
        run_rag_ab(cfg, model, rag)
        return

    if args.repl:
        run_repl(cfg, model, rag)
        return

    try:
        launch_gradio(cfg, model, rag, host=args.host, port=port)
    except Exception as e:
        print(f"[ui] Gradio failed ({e}); falling back to REPL", file=sys.stderr)
        print(f"Chatbox URL would have been: http://{args.host}:{port}")
        print("Open WebUI can also talk to Ollama at http://127.0.0.1:11434")
        run_repl(cfg, model, rag)


if __name__ == "__main__":
    main()
