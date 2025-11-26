"""
Legal chat assistant CLI.

Loads the locally cached Phi-3 Mini model for conversational answers and can
optionally ground responses using a retrieval step over local legal documents.

Usage examples (run from the repo root):

    # Ask a single question without retrieval
    python scripts/legal_chat_assistant.py --question "Summarize Section 1782."

    # Ask with retrieval over a directory of legal texts
    python scripts/legal_chat_assistant.py \
        --question "What did Intel v. AMD decide about Section 1782?" \
        --context-dir section_1782_mining/data

    # Start an interactive session with retrieval
    python scripts/legal_chat_assistant.py --interactive \
        --context-dir section_1782_mining/data
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.cache_utils import DynamicCache, StaticCache

import torch._dynamo
torch._dynamo.config.suppress_errors = True

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# Backwards compatibility: older transformers releases do not expose `seen_tokens`
# on cache objects, but Phi-3 remote code expects it. Provide a shim.
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]
if not hasattr(StaticCache, "seen_tokens"):
    StaticCache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: self.max_cache_len  # type: ignore[attr-defined]
if not hasattr(StaticCache, "get_max_length"):
    StaticCache.get_max_length = lambda self: self.max_cache_len  # type: ignore[attr-defined]

DEFAULT_MODEL_PATH = Path("models_cache/qwen2-1_5b-instruct")
DEFAULT_MAX_TOKENS = 512


@dataclass
class RetrievedChunk:
    """Small container describing retrieved context."""

    text: str
    source: Path
    chunk_id: int
    score: float


@dataclass
class WebSnippet:
    """Container for live web search snippets."""

    title: str
    url: str
    summary: str


class SemanticRetriever:
    """Lightweight semantic search helper using SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir) if cache_dir else None,
        )
        self.corpus_embeddings = None
        self.corpus_texts: Optional[List[str]] = None
        self.corpus_metadata: Optional[List[dict]] = None

    def index(self, texts: Sequence[str], metadata: Sequence[dict], batch_size: int = 64) -> None:
        self.corpus_texts = list(texts)
        self.corpus_metadata = list(metadata)
        self.corpus_embeddings = self.model.encode(
            self.corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[RetrievedChunk]:
        if self.corpus_embeddings is None or self.corpus_texts is None:
            raise RuntimeError("Corpus not indexed. Call index() first.")

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)[0]

        results: List[RetrievedChunk] = []
        for hit in hits:
            if hit["score"] < min_score:
                continue
            corpus_id = int(hit["corpus_id"])
            metadata = self.corpus_metadata[corpus_id] if self.corpus_metadata else {}
            results.append(
                RetrievedChunk(
                    text=self.corpus_texts[corpus_id],
                    score=float(hit["score"]),
                    source=Path(metadata.get("source", "unknown")),
                    chunk_id=int(metadata.get("chunk_id", 0)),
                )
            )
        return results


def iter_plaintext_files(root: Path) -> Iterable[Path]:
    """Yield text-like files beneath the provided root directory."""
    valid_suffixes = {".txt", ".md", ".markdown", ".rst"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_suffixes:
            yield path


def chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split large documents into overlapping chunks of roughly `chunk_size` words."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)

    return chunks


def fetch_web_snippets(
    question: str,
    max_results: int,
    request_timeout: float = 10.0,
) -> List[WebSnippet]:
    """Query DuckDuckGo and return top snippets."""
    if max_results <= 0:
        return []
    if DDGS is None:
        raise RuntimeError("duckduckgo-search package not installed; cannot perform web queries.")

    snippets: List[WebSnippet] = []
    try:
        with DDGS(timeout=request_timeout) as ddgs:
            results = ddgs.text(question, max_results=max_results)
            for result in results:
                title = (result.get("title") or result.get("article_title") or "").strip()
                url = result.get("href") or result.get("url") or ""
                summary = (result.get("body") or result.get("snippet") or "").strip()

                if not summary and url:
                    try:
                        resp = requests.get(
                            url,
                            timeout=request_timeout,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; LegalChatAssistant/1.0)"},
                        )
                        if resp.ok:
                            text = resp.text
                            summary = text[:600]
                    except Exception:
                        summary = ""

                snippets.append(
                    WebSnippet(
                        title=title or "Untitled result",
                        url=url or "Unknown URL",
                        summary=summary[:600],
                    )
                )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Web search failed: {exc}") from exc

    return snippets[:max_results]


def build_corpus(context_dir: Path, chunk_size: int = 500, overlap: int = 100) -> Tuple[List[str], List[dict]]:
    """
    Load documents from `context_dir` and return chunked texts with metadata.

    Returns:
        texts: List of chunked text segments.
        metadata: Corresponding metadata dictionaries.
    """
    texts: List[str] = []
    metadata: List[dict] = []

    for doc_path in iter_plaintext_files(context_dir):
        try:
            contents = doc_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            contents = doc_path.read_text(encoding="latin-1")

        for chunk_id, chunk in enumerate(chunk_document(contents, chunk_size, overlap)):
            texts.append(chunk)
            metadata.append({"source": str(doc_path), "chunk_id": chunk_id})

    return texts, metadata


def retrieve_context(
    question: str,
    search_engine: SemanticRetriever,
    texts: Sequence[str],
    metadata: Sequence[dict],
    top_k: int = 3,
) -> List[RetrievedChunk]:
    """Run semantic search and return top scored chunks."""
    if not texts:
        return []

    search_engine.index(list(texts), list(metadata))
    return search_engine.search(question, top_k=top_k, min_score=0.0)


def build_prompt(
    question: str,
    retrieved: Optional[List[RetrievedChunk]] = None,
    web_snippets: Optional[List[WebSnippet]] = None,
) -> str:
    """Construct the final prompt for the chat model."""
    instructions = textwrap.dedent(
        """
        You are TheMatrix legal research assistant. Provide concise, well-structured answers.
        Cite retrieved context snippets when available, using `[source, chunk_id]` notation.
        If the context is insufficient, say so explicitly before general reasoning.
        """
    ).strip()

    prompt_lines = [instructions]

    if retrieved:
        prompt_lines.append("\nContext snippets:")
        for chunk in retrieved:
            prompt_lines.append(
                f"[{chunk.source.name}, {chunk.chunk_id}] {chunk.text.strip()}"
            )

    if web_snippets:
        prompt_lines.append("\nWeb findings:")
        for idx, snippet in enumerate(web_snippets, start=1):
            prompt_lines.append(
                f"[web:{idx}] {snippet.title}\nURL: {snippet.url}\nSummary: {snippet.summary}"
            )

    prompt_lines.append("\nQuestion:")
    prompt_lines.append(question.strip())
    prompt_lines.append("\nAnswer:")

    return "\n".join(prompt_lines)


def load_chat_pipeline(model_path: Path) -> pipeline:
    """Load the Phi-3 pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory '{model_path}' not found. Download it first via snapshot_download."
        )

    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto" if device == 0 else None,
    )

    # Ensure compatibility with current transformers release.
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"
    if hasattr(model.config, "cache_implementation"):
        model.config.cache_implementation = "static"
    if hasattr(model.generation_config, "cache_implementation"):
        model.generation_config.cache_implementation = "static"

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=DEFAULT_MAX_TOKENS,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        model_kwargs={"use_cache": False},
        device=device,
    )


def run_single_query(
    chat,
    question: str,
    retrieved: Optional[List[RetrievedChunk]] = None,
    web_snippets: Optional[List[WebSnippet]] = None,
) -> str:
    """Generate a single response and return the text."""
    prompt = build_prompt(question, retrieved, web_snippets)
    outputs = chat(prompt)
    generated = outputs[0]["generated_text"]
    # The pipeline returns the entire prompt + generation; trim if necessary.
    if "Answer:" in generated:
        generated = generated.split("Answer:", 1)[1].strip()
    return generated.strip()


def interactive_loop(
    chat,
    context_texts: Optional[List[str]],
    context_meta: Optional[List[dict]],
    web_results: int,
    web_timeout: float,
) -> None:
    """Run a simple REPL for chatting."""
    search_engine: Optional[SemanticRetriever] = None
    if context_texts and context_meta:
        search_engine = SemanticRetriever()

    print("Legal chat assistant. Type 'exit' or Ctrl+C to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        retrieved: Optional[List[RetrievedChunk]] = None
        if search_engine:
            retrieved = retrieve_context(question, search_engine, context_texts, context_meta)

        web_snippets: Optional[List[WebSnippet]] = None
        if web_results > 0:
            try:
                web_snippets = fetch_web_snippets(question, web_results, web_timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"[web] Warning: {exc}")

        answer = run_single_query(chat, question, retrieved, web_snippets)
        print(f"\nAssistant: {answer}\n")


def save_retrieval_trace(chunks: Sequence[RetrievedChunk], output_path: Path) -> None:
    """Persist retrieved chunk metadata for later inspection."""
    payload = [
        {"text": chunk.text, "source": str(chunk.source), "chunk_id": chunk.chunk_id, "score": chunk.score}
        for chunk in chunks
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Phi-3 Mini using local legal context.")
    parser.add_argument("--question", type=str, help="Single question to ask the assistant.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat loop.")
    parser.add_argument(
        "--context-dir",
        type=Path,
        help="Optional directory containing legal text files to use for retrieval.",
    )
    parser.add_argument(
        "--save-trace",
        type=Path,
        help="If set, save retrieved context snippets to this JSON file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the downloaded Phi-3 Mini model directory.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of words per retrieval chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Word overlap between retrieval chunks.",
    )
    parser.add_argument(
        "--web-results",
        type=int,
        default=0,
        help="Number of live web search results to pull via DuckDuckGo (0 disables web search).",
    )
    parser.add_argument(
        "--web-timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for each web request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.interactive and not args.question:
        raise SystemExit("Provide --question for single mode or --interactive for REPL mode.")

    chat = load_chat_pipeline(args.model_path)

    context_texts: Optional[List[str]] = None
    context_meta: Optional[List[dict]] = None
    if args.context_dir:
        if not args.context_dir.exists():
            raise SystemExit(f"Context directory '{args.context_dir}' does not exist.")

        print(f"Indexing context from {args.context_dir} ...")
        context_texts, context_meta = build_corpus(args.context_dir, args.chunk_size, args.chunk_overlap)
        print(f"Loaded {len(context_texts)} context chunks from {args.context_dir}.")

    if args.interactive:
        interactive_loop(chat, context_texts, context_meta, args.web_results, args.web_timeout)
        return

    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    if context_texts and context_meta:
        search_engine = SemanticRetriever()
        retrieved_chunks = retrieve_context(args.question, search_engine, context_texts, context_meta)

    web_snippets: Optional[List[WebSnippet]] = None
    if args.web_results > 0:
        try:
            web_snippets = fetch_web_snippets(args.question, args.web_results, args.web_timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[web] Warning: {exc}")

    answer = run_single_query(chat, args.question, retrieved_chunks, web_snippets)
    print(answer)

    if retrieved_chunks and args.save_trace:
        save_retrieval_trace(retrieved_chunks, args.save_trace)
        print(f"\nRetrieved context saved to {args.save_trace}.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
