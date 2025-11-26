#!/usr/bin/env python3
"""
Process the Harvard “Art of War” lawsuit corpus and generate multi-model embeddings.

Steps:
    1. Walk the master directory, extracting text from DOCX, PDF, CSV, XLSX, MD/TXT.
    2. Chunk content into ~400-word segments with overlap.
    3. Encode each chunk with a suite of BERT-based models (legal + general).
    4. Persist chunk metadata and embeddings for downstream retrieval workflows.
    5. Log the ingestion event into the shared system memory store.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from docx import Document as DocxDocument
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

from system_memory import LawsuitMemoryManager

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_BASE_DIR = Path("Lawsuit Data Analysis") / "Harvard - The Art of War-20251007T134844Z-1-001" / "Harvard - The Art of War"
DEFAULT_OUTPUT_DIR = Path("reports") / "analysis_outputs"
CHUNK_WORDS = 380
CHUNK_OVERLAP = 60

MODEL_REGISTRY: Dict[str, str] = {
    "legal": "nlpaueb/legal-bert-base-uncased",
    "general": "sentence-transformers/all-mpnet-base-v2",
}

LEGAL_FOLDERS = {
    "2. evidence",
    "3. 1782",
    "4. hong kong",
    "5. archive",
    "initial filing",
    "pacer",
    "harvard emails",
    "useful - legal",  # allow manual overrides if created later
}

RESEARCH_FOLDERS = {
    "1. deep research",
    "useful",
    "project wizard web",
}

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt", ".md", ".csv", ".xlsx", ".doc"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    chunk_id: int
    document_id: int
    relative_path: str
    category: str
    recommended_model: str
    text: str
    start_word: int
    end_word: int


@dataclass
class DocumentRecord:
    document_id: int
    relative_path: str
    category: str
    extension: str
    word_count: int
    chunk_count: int


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def iter_files(base_dir: Path) -> Iterable[Path]:
    for path in base_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def classify_document(base_dir: Path, path: Path) -> str:
    try:
        relative_parts = path.relative_to(base_dir).parts
    except ValueError:
        return "mixed"

    top_segment = relative_parts[0].lower() if relative_parts else ""
    if top_segment in LEGAL_FOLDERS:
        return "legal"
    if top_segment in RESEARCH_FOLDERS:
        return "research"
    return "mixed"


def recommended_model_for(category: str) -> str:
    if category == "legal":
        return MODEL_REGISTRY["legal"]
    if category == "research":
        return MODEL_REGISTRY["general"]
    return MODEL_REGISTRY["legal"]


def load_text_from_docx(path: Path) -> str:
    document = DocxDocument(path)
    paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def load_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                texts.append(text.strip())
        return "\n\n".join(texts)
    except Exception as exc:
        LOGGER.warning("PDF extraction failed for %s: %s", path, exc)
        return ""


def load_text_from_csv(path: Path) -> str:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("CSV parsing failed for %s: %s", path, exc)
        return ""
    return dataframe_to_text(df)


def load_text_from_xlsx(path: Path) -> str:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as exc:
        LOGGER.warning("XLSX parsing failed for %s: %s", path, exc)
        return ""
    text_blocks = []
    for name, df in sheets.items():
        text_blocks.append(f"[Sheet: {name}]\n{dataframe_to_text(df)}")
    return "\n\n".join(text_blocks)


def load_text_generic(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as exc:
            LOGGER.warning("Text read failed for %s: %s", path, exc)
            return ""


def dataframe_to_text(df: pd.DataFrame, max_rows: int = 50) -> str:
    subset = df.head(max_rows)
    return subset.to_csv(index=False)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return load_text_from_docx(path)
    if suffix == ".pdf":
        return load_text_from_pdf(path)
    if suffix == ".csv":
        return load_text_from_csv(path)
    if suffix == ".xlsx":
        return load_text_from_xlsx(path)
    if suffix in {".txt", ".md"}:
        return load_text_generic(path)
    if suffix == ".doc":
        # Legacy Word format; treat like docx via python-docx after conversion attempt
        LOGGER.warning("Skipping legacy .doc file pending manual conversion: %s", path)
        return ""
    return ""


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    words = text.split()
    if not words:
        return []
    step = max(chunk_words - overlap, 1)
    chunks: List[Tuple[str, int, int]] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunk_words_slice = words[start:end]
        if chunk_words_slice:
            chunk_text_value = " ".join(chunk_words_slice)
            chunks.append((chunk_text_value, start, end))
        if end == len(words):
            break
    return chunks


def encode_chunks(model: SentenceTransformer, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=DEVICE,
    )


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------

def process_corpus(
    base_dir: Path,
    output_dir: Path,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> Dict[str, any]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Scanning corpus at %s", base_dir)

    document_records: List[DocumentRecord] = []
    chunk_records: List[ChunkRecord] = []
    chunk_texts: List[str] = []

    document_id = 0
    chunk_id = 0

    for file_path in iter_files(base_dir):
        rel_path = file_path.relative_to(base_dir)
        category = classify_document(base_dir, file_path)
        recommended_model = recommended_model_for(category)
        text = extract_text(file_path)
        word_count = len(text.split())

        if not text or word_count == 0:
            LOGGER.debug("Skipping empty document: %s", rel_path)
            continue

        chunks = chunk_text(text, chunk_words=chunk_words, overlap=overlap)
        if not chunks:
            LOGGER.debug("No chunks generated for %s", rel_path)
            continue

        doc_record = DocumentRecord(
            document_id=document_id,
            relative_path=str(rel_path),
            category=category,
            extension=file_path.suffix.lower(),
            word_count=word_count,
            chunk_count=len(chunks),
        )
        document_records.append(doc_record)

        for chunk_text_value, start_word, end_word in chunks:
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    relative_path=str(rel_path),
                    category=category,
                    recommended_model=recommended_model,
                    text=chunk_text_value,
                    start_word=start_word,
                    end_word=end_word,
                )
            )
            chunk_texts.append(chunk_text_value)
            chunk_id += 1

        document_id += 1

    if not chunk_records:
        raise RuntimeError("No chunks produced. Check input directory and file formats.")

    LOGGER.info(
        "Prepared %d documents and %d chunks. Encoding with %d models.",
        len(document_records),
        len(chunk_records),
        len(MODEL_REGISTRY),
    )

    embeddings: Dict[str, np.ndarray] = {}
    LOGGER.info("Embedding device set to %s", DEVICE)

    for model_key, model_name in MODEL_REGISTRY.items():
        LOGGER.info("Encoding using %s (%s)", model_key, model_name)
        model = SentenceTransformer(model_name, cache_folder="models_cache", device=DEVICE)
        embeddings[model_key] = encode_chunks(model, chunk_texts)

    chunk_output = output_dir / "harvard_corpus_chunks.jsonl"
    with chunk_output.open("w", encoding="utf-8") as fh:
        for chunk in chunk_records:
            record = asdict(chunk)
            record["text"] = chunk.text
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    doc_output = output_dir / "harvard_corpus_documents.json"
    doc_output.write_text(
        json.dumps([asdict(doc) for doc in document_records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for model_key, matrix in embeddings.items():
        embed_path = output_dir / f"harvard_embeddings_{model_key}.npz"
        np.savez_compressed(embed_path, embeddings=matrix)

    summary = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "document_count": len(document_records),
        "chunk_count": len(chunk_records),
        "models": {k: v for k, v in MODEL_REGISTRY.items()},
        "chunk_file": str(chunk_output),
        "document_file": str(doc_output),
        "embedding_files": {
            model_key: str(output_dir / f"harvard_embeddings_{model_key}.npz")
            for model_key in MODEL_REGISTRY.keys()
        },
    }

    summary_path = output_dir / "harvard_corpus_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Process Harvard lawsuit corpus into embeddings.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root directory containing the Harvard corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for chunk metadata and embeddings.",
    )
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Overlap words between chunks.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    summary = process_corpus(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        chunk_words=args.chunk_words,
        overlap=args.overlap,
    )

    memory_manager = LawsuitMemoryManager()
    memory_manager.log_event(
        event_type="corpus_embedding_run",
        summary="Harvard Art of War corpus embedded with legal and general BERT models.",
        metadata={
            "document_count": summary["document_count"],
            "chunk_count": summary["chunk_count"],
            "models": summary["models"],
        },
        artifacts={
            "chunk_file": summary["chunk_file"],
            "document_file": summary["document_file"],
            "embedding_files": summary["embedding_files"],
        },
    )

    LOGGER.info("Processing complete. Summary written to %s", summary["output_dir"])


if __name__ == "__main__":
    main()
