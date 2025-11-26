#!/usr/bin/env python3
"""
Generate embeddings for U.S. case law relevant to Section 1782 using legal transformer models.

Reads the aggregated CourtListener data stored in case_law_data/MaFederalMotions.db, filters
cases whose cleaned text references Section 1782, chunks the opinions, and encodes them with
our legal + general sentence transformers. Outputs chunk metadata and compressed embedding
matrices for later ingestion into the Chroma store.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

DATASET_NAME = "case_law"
DB_PATH = Path("case_law_data") / "MaFederalMotions.db"
OUTPUT_DIR = Path("reports") / "analysis_outputs"
CHUNK_WORDS = 380
CHUNK_OVERLAP = 60

MODEL_REGISTRY: Dict[str, str] = {
    "legal": "nlpaueb/legal-bert-base-uncased",
    "general": "sentence-transformers/all-mpnet-base-v2",
}

LOGGER = logging.getLogger(__name__)


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def chunk_text(text: str, chunk_size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks: List[Tuple[str, int, int]] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunk_text_value = " ".join(chunk_words)
            chunks.append((chunk_text_value, start, end))
        if end == len(words):
            break
    return chunks


def encode_chunks(model: SentenceTransformer, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def load_cases(sqlite_path: Path) -> List[Dict]:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Database not found: {sqlite_path}")

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT id, case_name, court, court_id, docket_number, date_filed, topic, cleaned_text
        FROM cases
        WHERE cleaned_text IS NOT NULL
          AND cleaned_text LIKE '%1782%'
    """

    with conn:
        rows = conn.execute(query).fetchall()

    cases = []
    for row in rows:
        text = (row["cleaned_text"] or "").strip()
        if not text:
            continue

        cases.append(
            {
                "id": row["id"],
                "case_name": row["case_name"],
                "court": row["court"],
                "court_id": row["court_id"],
                "docket_number": row["docket_number"],
                "date_filed": row["date_filed"],
                "topic": row["topic"],
                "text": text,
            }
        )

    LOGGER.info("Loaded %d cases containing Section 1782 references.", len(cases))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Section 1782 case law using legal BERT models.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Path to MaFederalMotions.db")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between chunks")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cases = load_cases(args.db_path)
    if not cases:
        LOGGER.warning("No cases found with Section 1782 references. Nothing to embed.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunk_records: List[Dict] = []
    chunk_texts: List[str] = []
    chunk_id = 0

    for doc_index, case in enumerate(cases):
        rel_path = sanitize_filename(f"{case['docket_number']}_{case['case_name']}") + ".txt"
        chunks = chunk_text(case["text"], chunk_size=args.chunk_words, overlap=args.overlap)
        if not chunks:
            continue
        for chunk_text_value, start_word, end_word in chunks:
            chunk_records.append(
                {
                    "dataset": DATASET_NAME,
                    "chunk_id": chunk_id,
                    "document_id": case["id"],
                    "case_name": case["case_name"],
                    "relative_path": rel_path,
                    "court": case["court"],
                    "court_id": case["court_id"],
                    "docket_number": case["docket_number"],
                    "date_filed": case["date_filed"],
                    "topic": case["topic"],
                    "text": chunk_text_value,
                    "start_word": start_word,
                    "end_word": end_word,
                }
            )
            chunk_texts.append(chunk_text_value)
            chunk_id += 1

    if not chunk_records:
        LOGGER.warning("No chunks produced after processing cases.")
        return

    LOGGER.info("Generated %d chunks across %d cases.", len(chunk_records), len(cases))

    embeddings: Dict[str, np.ndarray] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_key, model_name in MODEL_REGISTRY.items():
        LOGGER.info("Encoding chunks with %s (%s)", model_key, model_name)
        model = SentenceTransformer(model_name, cache_folder=str(Path("models_cache")), device=device)
        embeddings[model_key] = encode_chunks(model, chunk_texts)

    chunk_path = args.output_dir / f"{DATASET_NAME}_chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8") as fh:
        for record in chunk_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Saved chunk metadata to %s", chunk_path)

    for model_key, matrix in embeddings.items():
        embed_path = args.output_dir / f"{DATASET_NAME}_embeddings_{model_key}.npz"
        np.savez_compressed(embed_path, embeddings=matrix)
        LOGGER.info("Saved %s embeddings to %s", model_key, embed_path)

    summary = {
        "dataset": DATASET_NAME,
        "db_path": str(args.db_path),
        "case_count": len(cases),
        "chunk_count": len(chunk_records),
        "models": MODEL_REGISTRY,
        "chunk_file": str(chunk_path),
        "embedding_files": {
            model_key: str(args.output_dir / f"{DATASET_NAME}_embeddings_{model_key}.npz")
            for model_key in MODEL_REGISTRY.keys()
        },
    }
    summary_path = args.output_dir / f"{DATASET_NAME}_corpus_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
