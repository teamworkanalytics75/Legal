#!/usr/bin/env python3
"""
Generate embeddings for Massachusetts seal/pseudonym motions.

Reads the consolidated SQLite database (case_law_data/ma_motions.db),
chunks each opinion, and encodes those chunks with the legal and
general transformer models we already rely on elsewhere.

Outputs:
  - reports/analysis_outputs/ma_motions_chunks.jsonl
  - reports/analysis_outputs/ma_motions_embeddings_*.npz
  - summary JSON describing the run
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Default configuration
DATASET_NAME = "ma_motions"
DB_PATH = Path("case_law_data") / "ma_motions.db"
OUTPUT_DIR = Path("reports") / "analysis_outputs"
CHUNK_WORDS = 380
CHUNK_OVERLAP = 60

MODEL_REGISTRY: Dict[str, str] = {
    "legal": "nlpaueb/legal-bert-base-uncased",
    "general": "sentence-transformers/all-mpnet-base-v2",
}

LOGGER = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
    """Split opinion text into overlapping word windows."""
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: List[Tuple[str, int, int]] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append((" ".join(chunk_words), start, end))
        if end == len(words):
            break
    return chunks


def encode_chunks(model: SentenceTransformer, texts: Iterable[str]) -> np.ndarray:
    """Encode list of strings as dense vectors."""
    return model.encode(
        list(texts),
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def load_cases(db_path: Path) -> List[Dict]:
    """Load all cases from the MA motions database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            id,
            cluster_id,
            case_name,
            court,
            court_id,
            docket_number,
            date_filed,
            topic,
            cleaned_text
        FROM cases
        WHERE cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
    """

    cases = [dict(row) for row in conn.execute(query)]
    conn.close()

    LOGGER.info("Loaded %d cases with text.", len(cases))
    return cases


def sanitize_filename(value: str) -> str:
    """Create filesystem-safe filenames from case metadata."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Massachusetts motions database.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Path to ma_motions.db")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS, help="Words per chunk window")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Word overlap between chunks")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cases = load_cases(args.db_path)
    if not cases:
        LOGGER.warning("No cases available with text; aborting.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunk_records: List[Dict] = []
    chunk_texts: List[str] = []
    chunk_id = 0

    for case in cases:
        text = case["cleaned_text"]
        chunks = chunk_text(text, args.chunk_words, args.overlap)
        if not chunks:
            continue

        rel_path = sanitize_filename(f"{case.get('docket_number', '')}_{case['case_name']}") + ".txt"

        for chunk_text_value, start_word, end_word in chunks:
            chunk_records.append(
                {
                    "dataset": DATASET_NAME,
                    "chunk_id": chunk_id,
                    "document_id": case["id"],
                    "cluster_id": case.get("cluster_id"),
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
        LOGGER.warning("No chunk records produced. Nothing to embed.")
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

    embedding_files = {}
    for model_key, matrix in embeddings.items():
        embed_path = args.output_dir / f"{DATASET_NAME}_embeddings_{model_key}.npz"
        np.savez_compressed(embed_path, embeddings=matrix)
        embedding_files[model_key] = str(embed_path)
        LOGGER.info("Saved %s embeddings to %s", model_key, embed_path)

    summary = {
        "dataset": DATASET_NAME,
        "db_path": str(args.db_path.resolve()),
        "case_count": len(cases),
        "chunk_count": len(chunk_records),
        "models": MODEL_REGISTRY,
        "chunk_file": str(chunk_path),
        "embedding_files": embedding_files,
    }
    summary_path = args.output_dir / f"{DATASET_NAME}_corpus_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()

