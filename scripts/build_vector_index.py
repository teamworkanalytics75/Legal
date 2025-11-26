#!/usr/bin/env python3
"""
Build a Chroma vector index from available embedding files.

The script aggregates chunk metadata and embeddings across multiple datasets
and materialises them into persistent Chroma collections. Each dataset/model
pair receives its own collection (e.g., harvard_legal, case_law_legal).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import sys

import chromadb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from system_memory import LawsuitMemoryManager

LOGGER = logging.getLogger(__name__)


def load_embeddings(path: Path) -> np.ndarray:
    data = np.load(path)
    if "embeddings" not in data:
        raise ValueError(f"No 'embeddings' key in {path}")
    return data["embeddings"]


def load_chunks(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def parse_dataset_name_from_chunks(path: Path) -> str:
    stem = path.stem  # e.g., harvard_corpus_chunks, case_law_chunks
    if stem.endswith("_chunks"):
        stem = stem[: -len("_chunks")]
    return normalize_dataset_name(stem)


def parse_dataset_and_model(path: Path) -> Tuple[str, str]:
    stem = path.stem  # e.g., harvard_embeddings_legal
    if "_embeddings_" not in stem:
        raise ValueError(f"Embedding file name must follow <dataset>_embeddings_<model>.npz format: {path}")
    dataset_raw, model_key = stem.split("_embeddings_", maxsplit=1)
    return normalize_dataset_name(dataset_raw), model_key


def normalize_dataset_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("harvard"):
        return "harvard"
    if "case_law" in lower:
        return "case_law"
    return lower


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma vector store from embedding files.")
    parser.add_argument(
        "--chunk-files",
        type=Path,
        nargs="+",
        default=[Path("reports") / "analysis_outputs" / "harvard_corpus_chunks.jsonl"],
        help="One or more JSONL files containing chunk metadata.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("reports") / "analysis_outputs",
        help="Directory containing compressed embedding matrices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chroma_collections"),
        help="Persistent Chroma directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    for chunk_file in args.chunk_files:
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
    if not args.embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings dir not found: {args.embeddings_dir}")

    # Load chunk metadata grouped by dataset
    chunks_by_dataset: Dict[str, List[Dict]] = {}
    total_chunks = 0
    for chunk_file in args.chunk_files:
        dataset_name = parse_dataset_name_from_chunks(chunk_file)
        records = load_chunks(chunk_file)
        if not records:
            continue
        for record in records:
            record.setdefault("dataset", dataset_name)
        chunks_by_dataset.setdefault(dataset_name, []).extend(records)
        total_chunks += len(records)

    if not chunks_by_dataset:
        raise RuntimeError("No chunk records were loaded.")

    LOGGER.info(
        "Loaded chunk metadata for datasets: %s",
        ", ".join(f"{dataset} ({len(records)})" for dataset, records in chunks_by_dataset.items()),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(args.output_dir))

    collections_built: List[str] = []

    for embedding_file in args.embeddings_dir.glob("*_embeddings_*.npz"):
        dataset, model_key = parse_dataset_and_model(embedding_file)
        chunk_records = chunks_by_dataset.get(dataset)
        if chunk_records is None:
            LOGGER.warning("Skipping %s; no chunk records for dataset '%s'.", embedding_file, dataset)
            continue

        vectors = load_embeddings(embedding_file)
        if vectors.shape[0] != len(chunk_records):
            raise ValueError(
                f"Embedding count {vectors.shape[0]} does not match chunk records {len(chunk_records)} for {embedding_file}"
            )

        collection_name = f"{dataset}_{model_key}"
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"dataset": dataset, "model_key": model_key},
        )

        ids = []
        documents = []
        metadatas = []
        for chunk in chunk_records:
            chunk_dataset = chunk.get("dataset", dataset)
            ids.append(f"{chunk_dataset}:{chunk['chunk_id']}")
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "dataset": chunk_dataset,
                    "document_id": chunk["document_id"],
                    "case_name": chunk.get("case_name"),
                    "relative_path": chunk.get("relative_path"),
                    "category": chunk.get("category"),
                    "court": chunk.get("court"),
                    "court_id": chunk.get("court_id"),
                    "docket_number": chunk.get("docket_number"),
                    "date_filed": chunk.get("date_filed"),
                    "topic": chunk.get("topic"),
                    "recommended_model": chunk.get("recommended_model"),
                    "start_word": chunk.get("start_word"),
                    "end_word": chunk.get("end_word"),
                }
            )

        batch_size = 2000
        LOGGER.info("Upserting %d vectors into collection '%s' in batches of %d", len(ids), collection_name, batch_size)
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                metadatas=metadatas[start:end],
                documents=documents[start:end],
                embeddings=vectors[start:end].tolist(),
            )

        collections_built.append(collection_name)

    if not collections_built:
        LOGGER.warning("No collections were built; check that embedding files exist.")
    else:
        LOGGER.info("Built collections: %s", ", ".join(collections_built))

    memory_manager = LawsuitMemoryManager()
    memory_manager.log_event(
        event_type="vector_index_build",
        summary="Updated Chroma vector index for available datasets.",
        metadata={
            "collections": collections_built,
            "datasets": {dataset: len(records) for dataset, records in chunks_by_dataset.items()},
            "chunk_count": total_chunks,
        },
        artifacts={
            "store_path": str(args.output_dir),
            "chunk_files": [str(path) for path in args.chunk_files],
        },
    )

    LOGGER.info("Vector index build complete. Stored at %s", args.output_dir)


if __name__ == "__main__":
    main()
