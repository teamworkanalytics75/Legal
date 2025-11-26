#!/usr/bin/env python3
"""Analyze lawsuit embeddings for keyword extraction and case-law similarity."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_embedding_tables(suffix: str) -> pd.DataFrame:
    base = Path("reports") / "analysis_outputs"
    emb_path = base / f"lawsuit_doc_embeddings__{suffix}.npz"
    meta_path = base / f"lawsuit_doc_chunks__{suffix}.json"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Embedding or metadata file missing for suffix '{suffix}'")

    embeddings = np.load(emb_path)["embeddings"]
    records = json.loads(meta_path.read_text(encoding="utf-8"))

    df = pd.DataFrame(records)
    df["embedding"] = list(embeddings)
    return df


def summarize_keywords(df: pd.DataFrame, *, top_n: int = 25) -> pd.DataFrame:
    doc_text = df.groupby("doc")["text"].apply(lambda texts: "\n".join(texts))
    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), stop_words="english")
    tfidf = vectorizer.fit_transform(doc_text.values)
    vocab = np.array(vectorizer.get_feature_names_out())

    rows: List[Dict[str, str]] = []
    for idx, doc in enumerate(doc_text.index):
        row = tfidf[idx].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        tokens = vocab[top_idx]
        scores = row[top_idx]
        rows.append(
            {
                "doc": doc,
                "keywords": ", ".join(f"{token} ({score:.3f})" for token, score in zip(tokens, scores)),
            }
        )
    return pd.DataFrame(rows)


def compute_doc_vectors(df: pd.DataFrame) -> pd.DataFrame:
    vectors = df.groupby("doc")["embedding"].apply(lambda rows: np.vstack(rows).mean(axis=0))
    return vectors.reset_index().rename(columns={"embedding": "vector"})


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def load_case_embeddings() -> pd.DataFrame:
    feat_path = Path("case_law_data") / "features" / "unified_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"Case features not found: {feat_path}")

    cols = ["cluster_id"] + [f"bert_{i}" for i in range(768)]
    df = pd.read_csv(feat_path, usecols=cols)
    return df


def build_similarity_table(doc_vectors: pd.DataFrame, cases: pd.DataFrame, top_k: int) -> pd.DataFrame:
    case_matrix = cases[[f"bert_{i}" for i in range(768)]].to_numpy(dtype=float)
    sim = cosine_sim_matrix(np.vstack(doc_vectors["vector"].to_numpy()), case_matrix)

    rows: List[Dict[str, float]] = []
    for i, doc in enumerate(doc_vectors["doc"]):
        sims = sim[i]
        top_idx = sims.argsort()[::-1][:top_k]
        for rank, idx in enumerate(top_idx, 1):
            rows.append(
                {
                    "doc": doc,
                    "rank": rank,
                    "similarity": float(sims[idx]),
                    "cluster_id": int(cases.iloc[idx]["cluster_id"]),
                }
            )
    return pd.DataFrame(rows)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword extraction and similarity vs case-law embeddings.")
    parser.add_argument("--suffix", default="legal-bert-base-uncased", help="Embedding suffix to analyze.")
    parser.add_argument("--top-k", type=int, default=25, help="Number of similar cases to retain per document.")
    parser.add_argument("--keyword-count", type=int, default=25, help="Number of keywords to output per document.")
    args = parser.parse_args()

    df = load_embedding_tables(args.suffix)
    keywords_df = summarize_keywords(df, top_n=args.keyword_count)
    doc_vectors = compute_doc_vectors(df)
    cases = load_case_embeddings()
    similarity_df = build_similarity_table(doc_vectors, cases, args.top_k)

    out_dir = Path("reports") / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = sanitize_filename(args.suffix)
    keywords_path = out_dir / f"lawsuit_keywords__{suffix}.csv"
    similarity_path = out_dir / f"lawsuit_similarity__{suffix}.csv"

    keywords_df.to_csv(keywords_path, index=False)
    similarity_df.to_csv(similarity_path, index=False)

    print(f"Keyword summary written to {keywords_path}")
    print(f"Similarity table written to {similarity_path}")


if __name__ == "__main__":
    main()
