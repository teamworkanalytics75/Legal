#!/usr/bin/env python3
"""
Lightweight NLP analysis for the 1782 RECAP text corpus.

This script assumes that `data/case_law/1782_text` contains plain-text
files produced by `extract_text_1782_pdfs.py`. It generates:

1. Corpus-level statistics (document length, token counts).
2. Aggregated named-entity counts via spaCy (`en_core_web_sm`).
3. TF-IDF features with top terms overall and per cluster.
4. KMeans clustering to surface thematic groupings.
5. Markdown/JSON reports under `data/case_law/analysis_results/`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEXT_DIR = Path("data/case_law/1782_text")
SUMMARY_PATH = TEXT_DIR / "extraction_summary.json"
OUTPUT_DIR = Path("data/case_law/analysis_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_texts() -> Tuple[List[str], List[str]]:
    """Return lists of filenames and corresponding document text."""
    if not TEXT_DIR.exists():
        raise FileNotFoundError(f"Text directory not found: {TEXT_DIR}")

    files = sorted(TEXT_DIR.glob("*.txt"))
    filenames: List[str] = []
    texts: List[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        filenames.append(path.name)
        texts.append(text)
    logger.info("Loaded %s text files", len(texts))
    return filenames, texts


def summarize_lengths(texts: List[str]) -> Dict[str, float]:
    lengths = np.array([len(t) for t in texts])
    tokens = np.array([len(t.split()) for t in texts])
    return {
        "documents": int(lengths.size),
        "chars_mean": float(np.mean(lengths)),
        "chars_median": float(np.median(lengths)),
        "chars_min": float(np.min(lengths)),
        "chars_max": float(np.max(lengths)),
        "tokens_mean": float(np.mean(tokens)),
        "tokens_median": float(np.median(tokens)),
        "tokens_min": float(np.min(tokens)),
        "tokens_max": float(np.max(tokens)),
    }


def run_spacy_ner(texts: List[str], filenames: List[str]) -> Dict[str, int]:
    """Aggregate named-entity stats using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000
    max_chars = 400_000
    entity_counter: Counter = Counter()
    doc_entities: Dict[str, List[Tuple[str, str]]] = {}

    truncated_texts = [text[:max_chars] for text in texts]

    for doc, fname in zip(nlp.pipe(truncated_texts, batch_size=32), filenames):
        ents = [(ent.label_, ent.text) for ent in doc.ents]
        doc_entities[fname] = ents
        for label, _ in ents:
            entity_counter[label] += 1

    # Persist document-level entity lists for downstream use.
    ent_path = OUTPUT_DIR / "1782_doc_entities.json"
    with ent_path.open("w", encoding="utf-8") as handle:
        json.dump(doc_entities, handle, indent=2)
    logger.info("Entity annotations written to %s", ent_path)

    return entity_counter


def build_tfidf(texts: List[str], max_features: int = 5000) -> Tuple[np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)
    logger.info("TF-IDF matrix shape: %s", matrix.shape)
    return matrix, vectorizer


def top_terms(vectorizer: TfidfVectorizer, matrix, top_k: int = 25) -> List[Tuple[str, float]]:
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    feature_names = np.array(vectorizer.get_feature_names_out())
    ranking = np.argsort(scores)[::-1][:top_k]
    return [(feature_names[i], float(scores[i])) for i in ranking]


def cluster_documents(matrix, n_clusters: int = 6) -> Tuple[np.ndarray, float]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(matrix)
    sil = silhouette_score(matrix, labels) if n_clusters > 1 else 0.0
    logger.info("KMeans silhouette score: %.3f", sil)
    return labels, sil


def cluster_top_terms(labels: np.ndarray, matrix, vectorizer: TfidfVectorizer, top_k: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_terms: Dict[int, List[Tuple[str, float]]] = {}
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        cluster_matrix = matrix[mask]
        scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        ranking = np.argsort(scores)[::-1][:top_k]
        cluster_terms[cluster_id] = [(feature_names[i], float(scores[i])) for i in ranking]
    return cluster_terms


def load_ocr_pending() -> List[str]:
    if not SUMMARY_PATH.exists():
        return []
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    pending = [entry["pdf"] for entry in data.get("details", []) if entry.get("note")]
    return pending


def write_reports(summary: Dict[str, object], analysis_md: Path, analysis_json: Path) -> None:
    analysis_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# 1782 Text Corpus – NLP Snapshot")
    lines.append("")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("")
    lines.append("## Corpus Stats")
    corpus = summary["corpus_stats"]
    lines.append(f"- Documents analysed: {corpus['documents']}")
    lines.append(f"- Avg length: {corpus['chars_mean']:.0f} chars / {corpus['tokens_mean']:.0f} tokens")
    lines.append(f"- Median length: {corpus['chars_median']:.0f} chars")
    lines.append(f"- Min/Max length: {corpus['chars_min']:.0f}–{corpus['chars_max']:.0f} chars")
    lines.append("")

    lines.append("## Named Entities (top 10 labels)")
    for label, count in summary["entity_counts"][:10]:
        lines.append(f"- {label}: {count}")
    lines.append("")

    lines.append("## Top TF-IDF Terms")
    for term, score in summary["top_terms"][:20]:
        lines.append(f"- {term} — {score:.4f}")
    lines.append("")

    lines.append("## Cluster Overview")
    lines.append(f"- Clusters: {summary['clusters']['k']} (silhouette {summary['clusters']['silhouette']:.3f})")
    for cid, terms in summary["clusters"]["top_terms"].items():
        term_string = ", ".join(term for term, _ in terms[:8])
        lines.append(f"  - Cluster {cid}: {term_string}")
    lines.append("")

    if summary["ocr_pending"]:
        lines.append("## Needs OCR Follow-up")
        for pdf in summary["ocr_pending"]:
            lines.append(f"- {pdf}")
        lines.append("")

    lines.append("---")
    lines.append("Generated via `scripts/analyze_1782_texts.py`.")

    analysis_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    filenames, texts = load_texts()
    if not texts:
        logger.error("No text files found for analysis.")
        return

    corpus_stats = summarize_lengths(texts)
    entity_counts = run_spacy_ner(texts, filenames)

    matrix, vectorizer = build_tfidf(texts)
    overall_terms = top_terms(vectorizer, matrix)

    labels, sil = cluster_documents(matrix, n_clusters=6)
    cluster_terms = cluster_top_terms(labels, matrix, vectorizer)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "documents": len(texts),
        "corpus_stats": corpus_stats,
        "entity_counts": entity_counts.most_common(),
        "top_terms": overall_terms,
        "clusters": {
            "k": 6,
            "silhouette": sil,
            "labels": labels.tolist(),
            "top_terms": cluster_terms,
        },
        "ocr_pending": load_ocr_pending(),
        "filenames": filenames,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_json = OUTPUT_DIR / f"1782_text_analysis_{timestamp}.json"
    analysis_md = OUTPUT_DIR / f"1782_text_analysis_{timestamp}.md"

    write_reports(summary, analysis_md, analysis_json)
    logger.info("Analysis complete. Reports at %s / %s", analysis_json, analysis_md)


if __name__ == "__main__":
    main()
