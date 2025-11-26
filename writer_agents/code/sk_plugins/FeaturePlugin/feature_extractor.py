#!/usr/bin/env python3
"""
Feature Extractor - Handles feature extraction and caching.

Extracted from RefinementLoop to follow single responsibility principle.
"""

import hashlib
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Handles feature extraction from draft text with caching."""

    _personal_check_logged: bool = False

    def __init__(self, cache_timeout: timedelta = timedelta(minutes=5)):
        """
        Initialize FeatureExtractor.

        Args:
            cache_timeout: How long to cache feature extraction results
        """
        self._feature_cache: Dict[str, tuple] = {}
        self._cache_timeout = cache_timeout
        self._personal_metrics_ready = self._verify_personal_corpus_outputs()

    def extract_features(self, draft_text: str) -> Dict[str, float]:
        """
        Extract features from draft text.

        Uses caching to avoid re-extracting features for the same document.

        Args:
            draft_text: The draft text to extract features from

        Returns:
            Dictionary of feature name -> value
        """
        # Check cache first
        doc_hash = self._get_document_hash(draft_text)
        self._clear_expired_cache()

        if doc_hash in self._feature_cache:
            features, _ = self._feature_cache[doc_hash]
            logger.debug(f"Using cached feature extraction for document hash {doc_hash[:8]}...")
            return features

        # Extract features
        # Ensure the writer_agents/code directory is on sys.path so we can import
        # analyze_ma_motion_doc.py which lives alongside sk_plugins
        code_dir = Path(__file__).parents[3]
        if str(code_dir) not in sys.path:
            sys.path.append(str(code_dir))
        # Legacy fallback: if an analysis/ folder exists, include it as well
        analysis_dir = code_dir / "analysis"
        if analysis_dir.exists() and str(analysis_dir) not in sys.path:
            sys.path.append(str(analysis_dir))

        try:
            from analyze_ma_motion_doc import compute_draft_features
            start_time = datetime.now()
            features = compute_draft_features(draft_text)
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Feature extraction took {elapsed:.3f}s")

            # Cache the results
            self._feature_cache[doc_hash] = (features, datetime.now())
            logger.debug(f"Cached feature extraction for document hash {doc_hash[:8]}...")

            return features
        except ImportError as e:
            logger.error(f"Failed to import compute_draft_features: {e}")
            return {}
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

    def _get_document_hash(self, draft_text: str) -> str:
        """
        Generate hash for document caching.

        Uses first 1000 chars + length as hash key for cache.

        Args:
            draft_text: The document text

        Returns:
            MD5 hash string
        """
        # Use first 1000 chars + length as hash key for cache
        cache_key_text = draft_text[:1000] + str(len(draft_text))
        return hashlib.md5(cache_key_text.encode()).hexdigest()

    def _clear_expired_cache(self) -> None:
        """Remove expired entries from feature cache."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._feature_cache.items()
            if now - timestamp > self._cache_timeout
        ]
        for key in expired_keys:
            del self._feature_cache[key]
        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")

    def clear_cache(self) -> None:
        """Clear all cached features."""
        self._feature_cache.clear()
        logger.debug("Cleared all feature cache entries")

    def _verify_personal_corpus_outputs(self) -> bool:
        """
        Check that personal corpus artifacts exist so the enhanced feature set
        (CatBoost deltas + semantic retrieval) can rely on real data.
        """
        if FeatureExtractor._personal_check_logged:
            # Already ran the check in this process.
            return True

        results_dir = Path(__file__).parents[4] / "case_law_data" / "results"
        required = {
            "personal_corpus_features.csv": results_dir / "personal_corpus_features.csv",
            "personal_corpus_aggregated_statistics.json": results_dir / "personal_corpus_aggregated_statistics.json",
        }

        missing = [name for name, path in required.items() if not path.exists()]
        FeatureExtractor._personal_check_logged = True

        if missing:
            logger.warning(
                "Personal corpus metrics missing (%s). "
                "New CatBoost features will fall back to generic heuristics until "
                "you run writer_agents/scripts/refresh_personal_corpus.py.",
                ", ".join(missing),
            )
            return False

        logger.info(
            "Personal corpus metrics detected (%s) – enabling personalized feature deltas.",
            ", ".join(required.keys()),
        )
        return True
