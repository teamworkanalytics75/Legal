#!/usr/bin/env python3
"""
Constraint Data Loader - Centralized loading of optimal ranges from analysis JSON.

Loads section-specific thresholds and feature importance scores from CatBoost analysis results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Default paths to analysis results
# Calculate path relative to this file: writer_agents/code/sk_plugins/FeaturePlugin/constraint_data_loader.py
# Need to go up to project root (TheMatrix): parents[4]
# Then: case_law_data/analysis/
DEFAULT_ANALYSIS_DIR = Path(__file__).parents[4] / "case_law_data" / "analysis"
DEFAULT_ANALYSIS_PATH = DEFAULT_ANALYSIS_DIR / "section_optimal_thresholds.json"
DEFAULT_IMPORTANCE_PATH = DEFAULT_ANALYSIS_DIR / "section_feature_importance.json"


class ConstraintDataLoader:
    """Loads constraint thresholds and feature importance from analysis results."""
    
    def __init__(self, analysis_path: Optional[Path] = None, importance_path: Optional[Path] = None):
        """
        Initialize constraint data loader.
        
        Args:
            analysis_path: Path to section_optimal_thresholds.json. If None, uses default path.
            importance_path: Path to section_feature_importance.json. If None, uses default path.
        """
        self.analysis_path = analysis_path or DEFAULT_ANALYSIS_PATH
        self.importance_path = importance_path or DEFAULT_IMPORTANCE_PATH
        self._data: Optional[Dict[str, Any]] = None
        self._importance_data: Optional[Dict[str, Any]] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load analysis data from JSON files."""
        # Load thresholds
        try:
            if self.analysis_path.exists():
                with open(self.analysis_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                logger.debug(f"Loaded constraint data from {self.analysis_path}")
            else:
                logger.warning(f"Analysis file not found: {self.analysis_path}, using defaults")
                self._data = {}
        except Exception as e:
            logger.warning(f"Failed to load constraint data: {e}, using defaults")
            self._data = {}
        
        # Load feature importance
        try:
            if self.importance_path.exists():
                with open(self.importance_path, 'r', encoding='utf-8') as f:
                    self._importance_data = json.load(f)
                logger.debug(f"Loaded feature importance data from {self.importance_path}")
            else:
                logger.debug(f"Feature importance file not found: {self.importance_path}, will use fallbacks")
                self._importance_data = {}
        except Exception as e:
            logger.warning(f"Failed to load feature importance data: {e}, will use fallbacks")
            self._importance_data = {}
    
    def load_section_thresholds(self, section_name: str) -> Dict[str, Any]:
        """
        Load all thresholds for a specific section.
        
        Args:
            section_name: Name of section (e.g., 'legal_standard', 'balancing_test')
            
        Returns:
            Dictionary with thresholds:
            - word_count: {'optimal_range': [min, max], 'mean_value': float, ...}
            - paragraph_count: {'optimal_range': [min, max], ...}
            - avg_words_per_paragraph: {'optimal_range': [min, max], ...}
            - enumeration_depth: {'optimal_threshold': float, ...}
        """
        if not self._data or section_name not in self._data:
            logger.debug(f"No thresholds found for section: {section_name}")
            return {}
        
        return self._data[section_name]
    
    def get_word_count_range(self, section_name: str) -> Optional[Tuple[float, float]]:
        """
        Get optimal word count range for a section from CatBoost analysis.
        
        Uses optimal_range (Q25-Q75 from successful cases) if available,
        otherwise falls back to optimal_threshold from ROC-AUC analysis.
        ROC-AUC aware: wider margins for weak signals (≈0.50), tighter for stronger (≥0.56).
        
        Args:
            section_name: Name of section
            
        Returns:
            Tuple of (min, max) word count, or None if not found
        """
        thresholds = self.load_section_thresholds(section_name)
        word_count_data = thresholds.get('word_count', {})
        
        # Prefer optimal_range (Q25-Q75 from successful cases)
        optimal_range = word_count_data.get('optimal_range')
        if optimal_range and len(optimal_range) == 2:
            roc_auc = word_count_data.get('roc_auc', 0.5)
            logger.debug(f"Found optimal_range for {section_name}.word_count: {optimal_range} (roc_auc={roc_auc:.3f})")
            return (float(optimal_range[0]), float(optimal_range[1]))
        
        # Fallback: use optimal_threshold from ROC-AUC analysis
        optimal_threshold = word_count_data.get('optimal_threshold')
        roc_auc = word_count_data.get('roc_auc', 0.5)
        successful_mean = word_count_data.get('successful_mean')
        
        if optimal_threshold is not None:
            # ROC-AUC aware margin: wider for weak signals, tighter for stronger
            if roc_auc < 0.52:
                margin = optimal_threshold * 0.3  # Wider margin for weak signals (roc_auc ≈ 0.50)
                logger.debug(f"Using fallback threshold with wide margin (roc_auc={roc_auc:.3f}) for {section_name}.word_count")
            elif roc_auc >= 0.56:
                margin = optimal_threshold * 0.15  # Tighter margin for stronger signals (roc_auc ≥ 0.56)
                logger.debug(f"Using fallback threshold with tight margin (roc_auc={roc_auc:.3f}) for {section_name}.word_count")
            else:
                margin = optimal_threshold * 0.2  # Default margin
                logger.debug(f"Using fallback threshold with default margin (roc_auc={roc_auc:.3f}) for {section_name}.word_count")
            return (float(max(0, optimal_threshold - margin)), float(optimal_threshold + margin))
        elif successful_mean is not None:
            margin = successful_mean * 0.3
            logger.warning(f"Using successful_mean fallback for {section_name}.word_count (no optimal_range or optimal_threshold found)")
            return (float(max(0, successful_mean - margin)), float(successful_mean + margin))
        
        logger.warning(f"No word_count data found for {section_name} (missing: optimal_range, optimal_threshold, successful_mean)")
        return None
    
    def get_paragraph_count_range(self, section_name: str) -> Optional[Tuple[float, float]]:
        """
        Get optimal paragraph count range for a section.
        
        Uses optimal_range if available, otherwise falls back to optimal_threshold.
        """
        thresholds = self.load_section_thresholds(section_name)
        para_data = thresholds.get('paragraph_count', {})
        
        optimal_range = para_data.get('optimal_range')
        if optimal_range and len(optimal_range) == 2:
            logger.debug(f"Found optimal_range for {section_name}.paragraph_count: {optimal_range}")
            return (float(optimal_range[0]), float(optimal_range[1]))
        
        # Fallback: use optimal_threshold
        optimal_threshold = para_data.get('optimal_threshold')
        roc_auc = para_data.get('roc_auc', 0.5)
        successful_mean = para_data.get('successful_mean')
        
        if optimal_threshold is not None:
            # ROC-AUC aware margin
            if roc_auc < 0.52:
                margin = optimal_threshold * 0.3
                logger.debug(f"Using fallback threshold with wide margin (roc_auc={roc_auc:.3f}) for {section_name}.paragraph_count")
            else:
                margin = optimal_threshold * 0.2
                logger.debug(f"Using fallback threshold with tight margin (roc_auc={roc_auc:.3f}) for {section_name}.paragraph_count")
            return (float(max(0, optimal_threshold - margin)), float(optimal_threshold + margin))
        elif successful_mean is not None:
            margin = successful_mean * 0.3
            logger.warning(f"Using successful_mean fallback for {section_name}.paragraph_count (no optimal_range or optimal_threshold found)")
            return (float(max(0, successful_mean - margin)), float(successful_mean + margin))
        
        logger.warning(f"No paragraph_count data found for {section_name} (missing: optimal_range, optimal_threshold, successful_mean)")
        return None
    
    def get_avg_words_per_paragraph_range(self, section_name: str) -> Optional[Tuple[float, float]]:
        """
        Get optimal average words per paragraph range for a section.
        
        Uses optimal_range if available, otherwise falls back to optimal_threshold.
        """
        thresholds = self.load_section_thresholds(section_name)
        avg_words_data = thresholds.get('avg_words_per_paragraph', {})
        
        optimal_range = avg_words_data.get('optimal_range')
        if optimal_range and len(optimal_range) == 2:
            logger.debug(f"Found optimal_range for {section_name}.avg_words_per_paragraph: {optimal_range}")
            return (float(optimal_range[0]), float(optimal_range[1]))
        
        # Fallback: use optimal_threshold
        optimal_threshold = avg_words_data.get('optimal_threshold')
        roc_auc = avg_words_data.get('roc_auc', 0.5)
        successful_mean = avg_words_data.get('successful_mean')
        
        if optimal_threshold is not None:
            # ROC-AUC aware margin
            if roc_auc < 0.52:
                margin = optimal_threshold * 0.3
                logger.debug(f"Using fallback threshold with wide margin (roc_auc={roc_auc:.3f}) for {section_name}.avg_words_per_paragraph")
            else:
                margin = optimal_threshold * 0.2
                logger.debug(f"Using fallback threshold with tight margin (roc_auc={roc_auc:.3f}) for {section_name}.avg_words_per_paragraph")
            return (float(max(0, optimal_threshold - margin)), float(optimal_threshold + margin))
        elif successful_mean is not None:
            margin = successful_mean * 0.3
            logger.warning(f"Using successful_mean fallback for {section_name}.avg_words_per_paragraph (no optimal_range or optimal_threshold found)")
            return (float(max(0, successful_mean - margin)), float(successful_mean + margin))
        
        logger.warning(f"No avg_words_per_paragraph data found for {section_name} (missing: optimal_range, optimal_threshold, successful_mean)")
        return None
    
    def get_sentences_per_paragraph_range(self, section_name: str) -> Optional[Tuple[float, float]]:
        """
        Get optimal sentences per paragraph range for a section.
        
        Uses optimal_range (Q25-Q75) if available, otherwise falls back to optimal_threshold.
        """
        thresholds = self.load_section_thresholds(section_name)
        # Sentence-level features are stored directly in section (not in paragraph_analysis)
        sentences_data = thresholds.get('sentences_per_paragraph', {})
        
        # Prefer optimal_range (Q25-Q75 from successful cases)
        optimal_range = sentences_data.get('optimal_range')
        if optimal_range and len(optimal_range) == 2:
            logger.debug(f"Found optimal_range for {section_name}.sentences_per_paragraph: {optimal_range}")
            return (float(optimal_range[0]), float(optimal_range[1]))
        
        # Fallback: use optimal_threshold with margin
        optimal_threshold = sentences_data.get('optimal_threshold')
        roc_auc = sentences_data.get('roc_auc', 0.5)
        successful_mean = sentences_data.get('successful_mean')
        
        if optimal_threshold is not None:
            # For weak ROC-AUC (≈0.50), use wider margin; for stronger, use tighter
            if roc_auc < 0.52:
                margin = optimal_threshold * 0.3  # Wider margin for weak signals
                logger.debug(f"Using fallback threshold with wide margin (roc_auc={roc_auc:.3f}) for {section_name}.sentences_per_paragraph")
            else:
                margin = optimal_threshold * 0.2  # Tighter margin for stronger signals
                logger.debug(f"Using fallback threshold with tight margin (roc_auc={roc_auc:.3f}) for {section_name}.sentences_per_paragraph")
            return (float(max(0, optimal_threshold - margin)), float(optimal_threshold + margin))
        elif successful_mean is not None:
            margin = successful_mean * 0.3
            logger.warning(f"Using successful_mean fallback for {section_name}.sentences_per_paragraph (no optimal_range or optimal_threshold found)")
            return (float(max(0, successful_mean - margin)), float(successful_mean + margin))
        
        logger.warning(f"No sentences_per_paragraph data found for {section_name} (missing: optimal_range, optimal_threshold, successful_mean)")
        return None
    
    def get_words_per_sentence_range(self, section_name: str) -> Optional[Tuple[float, float]]:
        """
        Get optimal words per sentence range for a section.
        
        Uses optimal_range (Q25-Q75) if available, otherwise falls back to optimal_threshold.
        """
        thresholds = self.load_section_thresholds(section_name)
        # Sentence-level features are stored directly in section (not in paragraph_analysis)
        words_data = thresholds.get('words_per_sentence', {})
        
        # Prefer optimal_range (Q25-Q75 from successful cases)
        optimal_range = words_data.get('optimal_range')
        if optimal_range and len(optimal_range) == 2:
            logger.debug(f"Found optimal_range for {section_name}.words_per_sentence: {optimal_range}")
            return (float(optimal_range[0]), float(optimal_range[1]))
        
        # Fallback: use optimal_threshold with margin
        optimal_threshold = words_data.get('optimal_threshold')
        roc_auc = words_data.get('roc_auc', 0.5)
        successful_mean = words_data.get('successful_mean')
        
        if optimal_threshold is not None:
            # For weak ROC-AUC (≈0.50), use wider margin; for stronger, use tighter
            if roc_auc < 0.52:
                margin = optimal_threshold * 0.3  # Wider margin for weak signals
                logger.debug(f"Using fallback threshold with wide margin (roc_auc={roc_auc:.3f}) for {section_name}.words_per_sentence")
            else:
                margin = optimal_threshold * 0.2  # Tighter margin for stronger signals
                logger.debug(f"Using fallback threshold with tight margin (roc_auc={roc_auc:.3f}) for {section_name}.words_per_sentence")
            return (float(max(0, optimal_threshold - margin)), float(optimal_threshold + margin))
        elif successful_mean is not None:
            margin = successful_mean * 0.3
            logger.warning(f"Using successful_mean fallback for {section_name}.words_per_sentence (no optimal_range or optimal_threshold found)")
            return (float(max(0, successful_mean - margin)), float(successful_mean + margin))
        
        logger.warning(f"No words_per_sentence data found for {section_name} (missing: optimal_range, optimal_threshold, successful_mean)")
        return None
    
    def get_enumeration_depth_threshold(self, section_name: str) -> Optional[float]:
        """Get optimal enumeration depth threshold for a section."""
        thresholds = self.load_section_thresholds(section_name)
        enum_data = thresholds.get('enumeration_depth', {})
        threshold = enum_data.get('optimal_threshold')
        
        if threshold is not None:
            return float(threshold)
        return None
    
    def get_feature_importance(self, section_name: str, feature_name: str) -> float:
        """
        Get feature importance score for a feature in a section from CatBoost analysis.
        
        Prefers section-specific importance, falls back to global or known importances.
        Includes feature name mapping for synonyms (e.g., sentences_per_paragraph_mean -> sentences_per_paragraph).
        
        Args:
            section_name: Name of section (e.g., 'legal_standard', 'document' for document-level)
            feature_name: Name of feature (e.g., 'word_count', 'chars_per_word', 'max_enumeration_depth')
            
        Returns:
            Feature importance score (higher = more important), or 0.0 if not found
        """
        # Feature name mapping for synonyms (importance JSON may use different names)
        FEATURE_NAME_MAPPING = {
            'sentences_per_paragraph': ['sentences_per_paragraph_mean', 'sentences_per_paragraph_median', 'sentences_per_paragraph_max'],
            'words_per_sentence': ['words_per_sentence_mean', 'words_per_sentence_median', 'words_per_sentence_max', 'words_per_sentence_q25', 'words_per_sentence_q75'],
            'enumeration_depth': ['enumeration_density', 'enumeration_count'],
            'char_count': ['chars_per_word'],
            'paragraph_structure': ['avg_words_per_paragraph', 'paragraph_count'],
            'sentence_count': ['sentence_count'],
        }
        
        # Build list of feature names to try (original + synonyms)
        feature_names_to_try = [feature_name]
        for base_name, synonyms in FEATURE_NAME_MAPPING.items():
            if feature_name == base_name:
                feature_names_to_try.extend(synonyms)
            elif feature_name in synonyms:
                feature_names_to_try.append(base_name)
                feature_names_to_try.extend([s for s in synonyms if s != feature_name])
        
        # Try section-specific importance first
        if self._importance_data and section_name in self._importance_data:
            section_importance = self._importance_data[section_name]
            
            # Find matching feature in top_features list (try all name variants)
            if isinstance(section_importance, list):
                for feature_info in section_importance:
                    importance_feature_name = None
                    importance_value = None
                    
                    if isinstance(feature_info, dict):
                        importance_feature_name = feature_info.get('feature')
                        importance_value = float(feature_info.get('importance', 0.0))
                    elif isinstance(feature_info, list) and len(feature_info) == 2:
                        # Format: [feature_name, importance]
                        importance_feature_name = feature_info[0]
                        importance_value = float(feature_info[1])
                    
                    # Check if this feature matches any of our name variants
                    if importance_feature_name and importance_feature_name in feature_names_to_try:
                        if importance_value > 0:
                            logger.debug(f"Found section-specific importance for {section_name}.{feature_name} (matched as {importance_feature_name}): {importance_value}")
                            return importance_value
            
            # Also check feature_importance dict if it exists
            elif isinstance(section_importance, dict):
                for name_variant in feature_names_to_try:
                    if name_variant in section_importance:
                        importance = float(section_importance[name_variant])
                        if importance > 0:
                            logger.debug(f"Found section-specific importance for {section_name}.{feature_name} (matched as {name_variant}): {importance}")
                            return importance
        
        # Try global/document-level importance (simple scan across sections)
        if self._importance_data:
            # Look for document-level features across all sections
            for section, section_data in self._importance_data.items():
                if isinstance(section_data, list):
                    for feature_info in section_data:
                        importance_feature_name = None
                        importance_value = None
                        
                        if isinstance(feature_info, dict):
                            importance_feature_name = feature_info.get('feature')
                            importance_value = float(feature_info.get('importance', 0.0))
                        elif isinstance(feature_info, list) and len(feature_info) == 2:
                            importance_feature_name = feature_info[0]
                            importance_value = float(feature_info[1])
                        
                        if importance_feature_name and importance_feature_name in feature_names_to_try:
                            if importance_value > 0:
                                logger.debug(f"Found global importance for {feature_name} (matched as {importance_feature_name}): {importance_value}")
                                return importance_value
        
        # Fallback: return default importance based on known feature rankings
        known_importances = {
            'transition_legal_standard_to_factual_background': 64.80,
            'has_bullet_points': 31.56,
            'max_enumeration_depth': 27.27,
            'enumeration_depth': 27.27,  # Synonym
            'paragraph_count': 24.78,
            'avg_words_per_paragraph': 8.13,
            'word_count': 10.0,  # Default high importance
            'chars_per_word': 8.0,  # High importance (often top feature)
            'char_count': 8.0,  # Synonym
            'words_per_sentence': 7.0,
            'sentences_per_paragraph': 5.0,
            'sentence_count': 5.0,
        }
        fallback_importance = known_importances.get(feature_name, 1.0)
        logger.debug(f"Using fallback importance for {section_name}.{feature_name}: {fallback_importance} (no match found in importance data)")
        return fallback_importance


# Global instance for easy access
_loader_instance: Optional[ConstraintDataLoader] = None


def get_loader(analysis_path: Optional[Path] = None, importance_path: Optional[Path] = None) -> ConstraintDataLoader:
    """Get or create global ConstraintDataLoader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ConstraintDataLoader(analysis_path, importance_path)
    return _loader_instance
