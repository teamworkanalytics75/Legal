"""Feature engineering for ML system.

This module provides feature extraction from:
1. Legal text (TF-IDF, embeddings, complexity metrics)
2. Case metadata (court, jurisdiction, dates)
3. Agent execution patterns (performance, efficiency)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
from datetime import datetime
import logging

# Import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Embedding features will be disabled.")

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Extract features from legal text."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model.

        Args:
            embedding_model: Name of sentence transformer model
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.tfidf_vectorizer = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = None

    def extract_text_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract comprehensive text features.

        Args:
            texts: List of legal texts

        Returns:
            Dictionary of feature arrays
        """
        features = {}

        # Basic text features
        features.update(self._extract_basic_features(texts))

        # TF-IDF features
        features.update(self._extract_tfidf_features(texts))

        # Embedding features
        if self.embedding_model:
            features.update(self._extract_embedding_features(texts))

        # Legal-specific features
        features.update(self._extract_legal_features(texts))

        return features

    def _extract_basic_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract basic text statistics."""
        features = {}

        # Text length features
        char_lengths = [len(text) for text in texts]
        word_lengths = [len(text.split()) for text in texts]
        sentence_lengths = [len(re.split(r'[.!?]+', text)) for text in texts]

        features['char_length'] = np.array(char_lengths)
        features['word_length'] = np.array(word_lengths)
        features['sentence_length'] = np.array(sentence_lengths)

        # Average word length
        avg_word_lengths = []
        for text in texts:
            words = text.split()
            if words:
                avg_word_lengths.append(np.mean([len(word) for word in words]))
            else:
                avg_word_lengths.append(0)
        features['avg_word_length'] = np.array(avg_word_lengths)

        # Average sentence length
        avg_sentence_lengths = []
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            if sentences:
                avg_sentence_lengths.append(np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()]))
            else:
                avg_sentence_lengths.append(0)
        features['avg_sentence_length'] = np.array(avg_sentence_lengths)

        return features

    def _extract_tfidf_features(self, texts: List[str], max_features: int = 1000) -> Dict[str, np.ndarray]:
        """Extract TF-IDF features."""
        if not self.tfidf_vectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)

        return {'tfidf': tfidf_matrix.toarray()}

    def _extract_embedding_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract sentence embedding features."""
        try:
            embeddings = self.embedding_model.encode(texts)
            return {'embeddings': embeddings}
        except Exception as e:
            logger.warning(f"Failed to extract embeddings: {e}")
            return {}

    def _extract_legal_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract legal-specific features."""
        features = {}

        # Legal citation patterns
        citation_counts = []
        case_citation_counts = []
        statute_citation_counts = []

        for text in texts:
            # Count citations (e.g., "Smith v. Jones", "123 F.3d 456")
            citations = len(re.findall(r'\b[A-Z][a-z]+ v\. [A-Z][a-z]+\b', text))
            case_citations = len(re.findall(r'\b\d+ [A-Z]\.?\d+[a-z]? \d+\b', text))
            statute_citations = len(re.findall(r'\b\d+ U\.S\.C\. \d+\b', text))

            citation_counts.append(citations)
            case_citation_counts.append(case_citations)
            statute_citation_counts.append(statute_citations)

        features['citation_count'] = np.array(citation_counts)
        features['case_citation_count'] = np.array(case_citation_counts)
        features['statute_citation_count'] = np.array(statute_citation_counts)

        # Legal terminology density
        legal_terms = [
            'plaintiff', 'defendant', 'court', 'judge', 'jury', 'trial',
            'evidence', 'testimony', 'witness', 'attorney', 'counsel',
            'motion', 'objection', 'ruling', 'decision', 'opinion',
            'appeal', 'jurisdiction', 'venue', 'standing', 'merits'
        ]

        legal_term_counts = []
        for text in texts:
            text_lower = text.lower()
            count = sum(text_lower.count(term) for term in legal_terms)
            legal_term_counts.append(count)

        features['legal_term_count'] = np.array(legal_term_counts)

        # Procedural language indicators
        procedural_terms = [
            'motion', 'objection', 'ruling', 'order', 'judgment',
            'appeal', 'brief', 'hearing', 'trial', 'discovery'
        ]

        procedural_counts = []
        for text in texts:
            text_lower = text.lower()
            count = sum(text_lower.count(term) for term in procedural_terms)
            procedural_counts.append(count)

        features['procedural_term_count'] = np.array(procedural_counts)

        return features


class MetadataFeatureExtractor:
    """Extract features from case metadata."""

    def __init__(self):
        """Initialize feature extractors."""
        self.court_encoder = LabelEncoder()
        self.case_type_encoder = LabelEncoder()
        self.jurisdiction_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def extract_case_metadata_features(self, case_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract features from case metadata.

        Args:
            case_data: DataFrame with case metadata

        Returns:
            Dictionary of feature arrays
        """
        features = {}

        # Court type encoding
        if 'court' in case_data.columns:
            features.update(self._extract_court_features(case_data['court']))

        # Case type encoding
        if 'case_type' in case_data.columns:
            features.update(self._extract_case_type_features(case_data['case_type']))

        # Date features
        if 'date_filed' in case_data.columns:
            features.update(self._extract_date_features(case_data['date_filed']))

        # Jurisdiction features
        if 'jurisdiction_label' in case_data.columns:
            features.update(self._extract_jurisdiction_features(case_data['jurisdiction_label']))

        return features

    def _extract_court_features(self, courts: pd.Series) -> Dict[str, np.ndarray]:
        """Extract court-related features."""
        features = {}

        # Court type (federal/state/supreme)
        court_types = []
        for court in courts:
            court_str = str(court).lower()
            if 'supreme' in court_str:
                court_types.append('supreme')
            elif 'federal' in court_str or 'district' in court_str or 'circuit' in court_str:
                court_types.append('federal')
            else:
                court_types.append('state')

        # One-hot encode court types
        court_type_df = pd.get_dummies(court_types, prefix='court_type')
        for col in court_type_df.columns:
            features[col] = court_type_df[col].values

        # Court level (numeric)
        court_levels = []
        for court in courts:
            court_str = str(court).lower()
            if 'supreme' in court_str:
                court_levels.append(3)  # Supreme
            elif 'circuit' in court_str:
                court_levels.append(2)  # Circuit
            elif 'district' in court_str:
                court_levels.append(1)  # District
            else:
                court_levels.append(0)  # State/local

        features['court_level'] = np.array(court_levels)

        return features

    def _extract_case_type_features(self, case_types: pd.Series) -> Dict[str, np.ndarray]:
        """Extract case type features."""
        features = {}

        # One-hot encode case types
        case_type_df = pd.get_dummies(case_types, prefix='case_type')
        for col in case_type_df.columns:
            features[col] = case_type_df[col].values

        return features

    def _extract_date_features(self, dates: pd.Series) -> Dict[str, np.ndarray]:
        """Extract date-related features."""
        features = {}

        years = []
        months = []
        decades = []

        for date in dates:
            try:
                if pd.isna(date):
                    years.append(0)
                    months.append(0)
                    decades.append(0)
                    continue

                if isinstance(date, str):
                    date_obj = datetime.strptime(date, '%Y-%m-%d')
                else:
                    date_obj = date

                year = date_obj.year
                month = date_obj.month
                decade = (year // 10) * 10

                years.append(year)
                months.append(month)
                decades.append(decade)

            except (ValueError, TypeError):
                years.append(0)
                months.append(0)
                decades.append(0)

        features['year'] = np.array(years)
        features['month'] = np.array(months)
        features['decade'] = np.array(decades)

        # Normalize years (relative to 2000)
        features['year_normalized'] = np.array([max(0, year - 2000) for year in years])

        return features

    def _extract_jurisdiction_features(self, jurisdictions: pd.Series) -> Dict[str, np.ndarray]:
        """Extract jurisdiction features."""
        features = {}

        # One-hot encode jurisdictions
        jurisdiction_df = pd.get_dummies(jurisdictions, prefix='jurisdiction')
        for col in jurisdiction_df.columns:
            features[col] = jurisdiction_df[col].values

        return features


class AgentFeatureExtractor:
    """Extract features from agent execution data."""

    def __init__(self):
        """Initialize feature extractors."""
        self.job_type_encoder = LabelEncoder()
        self.phase_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def extract_agent_features(self, agent_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract features from agent execution data.

        Args:
            agent_data: DataFrame with agent execution data

        Returns:
            Dictionary of feature arrays
        """
        features = {}

        # Job type features
        if 'job_type' in agent_data.columns:
            features.update(self._extract_job_type_features(agent_data['job_type']))

        # Phase features
        if 'phase' in agent_data.columns:
            features.update(self._extract_phase_features(agent_data['phase']))

        # Performance features
        features.update(self._extract_performance_features(agent_data))

        # Resource usage features
        features.update(self._extract_resource_features(agent_data))

        return features

    def _extract_job_type_features(self, job_types: pd.Series) -> Dict[str, np.ndarray]:
        """Extract job type features."""
        features = {}

        # One-hot encode job types
        job_type_df = pd.get_dummies(job_types, prefix='job_type')
        for col in job_type_df.columns:
            features[col] = job_type_df[col].values

        return features

    def _extract_phase_features(self, phases: pd.Series) -> Dict[str, np.ndarray]:
        """Extract phase features."""
        features = {}

        # One-hot encode phases
        phase_df = pd.get_dummies(phases, prefix='phase')
        for col in phase_df.columns:
            features[col] = phase_df[col].values

        return features

    def _extract_performance_features(self, agent_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract performance-related features."""
        features = {}

        # Duration features
        if 'duration_seconds' in agent_data.columns:
            durations = agent_data['duration_seconds'].fillna(0)
            features['duration'] = durations.values
            features['log_duration'] = np.log1p(durations).values

        # Token efficiency
        if 'tokens_in' in agent_data.columns and 'tokens_out' in agent_data.columns:
            tokens_in = agent_data['tokens_in'].fillna(0)
            tokens_out = agent_data['tokens_out'].fillna(0)

            # Avoid division by zero
            efficiency = np.where(tokens_in > 0, tokens_out / tokens_in, 0)
            features['token_efficiency'] = efficiency.values

        # Retry count
        if 'retry_count' in agent_data.columns:
            retry_counts = agent_data['retry_count'].fillna(0)
            features['retry_count'] = retry_counts.values

        return features

    def _extract_resource_features(self, agent_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract resource usage features."""
        features = {}

        # Budget utilization
        if 'budget_tokens' in agent_data.columns and 'tokens_out' in agent_data.columns:
            budget_tokens = agent_data['budget_tokens'].fillna(0)
            tokens_out = agent_data['tokens_out'].fillna(0)

            # Avoid division by zero
            budget_utilization = np.where(budget_tokens > 0, tokens_out / budget_tokens, 0)
            features['budget_utilization'] = budget_utilization.values

        # Priority features
        if 'priority' in agent_data.columns:
            priorities = agent_data['priority'].fillna(0)
            features['priority'] = priorities.values

        return features


class FeatureEngineer:
    """Main feature engineering class that combines all extractors."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with all feature extractors.

        Args:
            embedding_model: Name of sentence transformer model
        """
        self.text_extractor = TextFeatureExtractor(embedding_model)
        self.metadata_extractor = MetadataFeatureExtractor()
        self.agent_extractor = AgentFeatureExtractor()

    def extract_all_features(self, data: pd.DataFrame, data_type: str = 'legal') -> Dict[str, np.ndarray]:
        """Extract all features from data.

        Args:
            data: Input DataFrame
            data_type: Type of data ('legal' or 'agent')

        Returns:
            Dictionary of all extracted features
        """
        features = {}

        if data_type == 'legal':
            # Extract text features
            if 'opinion_text' in data.columns:
                text_features = self.text_extractor.extract_text_features(data['opinion_text'].tolist())
                features.update(text_features)

            # Extract metadata features
            metadata_features = self.metadata_extractor.extract_case_metadata_features(data)
            features.update(metadata_features)

        elif data_type == 'agent':
            # Extract agent features
            agent_features = self.agent_extractor.extract_agent_features(data)
            features.update(agent_features)

        return features

    def create_feature_matrix(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine all features into a single matrix.

        Args:
            features: Dictionary of feature arrays

        Returns:
            Combined feature matrix
        """
        feature_arrays = []

        for name, array in features.items():
            if array.ndim == 1:
                # 1D array - add dimension
                feature_arrays.append(array.reshape(-1, 1))
            else:
                # Multi-dimensional array
                feature_arrays.append(array)

        if feature_arrays:
            return np.hstack(feature_arrays)
        else:
            return np.array([]).reshape(0, 0)

    def get_feature_names(self, features: Dict[str, np.ndarray]) -> List[str]:
        """Get names of all features.

        Args:
            features: Dictionary of feature arrays

        Returns:
            List of feature names
        """
        feature_names = []

        for name, array in features.items():
            if array.ndim == 1:
                feature_names.append(name)
            else:
                # Multi-dimensional features
                for i in range(array.shape[1]):
                    feature_names.append(f"{name}_{i}")

        return feature_names


# Convenience functions
def extract_legal_features(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract features from legal case data."""
    engineer = FeatureEngineer()
    return engineer.extract_all_features(data, 'legal')

def extract_agent_features(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract features from agent execution data."""
    engineer = FeatureEngineer()
    return engineer.extract_all_features(data, 'agent')

def create_feature_matrix(features: Dict[str, np.ndarray]) -> np.ndarray:
    """Create feature matrix from extracted features."""
    engineer = FeatureEngineer()
    return engineer.create_feature_matrix(features)
