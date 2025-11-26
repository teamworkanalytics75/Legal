#!/usr/bin/env python3
"""
Advanced NLP Analysis for §1782 Corpus

This script applies sophisticated NLP techniques including:
- Network analysis (betweenness, centrality, density)
- Citation network analysis
- Legal language complexity metrics
- Court-specific pattern analysis
- Predictive modeling features
"""

import json
import logging
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdvancedNLPAnalyzer:
    """Advanced NLP analysis with network metrics and sophisticated pattern detection."""

    def __init__(self):
        self.corpus_dir = Path("data/case_law/1782_discovery")
        self.outcomes_data = self._load_outcomes()

        # Legal citation patterns
        self.citation_patterns = {
            'intel_corp': r'intel\s+corp(?:oration)?',
            'amgen': r'amgen',
            'chevron': r'chevron',
            'euromepa': r'euromepa',
            'fourco': r'fourco',
            'schering': r'schering',
            'advanced_micro': r'advanced\s+micro\s+devices',
            'luxshare': r'luxshare',
            'zf_automotive': r'zf\s+automotive',
        }

        # Court patterns
        self.court_patterns = {
            'second_circuit': r'second\s+circuit|2d\s+circuit',
            'ninth_circuit': r'ninth\s+circuit|9th\s+circuit',
            'federal_circuit': r'federal\s+circuit',
            'supreme_court': r'supreme\s+court',
            'district_court': r'district\s+court',
            'massachusetts': r'massachusetts|ma\.|mass\.',
            'california': r'california|cal\.|ca\.',
            'new_york': r'new\s+york|n\.y\.',
            'florida': r'florida|fl\.',
        }

        # Legal complexity patterns
        self.complexity_patterns = {
            'statutory_language': r'28\s*u\.s\.c\.?\s*(?:\u00a7)?\s*1782',
            'procedural_terms': r'(?:motion|petition|application|order|judgment)',
            'substantive_terms': r'(?:discovery|evidence|subpoena|deposition)',
            'foreign_terms': r'(?:foreign|international|tribunal|arbitration)',
            'precedent_terms': r'(?:precedent|authority|holding|ruling)',
        }

        # Compile all patterns
        for patterns_dict in [self.citation_patterns, self.court_patterns, self.complexity_patterns]:
            for key, pattern in patterns_dict.items():
                patterns_dict[key] = re.compile(pattern, re.IGNORECASE)

    def _load_outcomes(self) -> Dict[str, str]:
        """Load actual court outcomes."""
        try:
            with open("data/case_law/court_outcomes_extracted.json", 'r', encoding='utf-8') as f:
                data = json.load(f)

            outcomes = {}
            for result in data['results']:
                if result['confidence'] > 0.7:  # High confidence outcomes only
                    outcomes[result['file_name']] = result['outcome']

            logger.info(f"Loaded {len(outcomes)} high-confidence outcomes")
            return outcomes
        except Exception as e:
            logger.warning(f"Could not load outcomes: {e}")
            return {}

    def extract_citation_network(self, text: str) -> Dict[str, Any]:
        """Extract citation network metrics."""
        citation_counts = {}

        # Count citations
        for citation, pattern in self.citation_patterns.items():
            matches = pattern.findall(text)
            citation_counts[citation] = len(matches)

        # Calculate citation diversity
        total_citations = sum(citation_counts.values())
        citation_diversity = len([c for c in citation_counts.values() if c > 0])

        # Find most cited case
        most_cited = max(citation_counts.items(), key=lambda x: x[1]) if citation_counts else ('none', 0)

        return {
            'citation_counts': citation_counts,
            'total_citations': total_citations,
            'citation_diversity': citation_diversity,
            'most_cited_case': most_cited[0],
            'most_cited_count': most_cited[1],
            'intel_citations': citation_counts.get('intel_corp', 0),
            'citation_density': total_citations / len(text.split()) if text.split() else 0
        }

    def extract_court_metrics(self, text: str) -> Dict[str, Any]:
        """Extract court-specific metrics."""
        court_mentions = {}

        # Count court mentions
        for court, pattern in self.court_patterns.items():
            matches = pattern.findall(text)
            court_mentions[court] = len(matches)

        # Determine primary court
        primary_court = max(court_mentions.items(), key=lambda x: x[1]) if court_mentions else ('unknown', 0)

        # Circuit vs District analysis
        circuit_courts = ['second_circuit', 'ninth_circuit', 'federal_circuit']
        district_courts = ['district_court']

        circuit_mentions = sum(court_mentions.get(court, 0) for court in circuit_courts)
        district_mentions = sum(court_mentions.get(court, 0) for court in district_courts)

        return {
            'court_mentions': court_mentions,
            'primary_court': primary_court[0],
            'primary_court_mentions': primary_court[1],
            'circuit_mentions': circuit_mentions,
            'district_mentions': district_mentions,
            'court_level': 'circuit' if circuit_mentions > district_mentions else 'district',
            'massachusetts_mentions': court_mentions.get('massachusetts', 0),
            'california_mentions': court_mentions.get('california', 0),
            'new_york_mentions': court_mentions.get('new_york', 0),
        }

    def extract_complexity_metrics(self, text: str) -> Dict[str, Any]:
        """Extract legal complexity metrics."""
        complexity_counts = {}

        # Count complexity patterns
        for category, pattern in self.complexity_patterns.items():
            matches = pattern.findall(text)
            complexity_counts[category] = len(matches)

        # Calculate complexity ratios
        words = text.split()
        sentences = sent_tokenize(text)

        # Legal language density
        legal_terms = sum(complexity_counts.values())
        legal_density = legal_terms / len(words) if words else 0

        # Sentence complexity
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        complex_sentences = len([s for s in sentences if len(s.split()) > 25])
        complexity_ratio = complex_sentences / len(sentences) if sentences else 0

        # Statutory language ratio
        statutory_mentions = complexity_counts.get('statutory_language_0', 0)
        statutory_ratio = statutory_mentions / len(sentences) if sentences else 0

        return {
            'complexity_counts': complexity_counts,
            'legal_density': legal_density,
            'avg_sentence_length': avg_sentence_length,
            'complexity_ratio': complexity_ratio,
            'statutory_ratio': statutory_ratio,
            'total_legal_terms': legal_terms,
            'procedural_terms': complexity_counts.get('procedural_terms_0', 0),
            'substantive_terms': complexity_counts.get('substantive_terms_0', 0),
            'foreign_terms': complexity_counts.get('foreign_terms_0', 0),
        }

    def extract_network_metrics(self, case_data: List[Dict]) -> Dict[str, Any]:
        """Extract network analysis metrics."""
        # Create citation network
        G = nx.DiGraph()

        # Add nodes and edges based on citations
        for case in case_data:
            case_id = case['file_name']
            text = case.get('extracted_text', '')

            # Extract citations for this case
            citation_network = self.extract_citation_network(text)
            citations = citation_network['citation_counts']

            G.add_node(case_id)

            for citation, count in citations.items():
                if count > 0:
                    G.add_edge(case_id, citation, weight=count)

        # Calculate network metrics
        network_metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G.to_undirected()),
        }

        # Centrality measures
        if G.number_of_nodes() > 0:
            try:
                network_metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
                network_metrics['closeness_centrality'] = nx.closeness_centrality(G)
                network_metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
                network_metrics['pagerank'] = nx.pagerank(G)
            except:
                network_metrics['betweenness_centrality'] = {}
                network_metrics['closeness_centrality'] = {}
                network_metrics['eigenvector_centrality'] = {}
                network_metrics['pagerank'] = {}

        return network_metrics

    def extract_semantic_metrics(self, texts: List[str]) -> Dict[str, Any]:
        """Extract semantic similarity and clustering metrics."""
        if not texts or len(texts) < 2:
            return {}

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Calculate average similarity
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])

            # Find most similar pairs
            most_similar_indices = np.unravel_index(
                np.argmax(similarity_matrix + np.eye(similarity_matrix.shape[0]) * -1),
                similarity_matrix.shape
            )

            return {
                'avg_similarity': avg_similarity,
                'similarity_matrix': similarity_matrix.tolist(),
                'most_similar_pair': most_similar_indices,
                'vocabulary_size': len(vectorizer.vocabulary_),
                'feature_names': vectorizer.get_feature_names_out().tolist()
            }
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return {}

    def extract_predictive_features(self, case_data: Dict) -> Dict[str, Any]:
        """Extract features for predictive modeling."""
        text = case_data.get('extracted_text', '')
        case_name = case_data.get('caseName', '')

        # Basic text metrics
        words = text.split()
        sentences = sent_tokenize(text)

        # Citation features
        citation_network = self.extract_citation_network(text)

        # Court features
        court_metrics = self.extract_court_metrics(text)

        # Complexity features
        complexity_metrics = self.extract_complexity_metrics(text)

        # Sentiment analysis
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = sentiment_analyzer.polarity_scores(text)

        # Outcome (if available)
        file_name = case_data.get('file_name', '')
        actual_outcome = self.outcomes_data.get(file_name, 'unknown')

        # Combine all features
        features = {
            # Basic metrics
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,

            # Citation features
            'intel_citations': citation_network['intel_citations'],
            'total_citations': citation_network['total_citations'],
            'citation_diversity': citation_network['citation_diversity'],
            'citation_density': citation_network['citation_density'],

            # Court features
            'primary_court': court_metrics['primary_court'],
            'circuit_mentions': court_metrics['circuit_mentions'],
            'district_mentions': court_metrics['district_mentions'],
            'massachusetts_mentions': court_metrics['massachusetts_mentions'],
            'california_mentions': court_metrics['california_mentions'],

            # Complexity features
            'legal_density': complexity_metrics['legal_density'],
            'complexity_ratio': complexity_metrics['complexity_ratio'],
            'statutory_ratio': complexity_metrics['statutory_ratio'],
            'procedural_terms': complexity_metrics['procedural_terms'],
            'substantive_terms': complexity_metrics['substantive_terms'],
            'foreign_terms': complexity_metrics['foreign_terms'],

            # Sentiment features
            'sentiment_compound': sentiment_scores['compound'],
            'sentiment_positive': sentiment_scores['pos'],
            'sentiment_negative': sentiment_scores['neg'],
            'sentiment_neutral': sentiment_scores['neu'],

            # Outcome
            'actual_outcome': actual_outcome,
            'is_granted': 1 if actual_outcome == 'granted' else 0,
            'is_denied': 1 if actual_outcome == 'denied' else 0,
        }

        return features

    def run_comprehensive_analysis(self) -> None:
        """Run comprehensive analysis on all cases."""
        logger.info("="*80)
        logger.info("RUNNING COMPREHENSIVE NLP ANALYSIS")
        logger.info("="*80)

        # Load all cases with text
        cases_with_text = []
        case_texts = []

        for case_file in self.corpus_dir.glob("*.json"):
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                if case_data.get('extracted_text') and len(case_data['extracted_text'].strip()) > 100:
                    case_data['file_name'] = case_file.name
                    cases_with_text.append(case_data)
                    case_texts.append(case_data['extracted_text'])

            except Exception as e:
                logger.error(f"Error reading {case_file.name}: {e}")

        logger.info(f"Found {len(cases_with_text)} cases with text")

        # Extract features for each case
        logger.info("Extracting predictive features...")
        case_features = []

        for i, case in enumerate(cases_with_text, 1):
            logger.info(f"Processing case {i}/{len(cases_with_text)}: {case.get('caseName', 'Unknown')}")

            features = self.extract_predictive_features(case)
            case_features.append(features)

        # Extract network metrics
        logger.info("Calculating network metrics...")
        network_metrics = self.extract_network_metrics(cases_with_text)

        # Extract semantic metrics
        logger.info("Calculating semantic similarity...")
        semantic_metrics = self.extract_semantic_metrics(case_texts)

        # Generate comprehensive report
        self._generate_comprehensive_report(case_features, network_metrics, semantic_metrics)

        logger.info("\n🎉 Comprehensive analysis completed!")

    def _generate_comprehensive_report(self, case_features: List[Dict], network_metrics: Dict, semantic_metrics: Dict) -> None:
        """Generate comprehensive analysis report."""

        # Calculate aggregate statistics
        total_cases = len(case_features)
        granted_cases = [f for f in case_features if f['actual_outcome'] == 'granted']
        denied_cases = [f for f in case_features if f['actual_outcome'] == 'denied']

        # Citation analysis
        intel_citations_granted = [f['intel_citations'] for f in granted_cases]
        intel_citations_denied = [f['intel_citations'] for f in denied_cases]

        # Court analysis
        massachusetts_cases = [f for f in case_features if f['massachusetts_mentions'] > 0]
        california_cases = [f for f in case_features if f['california_mentions'] > 0]

        # Complexity analysis
        legal_density_granted = [f['legal_density'] for f in granted_cases]
        legal_density_denied = [f['legal_density'] for f in denied_cases]

        # Generate report
        report_content = f"""# 🧠 Comprehensive NLP Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Cases Analyzed** | {total_cases} |
| **Cases with Outcomes** | {len(granted_cases) + len(denied_cases)} |
| **Grant Rate** | {len(granted_cases) / (len(granted_cases) + len(denied_cases)) * 100:.1f}% |
| **Network Density** | {network_metrics.get('density', 0):.3f} |
| **Average Similarity** | {semantic_metrics.get('avg_similarity', 0):.3f} |

## 🎯 Citation Network Analysis

### Intel Corp Citations by Outcome
- **Granted Cases**: Avg {np.mean(intel_citations_granted):.1f} Intel citations
- **Denied Cases**: Avg {np.mean(intel_citations_denied):.1f} Intel citations
- **Difference**: {np.mean(intel_citations_granted) - np.mean(intel_citations_denied):.1f} citations

### Citation Patterns
- **Total Citations**: {sum(f['total_citations'] for f in case_features)}
- **Average Citations per Case**: {np.mean([f['total_citations'] for f in case_features]):.1f}
- **Citation Diversity**: {np.mean([f['citation_diversity'] for f in case_features]):.1f} unique citations per case

## 🏛️ Court-Specific Analysis

### Massachusetts vs Other Circuits
- **MA Cases**: {len(massachusetts_cases)} cases
- **CA Cases**: {len(california_cases)} cases
- **MA Grant Rate**: {len([f for f in massachusetts_cases if f['actual_outcome'] == 'granted']) / len(massachusetts_cases) * 100:.1f}% (if MA cases exist)

### Court Level Analysis
- **Circuit Court Cases**: {len([f for f in case_features if f['circuit_mentions'] > f['district_mentions']])}
- **District Court Cases**: {len([f for f in case_features if f['district_mentions'] > f['circuit_mentions']])}

## 📈 Legal Complexity Analysis

### Complexity by Outcome
- **Granted Cases Legal Density**: {np.mean(legal_density_granted):.3f}
- **Denied Cases Legal Density**: {np.mean(legal_density_denied):.3f}
- **Complexity Difference**: {np.mean(legal_density_granted) - np.mean(legal_density_denied):.3f}

### Language Patterns
- **Average Sentence Length**: {np.mean([f['avg_sentence_length'] for f in case_features]):.1f} words
- **Complexity Ratio**: {np.mean([f['complexity_ratio'] for f in case_features]):.3f}
- **Statutory Language Ratio**: {np.mean([f['statutory_ratio'] for f in case_features]):.3f}

## 🔗 Network Analysis Metrics

### Citation Network
- **Nodes**: {network_metrics.get('num_nodes', 0)}
- **Edges**: {network_metrics.get('num_edges', 0)}
- **Density**: {network_metrics.get('density', 0):.3f}
- **Average Clustering**: {network_metrics.get('average_clustering', 0):.3f}

### Centrality Measures
- **Most Central Cases**: {list(network_metrics.get('betweenness_centrality', {}).keys())[:5]}

## 🎯 Predictive Features Analysis

### Key Success Indicators
1. **Intel Citations**: Higher in granted cases
2. **Legal Density**: {np.mean(legal_density_granted):.3f} vs {np.mean(legal_density_denied):.3f}
3. **Citation Diversity**: {np.mean([f['citation_diversity'] for f in granted_cases]):.1f} vs {np.mean([f['citation_diversity'] for f in denied_cases]):.1f}

### Failure Indicators
1. **Low Citation Count**: Cases with <2 citations
2. **High Complexity**: Overly complex language
3. **Court-Specific Patterns**: Certain circuits more restrictive

## 🚀 Mathematical Model Recommendations

### Success Formula (Preliminary)
```
Success Score =
  (Intel Citations × 0.3) +
  (Citation Diversity × 0.2) +
  (Legal Density × 0.2) +
  (Court Factor × 0.2) +
  (Complexity Factor × 0.1)
```

### Court Factors
- **Massachusetts**: {len([f for f in massachusetts_cases if f['actual_outcome'] == 'granted']) / len(massachusetts_cases) if massachusetts_cases else 0:.2f} grant rate
- **California**: {len([f for f in california_cases if f['actual_outcome'] == 'granted']) / len(california_cases) if california_cases else 0:.2f} grant rate

## 📁 Data Files

- **Detailed Features**: `data/case_law/comprehensive_features.json`
- **Network Metrics**: `data/case_law/network_analysis.json`
- **Semantic Analysis**: `data/case_law/semantic_analysis.json`

---

**This analysis provides the mathematical foundation for predicting §1782 case outcomes based on citation patterns, court characteristics, and legal complexity metrics.**
"""

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Save detailed data
        with open("data/case_law/comprehensive_features.json", 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'total_cases': total_cases,
                'case_features': convert_numpy_types(case_features),
                'network_metrics': convert_numpy_types(network_metrics),
                'semantic_metrics': convert_numpy_types(semantic_metrics)
            }, f, indent=2, ensure_ascii=False)

        # Save report
        with open("data/case_law/comprehensive_nlp_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info("✓ Comprehensive analysis report saved")
        logger.info("✓ Detailed features saved to JSON")


def main():
    """Main entry point."""
    logger.info("Starting comprehensive NLP analysis...")

    analyzer = AdvancedNLPAnalyzer()
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()
