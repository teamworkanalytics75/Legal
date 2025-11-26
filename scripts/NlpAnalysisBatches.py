#!/usr/bin/env python3
"""
NLP Analysis System for §1782 Caselaw Corpus

This script performs comprehensive NLP analysis on the extracted text content
in batches, discovering patterns and insights in the legal text.
"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
# Simplified readability calculation without external dependencies
def calculate_readability(text: str) -> Tuple[float, float]:
    """Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    if not sentences or not words:
        return 0.0, 0.0

    # Count syllables (simplified approximation)
    syllables = 0
    for word in words:
        if word.isalpha():
            # Simple syllable counting
            word_lower = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            prev_was_vowel = False

            for char in word_lower:
                if char in vowels:
                    if not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False

            # Handle silent 'e'
            if word_lower.endswith('e') and syllable_count > 1:
                syllable_count -= 1

            syllables += max(1, syllable_count)

    # Flesch Reading Ease
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)

    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

    # Flesch-Kincaid Grade Level
    grade_level = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59

    return max(0, min(100, flesch_score)), max(0, grade_level)

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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LegalNLPAnalyzer:
    """Comprehensive NLP analysis for legal text corpus."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.corpus_dir = Path("data/case_law/1782_discovery")
        self.analysis_results = {
            "analysis_date": datetime.now().isoformat(),
            "total_cases_analyzed": 0,
            "total_text_length": 0,
            "batches": [],
            "overall_patterns": {},
            "cumulative_stats": {}
        }

        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Legal-specific patterns
        self.legal_patterns = {
            'statutes': [
                r'28\s*u\.s\.c\.?\s*(?:\u00a7)?\s*1782',
                r'section\s*1782',
                r'(?:\u00a7)\s*1782',
                r'28\s*u\.s\.c\.?\s*(?:\u00a7)?\s*\d+',
                r'section\s*\d+',
            ],
            'courts': [
                r'(?:district|court|circuit|supreme)\s+court',
                r'(?:federal|state)\s+court',
                r'u\.s\.\s*(?:district|circuit)',
                r'court\s+of\s+(?:appeals|claims)',
            ],
            'procedures': [
                r'discovery',
                r'subpoena',
                r'deposition',
                r'evidence',
                r'arbitration',
                r'foreign\s+tribunal',
                r'international\s+proceeding',
            ],
            'parties': [
                r'(?:petitioner|respondent|plaintiff|defendant)',
                r'(?:applicant|movant)',
                r'(?:foreign|international)\s+(?:entity|corporation|company)',
            ]
        }

        # Compile regex patterns
        for category, patterns in self.legal_patterns.items():
            self.legal_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def get_cases_with_text(self) -> List[Path]:
        """Get all case files that have extracted text."""
        case_files = []

        for case_file in self.corpus_dir.glob("*.json"):
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                if case_data.get('extracted_text') and len(case_data['extracted_text'].strip()) > 100:
                    case_files.append(case_file)

            except Exception as e:
                logger.error(f"Error reading {case_file.name}: {e}")

        logger.info(f"Found {len(case_files)} cases with extractable text")
        return case_files

    def analyze_text_content(self, text: str) -> Dict[str, Any]:
        """Analyze a single text for various NLP metrics."""
        analysis = {
            'text_length': len(text),
            'word_count': len(word_tokenize(text)),
            'sentence_count': len(sent_tokenize(text)),
            'avg_sentence_length': 0,
            'readability_score': 0,
            'readability_grade': 0,
            'sentiment_scores': {},
            'legal_pattern_matches': {},
            'top_words': [],
            'named_entities': [],
            'pos_tags': {},
            'complexity_metrics': {}
        }

        if not text.strip():
            return analysis

        # Basic metrics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        analysis['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0

        # Readability
        try:
            analysis['readability_score'], analysis['readability_grade'] = calculate_readability(text)
        except:
            pass

        # Sentiment analysis
        try:
            analysis['sentiment_scores'] = self.sentiment_analyzer.polarity_scores(text)
        except:
            pass

        # Legal pattern matching
        for category, patterns in self.legal_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            analysis['legal_pattern_matches'][category] = len(matches)

        # Word frequency (excluding stop words)
        words_lower = [w.lower() for w in words if w.isalpha() and w.lower() not in self.stop_words]
        word_freq = Counter(words_lower)
        analysis['top_words'] = word_freq.most_common(20)

        # POS tagging
        try:
            pos_tags = pos_tag(words)
            pos_counter = Counter([tag for word, tag in pos_tags])
            analysis['pos_tags'] = dict(pos_counter.most_common(10))
        except:
            pass

        # Complexity metrics
        analysis['complexity_metrics'] = {
            'long_words_ratio': len([w for w in words if len(w) > 6]) / len(words) if words else 0,
            'complex_sentences_ratio': len([s for s in sentences if len(s.split()) > 20]) / len(sentences) if sentences else 0,
            'legal_terms_density': sum(analysis['legal_pattern_matches'].values()) / len(words) if words else 0
        }

        return analysis

    def analyze_batch(self, case_files: List[Path], batch_num: int) -> Dict[str, Any]:
        """Analyze a batch of case files."""
        logger.info(f"Analyzing batch {batch_num} ({len(case_files)} cases)...")

        batch_results = {
            'batch_number': batch_num,
            'cases_analyzed': len(case_files),
            'case_analyses': [],
            'batch_patterns': {},
            'batch_stats': {}
        }

        batch_text_length = 0
        all_legal_matches = defaultdict(int)
        all_top_words = Counter()
        all_sentiment_scores = []
        readability_scores = []

        for case_file in case_files:
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                text = case_data.get('extracted_text', '')
                case_name = case_data.get('caseName', case_file.stem)

                # Analyze the text
                analysis = self.analyze_text_content(text)
                analysis['case_name'] = case_name
                analysis['file_name'] = case_file.name

                batch_results['case_analyses'].append(analysis)

                # Aggregate data
                batch_text_length += analysis['text_length']
                readability_scores.append(analysis['readability_score'])

                # Aggregate legal pattern matches
                for category, count in analysis['legal_pattern_matches'].items():
                    all_legal_matches[category] += count

                # Aggregate top words
                for word, count in analysis['top_words']:
                    all_top_words[word] += count

                # Aggregate sentiment
                if analysis['sentiment_scores']:
                    all_sentiment_scores.append(analysis['sentiment_scores'])

            except Exception as e:
                logger.error(f"Error analyzing {case_file.name}: {e}")

        # Calculate batch statistics
        batch_results['batch_stats'] = {
            'total_text_length': batch_text_length,
            'avg_readability': sum(readability_scores) / len(readability_scores) if readability_scores else 0,
            'legal_pattern_totals': dict(all_legal_matches),
            'most_common_words': all_top_words.most_common(30),
            'avg_sentiment': self._calculate_avg_sentiment(all_sentiment_scores)
        }

        # Identify interesting patterns
        batch_results['batch_patterns'] = self._identify_patterns(batch_results)

        return batch_results

    def _calculate_avg_sentiment(self, sentiment_scores: List[Dict]) -> Dict[str, float]:
        """Calculate average sentiment scores."""
        if not sentiment_scores:
            return {}

        avg_sentiment = {}
        for key in ['neg', 'neu', 'pos', 'compound']:
            avg_sentiment[key] = sum(s[key] for s in sentiment_scores) / len(sentiment_scores)

        return avg_sentiment

    def _identify_patterns(self, batch_results: Dict) -> Dict[str, Any]:
        """Identify interesting patterns in the batch."""
        patterns = {
            'notable_cases': [],
            'complexity_insights': {},
            'legal_focus_areas': {},
            'language_patterns': {},
            'anomalies': []
        }

        cases = batch_results['case_analyses']

        # Find notable cases (longest, most complex, etc.)
        if cases:
            longest_case = max(cases, key=lambda x: x['text_length'])
            most_complex = max(cases, key=lambda x: x['complexity_metrics']['legal_terms_density'])

            patterns['notable_cases'] = [
                {
                    'type': 'longest_text',
                    'case': longest_case['case_name'],
                    'length': longest_case['text_length'],
                    'file': longest_case['file_name']
                },
                {
                    'type': 'most_legal_terms',
                    'case': most_complex['case_name'],
                    'density': most_complex['complexity_metrics']['legal_terms_density'],
                    'file': most_complex['file_name']
                }
            ]

        # Complexity insights
        readability_scores = [c['readability_score'] for c in cases if c['readability_score'] > 0]
        if readability_scores:
            patterns['complexity_insights'] = {
                'avg_readability': sum(readability_scores) / len(readability_scores),
                'most_readable': min(readability_scores),
                'least_readable': max(readability_scores),
                'complexity_level': 'High' if sum(readability_scores) / len(readability_scores) < 30 else 'Medium' if sum(readability_scores) / len(readability_scores) < 60 else 'Low'
            }

        # Legal focus areas
        legal_totals = batch_results['batch_stats']['legal_pattern_totals']
        if legal_totals:
            patterns['legal_focus_areas'] = {
                'most_mentioned': max(legal_totals.items(), key=lambda x: x[1]) if legal_totals else None,
                'total_legal_references': sum(legal_totals.values()),
                'categories_mentioned': len([k for k, v in legal_totals.items() if v > 0])
            }

        # Language patterns
        most_common_words = batch_results['batch_stats']['most_common_words']
        if most_common_words:
            patterns['language_patterns'] = {
                'top_legal_terms': [word for word, count in most_common_words[:10] if any(term in word.lower() for term in ['court', 'discovery', 'foreign', 'evidence', 'tribunal'])],
                'vocabulary_diversity': len(set(word for word, count in most_common_words)),
                'most_frequent_word': most_common_words[0] if most_common_words else None
            }

        return patterns

    def run_batch_analysis(self) -> None:
        """Run NLP analysis in batches."""
        logger.info("="*80)
        logger.info("STARTING BATCH NLP ANALYSIS")
        logger.info("="*80)

        case_files = self.get_cases_with_text()
        total_cases = len(case_files)

        logger.info(f"Found {total_cases} cases with text content")
        logger.info(f"Processing in batches of {self.batch_size}")

        # Process in batches
        for i in range(0, total_cases, self.batch_size):
            batch_files = case_files[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING BATCH {batch_num}")
            logger.info(f"{'='*60}")

            # Analyze batch
            batch_results = self.analyze_batch(batch_files, batch_num)
            self.analysis_results['batches'].append(batch_results)

            # Update cumulative stats
            self.analysis_results['total_cases_analyzed'] += batch_results['cases_analyzed']
            self.analysis_results['total_text_length'] += batch_results['batch_stats']['total_text_length']

            # Report batch findings
            self._report_batch_findings(batch_results, batch_num)

            # Update overall patterns
            self._update_overall_patterns(batch_results)

        # Generate final report
        self._generate_final_report()

    def _report_batch_findings(self, batch_results: Dict, batch_num: int) -> None:
        """Report interesting findings from a batch."""
        logger.info(f"\n🔍 BATCH {batch_num} FINDINGS:")
        logger.info(f"Cases analyzed: {batch_results['cases_analyzed']}")
        logger.info(f"Total text length: {batch_results['batch_stats']['total_text_length']:,} characters")

        patterns = batch_results['batch_patterns']

        # Notable cases
        if patterns['notable_cases']:
            logger.info(f"\n📋 Notable Cases:")
            for case in patterns['notable_cases']:
                logger.info(f"  {case['type']}: {case['case']} ({case.get('length', case.get('density', 'N/A'))})")

        # Complexity insights
        if patterns['complexity_insights']:
            complexity = patterns['complexity_insights']
            logger.info(f"\n📊 Complexity Insights:")
            logger.info(f"  Average readability: {complexity['avg_readability']:.1f}")
            logger.info(f"  Complexity level: {complexity['complexity_level']}")

        # Legal focus areas
        if patterns['legal_focus_areas']:
            legal = patterns['legal_focus_areas']
            logger.info(f"\n⚖️ Legal Focus Areas:")
            logger.info(f"  Total legal references: {legal['total_legal_references']}")
            logger.info(f"  Categories mentioned: {legal['categories_mentioned']}")
            if legal['most_mentioned']:
                logger.info(f"  Most mentioned: {legal['most_mentioned'][0]} ({legal['most_mentioned'][1]} times)")

        # Language patterns
        if patterns['language_patterns']:
            lang = patterns['language_patterns']
            logger.info(f"\n📝 Language Patterns:")
            logger.info(f"  Vocabulary diversity: {lang['vocabulary_diversity']} unique words")
            if lang['top_legal_terms']:
                logger.info(f"  Top legal terms: {', '.join(lang['top_legal_terms'][:5])}")
            if lang['most_frequent_word']:
                logger.info(f"  Most frequent word: '{lang['most_frequent_word'][0]}' ({lang['most_frequent_word'][1]} times)")

    def _update_overall_patterns(self, batch_results: Dict) -> None:
        """Update overall patterns with batch results."""
        # This will be called after each batch to maintain running totals
        pass

    def _generate_final_report(self) -> None:
        """Generate comprehensive final report."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING FINAL NLP ANALYSIS REPORT")
        logger.info("="*80)

        # Calculate overall statistics
        total_batches = len(self.analysis_results['batches'])
        total_cases = self.analysis_results['total_cases_analyzed']
        total_text = self.analysis_results['total_text_length']

        # Aggregate all patterns
        all_legal_matches = defaultdict(int)
        all_words = Counter()
        all_readability = []

        for batch in self.analysis_results['batches']:
            for category, count in batch['batch_stats']['legal_pattern_totals'].items():
                all_legal_matches[category] += count

            for word, count in batch['batch_stats']['most_common_words']:
                all_words[word] += count

            if batch['batch_stats']['avg_readability'] > 0:
                all_readability.append(batch['batch_stats']['avg_readability'])

        # Generate report
        report_content = f"""# 🧠 NLP Analysis Report for §1782 Caselaw Corpus

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Analysis Summary

| Metric | Value |
|--------|-------|
| **Total Cases Analyzed** | {total_cases} |
| **Total Text Length** | {total_text:,} characters |
| **Number of Batches** | {total_batches} |
| **Average Text per Case** | {total_text/total_cases:,.0f} characters |

## ⚖️ Legal Pattern Analysis

### Most Referenced Legal Categories
"""

        for category, count in sorted(all_legal_matches.items(), key=lambda x: x[1], reverse=True):
            report_content += f"- **{category.title()}**: {count} references\n"

        report_content += f"""
## 📝 Language Analysis

### Most Common Words Across All Cases
"""

        for word, count in all_words.most_common(20):
            report_content += f"- **{word}**: {count} occurrences\n"

        report_content += f"""
## 📊 Readability Analysis

- **Average Readability Score**: {sum(all_readability)/len(all_readability):.1f} (out of 100)
- **Readability Level**: {'High' if sum(all_readability)/len(all_readability) > 60 else 'Medium' if sum(all_readability)/len(all_readability) > 30 else 'Low'}

## 🔍 Key Insights

1. **Legal Focus**: The corpus shows strong focus on {max(all_legal_matches.items(), key=lambda x: x[1])[0]} with {max(all_legal_matches.items(), key=lambda x: x[1])[1]} references
2. **Text Complexity**: Average readability suggests {'accessible' if sum(all_readability)/len(all_readability) > 60 else 'moderate' if sum(all_readability)/len(all_readability) > 30 else 'complex'} legal writing
3. **Vocabulary**: {len(all_words)} unique words across the corpus
4. **Content Density**: {sum(all_legal_matches.values())} total legal references across {total_cases} cases

## 📋 Batch-by-Batch Analysis

"""

        for i, batch in enumerate(self.analysis_results['batches'], 1):
            report_content += f"""### Batch {i}
- **Cases**: {batch['cases_analyzed']}
- **Text Length**: {batch['batch_stats']['total_text_length']:,} characters
- **Legal References**: {sum(batch['batch_stats']['legal_pattern_totals'].values())}
- **Notable Cases**: {len(batch['batch_patterns']['notable_cases'])}

"""

        report_content += f"""
## 🎯 Recommendations

1. **Focus Areas**: Prioritize analysis of cases with high {max(all_legal_matches.items(), key=lambda x: x[1])[0]} references
2. **Complexity Review**: Cases with low readability scores may need special attention
3. **Pattern Recognition**: Look for recurring themes in the most common words
4. **Comparative Analysis**: Compare batches to identify temporal or thematic patterns

---

**This analysis provides a comprehensive overview of the linguistic and legal patterns in the §1782 caselaw corpus.**
"""

        # Save report
        report_path = Path("data/case_law/nlp_analysis_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # Save detailed results
        results_path = Path("data/case_law/nlp_analysis_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ NLP analysis report saved to: {report_path}")
        logger.info(f"✓ Detailed results saved to: {results_path}")


def main():
    """Main entry point."""
    logger.info("Starting comprehensive NLP analysis for §1782 caselaw corpus...")

    # Create analyzer with batch size of 20
    analyzer = LegalNLPAnalyzer(batch_size=64)

    # Run batch analysis
    analyzer.run_batch_analysis()

    logger.info("\n🎉 NLP analysis completed successfully!")
    logger.info("Check data/case_law/nlp_analysis_report.md for detailed results")


if __name__ == "__main__":
    main()
